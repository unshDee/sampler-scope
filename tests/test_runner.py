import asyncio
from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from preference_elicitation.config import load_experiment
from preference_elicitation.runner import execute_paid_run


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "configs" / "pilot.yaml"


class FakeResponses:
    def __init__(self, failures=0, partial_usage=False, started=None, release=None):
        self.calls = []
        self.failures = failures
        self.partial_usage = partial_usage
        self.started = started
        self.release = release

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) <= self.failures:
            raise RuntimeError("temporary test failure")
        if self.started is not None:
            self.started.set()
            await self.release.wait()
        usage = SimpleNamespace(
            input_tokens=100,
            input_tokens_details=SimpleNamespace(cached_tokens=0, cache_write_tokens=0),
            output_tokens=None if self.partial_usage else 2,
            output_tokens_details=SimpleNamespace(reasoning_tokens=0),
        )
        return SimpleNamespace(
            id=f"response-{len(self.calls)}",
            model=kwargs["model"],
            output_text="100",
            system_fingerprint="test-fingerprint",
            usage=usage,
        )


class FakeClient:
    def __init__(self, failures=0, partial_usage=False, started=None, release=None):
        self.responses = FakeResponses(
            failures=failures,
            partial_usage=partial_usage,
            started=started,
            release=release,
        )


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_paid_guard_prevents_calls(self):
        client = FakeClient()
        with self.assertRaises(PermissionError):
            await execute_paid_run(
                load_experiment(PILOT),
                confirm_paid=False,
                client=client,
                max_calls=1,
            )
        self.assertEqual(client.responses.calls, [])

    async def test_fake_smoke_logs_full_record_and_resume_uses_next_task(self):
        config = load_experiment(PILOT)
        client = FakeClient()
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "pilot.jsonl"
            costs = Path(directory) / "project-costs.jsonl"
            first = await execute_paid_run(
                config,
                confirm_paid=True,
                raw_path=raw,
                max_calls=1,
                max_attempts=1,
                client=client,
                project_cost_path=costs,
            )
            second = await execute_paid_run(
                config,
                confirm_paid=True,
                raw_path=raw,
                max_calls=1,
                max_attempts=1,
                client=client,
                project_cost_path=costs,
            )
            rows = [
                json.loads(line)
                for line in raw.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(first["completed_now"], 1)
        self.assertEqual(first["remaining_logical_calls"], 419)
        self.assertEqual(
            first["remaining_plan"]["tokens_and_cost"]["total_estimated_input_tokens"],
            49363,
        )
        self.assertEqual(second["already_completed"], 1)
        self.assertEqual(second["remaining_logical_calls"], 418)
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(rows[0]["task_id"], rows[1]["task_id"])
        self.assertIn("Return exactly one integer", rows[0]["prompt"])
        self.assertEqual(rows[0]["parse_status"], "ok")
        self.assertEqual(rows[0]["input_tokens"], 100)
        self.assertEqual(len(client.responses.calls), 2)
        self.assertEqual(client.responses.calls[0]["reasoning"], {"effort": "none"})
        self.assertEqual(client.responses.calls[0]["service_tier"], "default")
        self.assertEqual(client.responses.calls[0]["max_output_tokens"], 16)

    async def test_retry_is_bounded_and_recorded(self):
        config = load_experiment(PILOT)
        client = FakeClient(failures=1)
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "pilot.jsonl"
            result = await execute_paid_run(
                config,
                confirm_paid=True,
                raw_path=raw,
                max_calls=1,
                max_attempts=2,
                client=client,
                project_cost_path=Path(directory) / "project-costs.jsonl",
            )
            row = json.loads(raw.read_text(encoding="utf-8"))
        self.assertEqual(result["completed_now"], 1)
        self.assertEqual(len(client.responses.calls), 2)
        self.assertEqual(row["attempts"], 2)

    async def test_budget_reservation_stops_before_request(self):
        config = replace(load_experiment(PILOT), budget_usd=0.000000001)
        client = FakeClient()
        with tempfile.TemporaryDirectory() as directory:
            result = await execute_paid_run(
                config,
                confirm_paid=True,
                raw_path=Path(directory) / "pilot.jsonl",
                max_calls=1,
                max_attempts=1,
                client=client,
                project_cost_path=Path(directory) / "project-costs.jsonl",
            )
        self.assertTrue(result["stopped_for_budget"])
        self.assertEqual(client.responses.calls, [])

    async def test_partial_usage_keeps_conservative_reservation(self):
        config = load_experiment(PILOT)
        client = FakeClient(partial_usage=True)
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "pilot.jsonl"
            await execute_paid_run(
                config,
                confirm_paid=True,
                raw_path=raw,
                max_calls=1,
                max_attempts=1,
                client=client,
                project_cost_path=Path(directory) / "project-costs.jsonl",
            )
            row = json.loads(raw.read_text(encoding="utf-8"))
        self.assertIsNone(row["input_tokens"])
        self.assertIn("usage was absent", row["cost_basis"])
        self.assertGreater(Decimal(row["incremental_cost_usd"]), Decimal("0.0002"))

    async def test_project_budget_is_shared_across_raw_runs(self):
        config = replace(load_experiment(PILOT), budget_usd=0.00026)
        client = FakeClient()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            costs = root / "project-costs.jsonl"
            first = await execute_paid_run(
                config,
                confirm_paid=True,
                raw_path=root / "smoke.jsonl",
                max_calls=1,
                max_attempts=1,
                client=client,
                project_cost_path=costs,
            )
            second = await execute_paid_run(
                config,
                confirm_paid=True,
                raw_path=root / "main.jsonl",
                max_calls=1,
                max_attempts=1,
                client=client,
                project_cost_path=costs,
            )
        self.assertEqual(first["completed_now"], 1)
        self.assertTrue(second["stopped_for_budget"])
        self.assertEqual(len(client.responses.calls), 1)

    async def test_second_paid_runner_stops_before_request(self):
        config = load_experiment(PILOT)
        started = asyncio.Event()
        release = asyncio.Event()
        first_client = FakeClient(started=started, release=release)
        second_client = FakeClient()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            costs = root / "project-costs.jsonl"
            first_task = asyncio.create_task(
                execute_paid_run(
                    config,
                    confirm_paid=True,
                    raw_path=root / "first.jsonl",
                    max_calls=1,
                    max_attempts=1,
                    client=first_client,
                    project_cost_path=costs,
                )
            )
            await started.wait()
            with self.assertRaisesRegex(RuntimeError, "another paid runner"):
                await execute_paid_run(
                    config,
                    confirm_paid=True,
                    raw_path=root / "second.jsonl",
                    max_calls=1,
                    max_attempts=1,
                    client=second_client,
                    project_cost_path=costs,
                )
            release.set()
            await first_task
        self.assertEqual(len(first_client.responses.calls), 1)
        self.assertEqual(second_client.responses.calls, [])


if __name__ == "__main__":
    unittest.main()
