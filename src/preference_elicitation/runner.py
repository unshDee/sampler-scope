import asyncio
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
from functools import wraps
from typing import Any
from uuid import uuid4

from .config import ExperimentConfig, ModelConfig
from .costs import (
    BudgetExceeded,
    BudgetLedger,
    ModelRates,
    Usage,
    calculate_cost,
    reservation_cost,
)
from .parsing import parse_response
from .prompts import render_prompt


@dataclass(frozen=True)
class PlannedCall:
    task_id: str
    model: str
    scenario_id: str
    semantic_x: str
    semantic_y: str
    displayed_x: str
    displayed_y: str
    order: str
    mirrored: bool
    frame: str
    method: str
    repetition: int
    ladder_level: int | None
    donation_usd: float | None
    prompt_version: str
    prompt: str
    prompt_hash: str
    estimated_input_tokens: int
    estimated_output_tokens: int


def _task_id(
    config: ExperimentConfig, dimensions: dict[str, Any], prompt_hash: str
) -> str:
    # Prompt text is part of the id, so an edited prompt cannot reuse an old result.
    payload = json.dumps(
        {"config": config.digest, "dimensions": dimensions, "prompt_hash": prompt_hash},
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()[:24]


def _estimated_input_tokens(config: ExperimentConfig, prompt: str) -> int:
    estimate = config.token_estimate
    return math.ceil(
        len(prompt.encode("utf-8")) / estimate["chars_per_token"]
    ) + math.ceil(estimate["request_overhead"])


def build_plan(config: ExperimentConfig) -> list[PlannedCall]:
    calls: list[PlannedCall] = []
    estimated_output = math.ceil(config.token_estimate["output_per_call"])
    for model in config.models:
        for scenario in config.scenarios:
            for frame in config.frames:
                for order in config.orders:
                    displayed_x, displayed_y, mirrored = scenario.displayed(order)
                    for method, settings in config.methods.items():
                        repetitions = int(settings["repetitions"])
                        levels: list[float | None]
                        if method == "tradeoff":
                            levels = [
                                float(value)
                                for value in settings["donation_amounts_usd"]
                            ]
                        else:
                            levels = [None]
                        for repetition in range(1, repetitions + 1):
                            for level_index, donation in enumerate(levels, 1):
                                prompt = render_prompt(
                                    version=config.prompt_version,
                                    frame=frame,
                                    method=method,
                                    displayed_x=displayed_x,
                                    displayed_y=displayed_y,
                                    donation_usd=donation,
                                )
                                prompt_hash = sha256(prompt.encode("utf-8")).hexdigest()
                                dimensions = {
                                    "model": model.id,
                                    "scenario": scenario.id,
                                    "frame": frame,
                                    "order": order,
                                    "method": method,
                                    "repetition": repetition,
                                    "ladder_level": level_index
                                    if donation is not None
                                    else None,
                                    "donation_usd": donation,
                                }
                                calls.append(
                                    PlannedCall(
                                        task_id=_task_id(
                                            config, dimensions, prompt_hash
                                        ),
                                        model=model.id,
                                        scenario_id=scenario.id,
                                        semantic_x=scenario.semantic_x,
                                        semantic_y=scenario.semantic_y,
                                        displayed_x=displayed_x,
                                        displayed_y=displayed_y,
                                        order=order,
                                        mirrored=mirrored,
                                        frame=frame,
                                        method=method,
                                        repetition=repetition,
                                        ladder_level=level_index
                                        if donation is not None
                                        else None,
                                        donation_usd=donation,
                                        prompt_version=config.prompt_version,
                                        prompt=prompt,
                                        prompt_hash=prompt_hash,
                                        estimated_input_tokens=_estimated_input_tokens(
                                            config, prompt
                                        ),
                                        estimated_output_tokens=estimated_output,
                                    )
                                )
    rng = random.Random(config.seed)
    rng.shuffle(calls)
    return calls


def _money(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.000000001")), "f")


def _counter(calls: list[PlannedCall], attribute: str) -> dict[str, int]:
    counts = Counter(str(getattr(call, attribute)) for call in calls)
    return dict(sorted(counts.items()))


def plan_summary(
    config: ExperimentConfig, calls: list[PlannedCall] | None = None
) -> dict[str, Any]:
    calls = calls if calls is not None else build_plan(config)
    model_settings = {model.id: model for model in config.models}
    model_costs: dict[str, dict[str, Any]] = {}
    for model_id, model in model_settings.items():
        selected = [call for call in calls if call.model == model_id]
        rates = ModelRates.from_mapping(model.rates)
        input_tokens = sum(call.estimated_input_tokens for call in selected)
        output_tokens = sum(call.estimated_output_tokens for call in selected)
        output_cap = len(selected) * config.max_output_tokens
        expected = calculate_cost(
            Usage(input_tokens=input_tokens, output_tokens=output_tokens), rates
        )
        capped = calculate_cost(
            Usage(input_tokens=input_tokens, output_tokens=output_cap), rates
        )
        one_attempt_reservations = sum(
            (
                reservation_cost(call.prompt, config.max_output_tokens, rates)
                for call in selected
            ),
            Decimal(0),
        )
        model_costs[model_id] = {
            "calls": len(selected),
            "estimated_input_tokens": input_tokens,
            "estimated_output_tokens": output_tokens,
            "maximum_output_tokens": output_cap,
            "projected_cost_usd": _money(expected),
            "projected_cost_at_output_cap_usd": _money(capped),
            "hard_reservation_one_attempt_each_usd": _money(one_attempt_reservations),
            "hard_reservation_all_attempts_usd": _money(
                one_attempt_reservations * (config.max_retries + 1)
            ),
            "reasoning_effort": model.reasoning_effort,
            "rates_usd_per_million": model.rates,
        }

    combinations = Counter((call.model, call.method) for call in calls)
    by_model_method = {
        model: {
            method: combinations[(model, method)]
            for method in sorted({call.method for call in calls if call.model == model})
        }
        for model in sorted({call.model for call in calls})
    }
    donation_counts = Counter(
        f"{call.donation_usd:g}" for call in calls if call.donation_usd is not None
    )
    total_expected = sum(
        (Decimal(row["projected_cost_usd"]) for row in model_costs.values()), Decimal(0)
    )
    total_capped = sum(
        (
            Decimal(row["projected_cost_at_output_cap_usd"])
            for row in model_costs.values()
        ),
        Decimal(0),
    )
    total_reserved_once = sum(
        (
            Decimal(row["hard_reservation_one_attempt_each_usd"])
            for row in model_costs.values()
        ),
        Decimal(0),
    )
    total_reserved_all_attempts = sum(
        (
            Decimal(row["hard_reservation_all_attempts_usd"])
            for row in model_costs.values()
        ),
        Decimal(0),
    )
    first_call = calls[0] if calls else None
    smoke_task = None
    if first_call is not None:
        first_rates = ModelRates.from_mapping(model_settings[first_call.model].rates)
        smoke_task = {
            "logical_calls": 1,
            "maximum_http_attempts": 1,
            "task_id": first_call.task_id,
            "model": first_call.model,
            "scenario_id": first_call.scenario_id,
            "frame": first_call.frame,
            "method": first_call.method,
            "order": first_call.order,
            "repetition": first_call.repetition,
            "ladder_level": first_call.ladder_level,
            "donation_usd": first_call.donation_usd,
            "estimated_input_tokens": first_call.estimated_input_tokens,
            "estimated_output_tokens": first_call.estimated_output_tokens,
            "maximum_output_tokens": config.max_output_tokens,
            "projected_cost_usd": _money(
                calculate_cost(
                    Usage(
                        input_tokens=first_call.estimated_input_tokens,
                        output_tokens=first_call.estimated_output_tokens,
                    ),
                    first_rates,
                )
            ),
            "projected_cost_at_output_cap_usd": _money(
                calculate_cost(
                    Usage(
                        input_tokens=first_call.estimated_input_tokens,
                        output_tokens=config.max_output_tokens,
                    ),
                    first_rates,
                )
            ),
            "hard_reservation_usd": _money(
                reservation_cost(
                    first_call.prompt, config.max_output_tokens, first_rates
                )
            ),
            "note": "A successful smoke task is reused by the corresponding resumable run.",
        }
    return {
        "dry_run": True,
        "run_name": config.name,
        "run_id": run_id(config),
        "config_digest": config.digest,
        "planned_logical_calls": len(calls),
        "maximum_http_attempts": len(calls) * (config.max_retries + 1),
        "smoke_task": smoke_task,
        "counts": {
            "by_model": _counter(calls, "model"),
            "by_model_and_method": by_model_method,
            "by_method": _counter(calls, "method"),
            "by_frame": _counter(calls, "frame"),
            "by_scenario": _counter(calls, "scenario_id"),
            "by_order": _counter(calls, "order"),
            "by_repetition": _counter(calls, "repetition"),
            "by_ladder_level": dict(
                sorted(donation_counts.items(), key=lambda item: float(item[0]))
            ),
        },
        "tokens_and_cost": {
            "by_model": model_costs,
            "total_estimated_input_tokens": sum(
                row["estimated_input_tokens"] for row in model_costs.values()
            ),
            "total_estimated_output_tokens": sum(
                row["estimated_output_tokens"] for row in model_costs.values()
            ),
            "total_maximum_output_tokens": sum(
                row["maximum_output_tokens"] for row in model_costs.values()
            ),
            "total_projected_cost_usd": _money(total_expected),
            "total_projected_cost_at_output_cap_usd": _money(total_capped),
            "total_hard_reservation_one_attempt_each_usd": _money(total_reserved_once),
            "total_hard_reservation_all_attempts_usd": _money(
                total_reserved_all_attempts
            ),
            "hard_stop_usd": config.budget_usd,
            "project_budget_usd": config.total_project_budget_usd,
        },
        "assumptions": {
            "input_estimator": (
                f"ceil(UTF-8 bytes / {config.token_estimate['chars_per_token']:g}) "
                f"+ {config.token_estimate['request_overhead']:g} request-overhead tokens"
            ),
            "expected_output_tokens_per_call": math.ceil(
                config.token_estimate["output_per_call"]
            ),
            "maximum_output_tokens_per_call": config.max_output_tokens,
            "cached_input_tokens": 0,
            "service_tier": config.service_tier,
            "pricing_verified_on": config.pricing_verified_on,
            "retry_note": "Logical call totals exclude retries; maximum HTTP attempts include configured retries.",
        },
    }


def run_id(config: ExperimentConfig) -> str:
    return f"{config.name}-{config.digest[:12]}"


def manifest(config: ExperimentConfig, calls: list[PlannedCall]) -> dict[str, Any]:
    return {
        "manifest_version": 1,
        "run_id": run_id(config),
        "run_name": config.name,
        "config_digest": config.digest,
        "source_hashes": config.source_hashes,
        "scenario_version": config.scenario_version,
        "prompt_version": config.prompt_version,
        "seed": config.seed,
        "service_tier": config.service_tier,
        "pricing_verified_on": config.pricing_verified_on,
        "models": [
            {
                "id": model.id,
                "role": model.role,
                "reasoning_effort": model.reasoning_effort,
                "rates_usd_per_million": model.rates,
            }
            for model in config.models
        ],
        "hard_stop_usd": config.budget_usd,
        "project_budget_usd": config.total_project_budget_usd,
        "cost_scope": "shared project ledger across smoke, pilot, and main runs",
        "task_count": len(calls),
        "task_ids_in_seeded_order": [call.task_id for call in calls],
    }


def write_manifest_once(path: Path, document: dict[str, Any]) -> None:
    encoded = json.dumps(document, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"existing manifest does not match this run: {path}")


def load_completed_task_ids(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"invalid run JSONL at line {line_number}: {path}"
                ) from exc
            if record.get("api_status") == "completed":
                completed.add(str(record["task_id"]))
    return completed


def load_project_spend(directory: Path, *additional_paths: Path) -> Decimal:
    # A cost event appears in both the journal and raw log. Its id prevents double count.
    paths = set(directory.glob("*.jsonl")) if directory.exists() else set()
    paths.update(path for path in additional_paths if path.exists())
    events: dict[str, Decimal] = {}
    for path in sorted(paths):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"invalid project cost JSONL at line {line_number}: {path}"
                    ) from exc
                if "incremental_cost_usd" not in record:
                    continue
                event_id = str(
                    record.get("cost_event_id")
                    or f"legacy:{path.resolve()}:{line_number}"
                )
                amount = Decimal(str(record["incremental_cost_usd"]))
                if amount < 0:
                    raise RuntimeError(
                        f"negative project cost at line {line_number}: {path}"
                    )
                previous = events.get(event_id)
                if previous is not None and previous != amount:
                    raise RuntimeError(f"conflicting duplicate cost event: {event_id}")
                events[event_id] = amount
    return sum(events.values(), Decimal(0))


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _response_usage(response: Any, rates: ModelRates) -> Usage | None:
    raw_usage = _get(response, "usage")
    if raw_usage is None:
        return None
    missing = object()
    input_tokens = _get(raw_usage, "input_tokens", missing)
    output_tokens = _get(raw_usage, "output_tokens", missing)
    if (
        input_tokens is missing
        or input_tokens is None
        or output_tokens is missing
        or output_tokens is None
    ):
        return None
    input_details = _get(raw_usage, "input_tokens_details", {})
    output_details = _get(raw_usage, "output_tokens_details", {})
    cache_write = _get(input_details, "cache_write_tokens", missing)
    if rates.cache_write_per_million is not None and (
        cache_write is missing or cache_write is None
    ):
        return None
    try:
        return Usage(
            input_tokens=int(input_tokens),
            cached_input_tokens=int(_get(input_details, "cached_tokens", 0) or 0),
            cache_write_tokens=int(0 if cache_write is missing else cache_write or 0),
            output_tokens=int(output_tokens),
            reasoning_tokens=int(_get(output_details, "reasoning_tokens", 0) or 0),
        )
    except (TypeError, ValueError):
        return None


async def _request(
    client: Any,
    task: PlannedCall,
    model: ModelConfig,
    config: ExperimentConfig,
    max_attempts: int,
) -> tuple[Any, int]:
    kwargs: dict[str, Any] = {
        "model": task.model,
        "input": task.prompt,
        "max_output_tokens": config.max_output_tokens,
        "service_tier": config.service_tier,
    }
    if model.reasoning_effort is not None:
        kwargs["reasoning"] = {"effort": model.reasoning_effort}
    for attempt in range(1, max_attempts + 1):
        try:
            return await client.responses.create(**kwargs), attempt
        except Exception:
            if attempt == max_attempts:
                raise
            await asyncio.sleep(min(2 ** (attempt - 1), 8))
    raise AssertionError("unreachable")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@contextmanager
def _project_run_lock(path: Path):
    # Workers inside one run may overlap. Separate paid CLI runs may not.
    try:
        import fcntl
    except ImportError as exc:
        raise RuntimeError(
            "Paid execution requires POSIX file locking support"
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "another paid runner is already active for this project"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _single_paid_runner(function):
    @wraps(function)
    async def wrapped(config: ExperimentConfig, **kwargs):
        if not kwargs.get("confirm_paid", False):
            return await function(config, **kwargs)
        configured_path = kwargs.get("project_cost_path")
        project_root = config.path.parent.parent
        cost_journal = (
            Path(configured_path).resolve()
            if configured_path
            else (project_root / "data" / "raw" / "project-costs.jsonl").resolve()
        )
        lock_path = cost_journal.with_suffix(cost_journal.suffix + ".lock")
        with _project_run_lock(lock_path):
            return await function(config, **kwargs)

    return wrapped


@_single_paid_runner
async def execute_paid_run(
    config: ExperimentConfig,
    *,
    confirm_paid: bool,
    raw_path: str | Path | None = None,
    max_calls: int | None = None,
    max_attempts: int | None = None,
    client: Any = None,
    project_cost_path: str | Path | None = None,
) -> dict[str, Any]:
    if not confirm_paid:
        raise PermissionError("Paid execution requires explicit confirm_paid=True")
    if max_calls is not None and max_calls < 1:
        raise ValueError("max_calls must be positive when set")
    attempts = max_attempts if max_attempts is not None else config.max_retries + 1
    if attempts < 1:
        raise ValueError("max_attempts must be positive")

    calls = build_plan(config)
    project_root = config.path.parent.parent
    destination = (
        Path(raw_path)
        if raw_path
        else project_root / "data" / "raw" / f"{run_id(config)}.jsonl"
    )
    destination = destination.resolve()
    cost_journal = (
        Path(project_cost_path).resolve()
        if project_cost_path
        else (project_root / "data" / "raw" / "project-costs.jsonl").resolve()
    )
    manifest_path = destination.with_suffix(".manifest.json")
    write_manifest_once(manifest_path, manifest(config, calls))
    completed = load_completed_task_ids(destination)
    pending = [call for call in calls if call.task_id not in completed]
    if max_calls is not None:
        pending = pending[:max_calls]

    if client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Install the API extra with: uv sync --extra api"
            ) from exc
        client = AsyncOpenAI(api_key=api_key, max_retries=0)

    ledger = BudgetLedger(config.budget_usd)
    project_spend = load_project_spend(cost_journal.parent, destination, cost_journal)
    if project_spend:
        previous_id = ledger.reserve(project_spend, "previous-project-spend")
        ledger.reconcile(previous_id, project_spend)
    models = {model.id: model for model in config.models}
    queue: asyncio.Queue[PlannedCall] = asyncio.Queue()
    for call in pending:
        queue.put_nowait(call)
    write_lock = asyncio.Lock()
    stop_for_budget = asyncio.Event()
    completed_now = 0
    completed_task_ids_now: set[str] = set()
    failed_now = 0

    async def worker() -> None:
        nonlocal completed_now, failed_now
        while not stop_for_budget.is_set():
            try:
                task = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            model = models[task.model]
            rates = ModelRates.from_mapping(model.rates)
            one_attempt_reserve = reservation_cost(
                task.prompt, config.max_output_tokens, rates
            )
            # A timed-out attempt may still be billed, so reserve every retry up front.
            reserve_amount = one_attempt_reserve * attempts
            try:
                reservation_id = ledger.reserve(reserve_amount, task.task_id)
            except BudgetExceeded:
                stop_for_budget.set()
                return
            try:
                response, attempt_count = await _request(
                    client, task, model, config, attempts
                )
            except Exception as exc:
                async with write_lock:
                    ledger.reconcile(reservation_id, reserve_amount)
                    timestamp = _timestamp()
                    cost_event_id = uuid4().hex
                    error_record = {
                        **asdict(task),
                        "run_id": run_id(config),
                        "timestamp": timestamp,
                        "api_status": "failed",
                        "attempts": attempts,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:500],
                        "incremental_cost_usd": _money(reserve_amount),
                        "cumulative_cost_usd": _money(ledger.spent),
                        "cost_basis": "conservative reservation for failed attempts with no usage",
                        "cost_event_id": cost_event_id,
                    }
                    # Cost goes first so a crash cannot hide spend from the next run.
                    _append_jsonl(
                        cost_journal,
                        {
                            "record_type": "cost_event",
                            "cost_event_id": cost_event_id,
                            "run_id": run_id(config),
                            "task_id": task.task_id,
                            "timestamp": timestamp,
                            "api_status": "failed",
                            "incremental_cost_usd": _money(reserve_amount),
                            "raw_path": str(destination),
                        },
                    )
                    _append_jsonl(destination, error_record)
                    failed_now += 1
                continue

            raw_response = str(_get(response, "output_text", "") or "")
            parsed = parse_response(raw_response, task.method, task.mirrored)
            usage = _response_usage(response, rates)
            if usage is None:
                actual_cost = one_attempt_reserve * attempt_count
                cost_basis = "conservative reservation for every attempt because response usage was absent"
            else:
                actual_cost = calculate_cost(usage, rates) + one_attempt_reserve * (
                    attempt_count - 1
                )
                cost_basis = (
                    "response usage plus conservative reservations for failed attempts"
                    if attempt_count > 1
                    else "response usage"
                )

            async with write_lock:
                ledger.reconcile(reservation_id, actual_cost)
                timestamp = _timestamp()
                cost_event_id = uuid4().hex
                record = {
                    **asdict(task),
                    "run_id": run_id(config),
                    "timestamp": timestamp,
                    "api_status": "completed",
                    "attempts": attempt_count,
                    "returned_model": _get(response, "model"),
                    "response_id": _get(response, "id"),
                    "model_fingerprint": _get(response, "system_fingerprint"),
                    "raw_response": raw_response,
                    "parse_status": parsed.status,
                    "parsed_raw_value": parsed.raw_value,
                    "parsed_semantic_value": parsed.semantic_value,
                    "parse_error": parsed.error,
                    "input_tokens": usage.input_tokens if usage else None,
                    "cached_input_tokens": usage.cached_input_tokens if usage else None,
                    "cache_write_tokens": usage.cache_write_tokens if usage else None,
                    "reasoning_tokens": usage.reasoning_tokens if usage else None,
                    "output_tokens": usage.output_tokens if usage else None,
                    "incremental_cost_usd": _money(actual_cost),
                    "cumulative_cost_usd": _money(ledger.spent),
                    "cost_basis": cost_basis,
                    "cost_event_id": cost_event_id,
                }
                # Cost goes first so a crash cannot hide spend from the next run.
                _append_jsonl(
                    cost_journal,
                    {
                        "record_type": "cost_event",
                        "cost_event_id": cost_event_id,
                        "run_id": run_id(config),
                        "task_id": task.task_id,
                        "timestamp": timestamp,
                        "api_status": "completed",
                        "incremental_cost_usd": _money(actual_cost),
                        "raw_path": str(destination),
                    },
                )
                _append_jsonl(destination, record)
                completed_now += 1
                completed_task_ids_now.add(task.task_id)

    workers = [
        asyncio.create_task(worker())
        for _ in range(min(config.max_concurrency, len(pending)))
    ]
    if workers:
        await asyncio.gather(*workers)
    remaining_calls = [
        call
        for call in calls
        if call.task_id not in (completed | completed_task_ids_now)
    ]
    remaining_summary = plan_summary(config, remaining_calls)
    return {
        "run_id": run_id(config),
        "raw_path": str(destination),
        "manifest_path": str(manifest_path),
        "project_cost_path": str(cost_journal),
        "already_completed": len(completed),
        "selected_pending": len(pending),
        "completed_now": completed_now,
        "failed_now": failed_now,
        "unprocessed_selected_calls": queue.qsize(),
        "remaining_logical_calls": len(remaining_calls),
        "remaining_plan": {
            "maximum_http_attempts": remaining_summary["maximum_http_attempts"],
            "counts": remaining_summary["counts"],
            "tokens_and_cost": remaining_summary["tokens_and_cost"],
        },
        "stopped_for_budget": stop_for_budget.is_set(),
        "project_cumulative_cost_usd": _money(ledger.spent),
    }
