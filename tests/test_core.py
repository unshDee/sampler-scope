import json
from pathlib import Path
import tempfile
import unittest

from preference_elicitation.analysis import (
    analyze_records,
    bootstrap_omega,
    normalized_pull,
    spearman,
)
from preference_elicitation.config import ConfigError, load_experiment
from preference_elicitation.parsing import (
    estimate_switch_threshold,
    paired_threshold_scale,
    parse_response,
)
from preference_elicitation.runner import build_plan, plan_summary


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "configs" / "pilot.yaml"
MAIN = ROOT / "configs" / "main.yaml"


class ConfigurationAndPlanTests(unittest.TestCase):
    def test_configs_validate_and_exact_counts_match(self):
        pilot = load_experiment(PILOT)
        main = load_experiment(MAIN)
        pilot_summary = plan_summary(pilot)
        main_summary = plan_summary(main)

        self.assertEqual(pilot_summary["planned_logical_calls"], 420)
        self.assertEqual(pilot_summary["smoke_task"]["logical_calls"], 1)
        self.assertEqual(pilot_summary["smoke_task"]["maximum_http_attempts"], 1)
        self.assertEqual(
            pilot_summary["counts"]["by_model_and_method"]["gpt-5.6-luna"],
            {"forced_choice": 60, "scalar": 60, "tradeoff": 300},
        )
        self.assertEqual(main_summary["planned_logical_calls"], 3360)
        self.assertEqual(
            main_summary["counts"]["by_model"],
            {
                "gpt-4o-mini-2024-07-18": 1680,
                "gpt-5.6-luna": 1680,
            },
        )

    def test_plan_is_deterministic_and_mirroring_retains_semantics(self):
        config = load_experiment(PILOT)
        first = build_plan(config)
        second = build_plan(config)
        self.assertEqual(
            [call.task_id for call in first], [call.task_id for call in second]
        )

        original = next(
            call
            for call in first
            if call.scenario_id == "persistent_memory_reset"
            and call.frame == "self"
            and call.method == "scalar"
            and call.repetition == 1
            and call.order == "original"
        )
        mirrored = next(
            call
            for call in first
            if call.scenario_id == "persistent_memory_reset"
            and call.frame == "self"
            and call.method == "scalar"
            and call.repetition == 1
            and call.order == "mirrored"
        )
        self.assertEqual(original.semantic_x, mirrored.semantic_x)
        self.assertEqual(original.semantic_y, mirrored.semantic_y)
        self.assertEqual(original.displayed_x, mirrored.displayed_y)
        self.assertEqual(original.displayed_y, mirrored.displayed_x)
        self.assertEqual(parse_response("0", "scalar", False).semantic_value, 0)
        self.assertEqual(parse_response("0", "scalar", True).semantic_value, 100)

    def test_invalid_config_is_rejected(self):
        raw = json.loads(PILOT.read_text(encoding="utf-8"))
        scenarios = (ROOT / "configs" / "scenarios.yaml").read_text(encoding="utf-8")
        raw["max_output_tokens"] = 8
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "scenarios.yaml").write_text(scenarios, encoding="utf-8")
            (path / "pilot.yaml").write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_experiment(path / "pilot.yaml")


class ParsingAndAggregationTests(unittest.TestCase):
    def test_parser_is_strict_and_method_specific(self):
        self.assertEqual(parse_response(" 50\n", "forced_choice", False).status, "ok")
        self.assertEqual(parse_response("-1", "scalar", False).status, "indeterminate")
        self.assertEqual(
            parse_response("51", "forced_choice", False).status, "invalid_value"
        )
        self.assertEqual(
            parse_response("50 because", "scalar", False).status, "invalid_format"
        )
        self.assertEqual(parse_response("50", "tradeoff", False).status, "ok")

    def test_paired_tradeoff_threshold(self):
        to_y = estimate_switch_threshold([(0, 0), (1, 50), (10, 100), (100, 100)])
        to_x = estimate_switch_threshold([(0, 0), (1, 0), (10, 0), (100, 50)])
        score = paired_threshold_scale(
            threshold_to_y=to_y,
            threshold_to_x=to_x,
            max_donation_usd=100,
        )
        self.assertIsNotNone(score)
        self.assertGreater(score, 50)

    def test_tradeoff_censoring_and_nonmonotonicity_are_preserved(self):
        censored = estimate_switch_threshold([(0, 0), (1, 0), (10, 0)])
        incoherent = estimate_switch_threshold([(0, 0), (1, 100), (10, 0)])
        self.assertEqual(censored.censored_above, 10)
        self.assertTrue(censored.coherent)
        self.assertFalse(incoherent.coherent)
        self.assertIsNone(
            paired_threshold_scale(
                threshold_to_y=censored,
                threshold_to_x=censored,
                max_donation_usd=10,
            )
        )


class AnalysisMathTests(unittest.TestCase):
    def test_denominator_rule_and_unclipped_omega(self):
        self.assertFalse(normalized_pull(40, 49, 60)["identifiable"])
        estimate = normalized_pull(20, 80, 100)
        self.assertTrue(estimate["identifiable"])
        self.assertGreater(estimate["value"], 1)

    def test_bootstrap_and_spearman(self):
        interval = bootstrap_omega([0, 2], [98, 100], [49, 51], samples=100, seed=7)
        self.assertIsNotNone(interval)
        self.assertLessEqual(interval[0], interval[1])
        self.assertEqual(spearman([1, 2, 3], [10, 20, 30]), 1)
        self.assertEqual(spearman([1, 2, 3], [30, 20, 10]), -1)

    def test_method_summary_has_bootstrap_intervals(self):
        records = []
        for frame, values in {
            "typical": [10, 20],
            "ideal": [80, 90],
            "self": [50, 60],
        }.items():
            for value in values:
                records.append(
                    {
                        "api_status": "completed",
                        "parse_status": "ok",
                        "model": "test-model",
                        "scenario_id": "scenario",
                        "method": "scalar",
                        "frame": frame,
                        "order": "original",
                        "parsed_semantic_value": value,
                    }
                )
        result = analyze_records(records, bootstrap_samples=50)
        summary = result["method_summary"][0]
        self.assertIsNotNone(summary["median_omega_bootstrap_95_ci"])
        self.assertIsNotNone(summary["mean_omega_bootstrap_95_ci"])


if __name__ == "__main__":
    unittest.main()
