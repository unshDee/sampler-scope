import unittest
from collections import Counter
from itertools import permutations

from samplerscope.environments import LABELS, benchmark_environments, label_permutations
from samplerscope.experiment import (
    analyze_logits,
    optimal_action_sets,
    optimal_actions,
    optimal_policy,
    run_experiment,
    summarize_results,
    synthetic_control_logits,
)
from samplerscope.mdp import evaluate_policy
from samplerscope.trace import LogitTrace, TraceRow, hash_prompt, hash_token_ids


class EnvironmentTests(unittest.TestCase):
    def test_benchmarks_have_complete_finite_dynamics(self):
        environments = benchmark_environments()
        self.assertEqual(sum(len(env.nonterminal_states) for env in environments), 42)

        for environment in environments:
            expected_states = 18 if environment.name == "service_recovery" else 24
            self.assertEqual(len(environment.nonterminal_states), expected_states)
            self.assertEqual(len(environment.states), expected_states + 2)
            self.assertEqual(len(environment.actions), 3)
            self.assertAlmostEqual(sum(environment.initial), 1.0)
            self.assertTrue(
                all(
                    "Round" in state.observation
                    for state in environment.nonterminal_states
                )
            )
            for state_index, state in enumerate(environment.states):
                for action_index in range(3):
                    total = sum(environment.transitions[state_index][action_index])
                    self.assertAlmostEqual(total, 0.0 if state.terminal else 1.0)

    def test_all_actions_are_optimal_somewhere(self):
        expected_returns = {
            "service_recovery": 0.98,
            "queue_control": 0.12119739583333335,
        }
        for environment in benchmark_environments():
            actions = optimal_actions(environment)
            self.assertEqual(set(actions.values()), set(environment.actions))
            result = evaluate_policy(
                environment.initial,
                environment.transitions,
                environment.outcomes["return"],
                optimal_policy(environment),
                environment.horizon,
                environment.terminal_states,
            )
            self.assertAlmostEqual(
                result.expected_return, expected_returns[environment.name]
            )

    def test_queue_control_keeps_all_co_optimal_actions(self):
        environment = benchmark_environments()[1]
        choices = optimal_action_sets(environment)
        tied = [actions for actions in choices.values() if len(actions) > 1]
        self.assertEqual(len(tied), 8)
        self.assertTrue(
            all(actions == ("add_capacity", "standard_service") for actions in tied)
        )

    def test_six_label_mappings_preserve_action_names(self):
        for environment in benchmark_environments():
            mappings = label_permutations(environment.actions)
            self.assertEqual(len(mappings), 6)
            self.assertEqual(
                {tuple(mapping.values()) for mapping in mappings},
                set(permutations(LABELS)),
            )


class ExperimentTests(unittest.TestCase):
    def test_synthetic_margin_controls_have_known_targets(self):
        decoders = {"raw": (), "greedy": ("greedy",)}
        for environment in benchmark_environments():
            high = analyze_logits(
                environment,
                synthetic_control_logits(environment, "high"),
                decoders,
            )
            low = analyze_logits(
                environment,
                synthetic_control_logits(environment, "low"),
                decoders,
            )
            high_by_decoder = {row["decoder"]: row for row in high}
            low_by_decoder = {row["decoder"]: row for row in low}

            self.assertAlmostEqual(high_by_decoder["greedy"]["optimality_gap"], 0.0)
            self.assertAlmostEqual(low_by_decoder["greedy"]["optimality_gap"], 0.0)
            self.assertLess(
                high_by_decoder["raw"]["optimality_gap"],
                low_by_decoder["raw"]["optimality_gap"],
            )
            self.assertGreater(
                high_by_decoder["raw"]["mean_optimal_action_probability"], 0.99
            )
            self.assertLess(
                low_by_decoder["raw"]["mean_optimal_action_probability"], 0.55
            )

    def test_trace_produces_one_exact_row_per_mapping_and_decoder(self):
        environment = benchmark_environments()[0]
        logits = synthetic_control_logits(environment, "low")
        trace_rows = []
        for mapping in label_permutations(environment.actions):
            for state in environment.nonterminal_states:
                trace_rows.append(
                    TraceRow(
                        environment=environment.name,
                        state_id=state.id,
                        label_by_action=mapping,
                        logits_by_action=logits[state.id],
                        token_id_by_action={
                            action: ord(mapping[action])
                            for action in environment.actions
                        },
                        valid_action_mass=0.5,
                        best_valid_token_rank=1,
                        rendered_prompt="synthetic prompt",
                        input_token_ids=tuple(range(10)),
                        prompt_sha256=hash_prompt("synthetic prompt"),
                        input_sha256=hash_token_ids(tuple(range(10))),
                        input_tokens=10,
                    )
                )
        trace = LogitTrace(
            model_id="synthetic",
            actions=environment.actions,
            labels=LABELS,
            rows=trace_rows,
        )

        matrix = run_experiment(
            environment,
            trace,
            {"raw": (), "greedy": ("greedy",)},
        )

        self.assertEqual(len(matrix), 12)
        self.assertEqual(
            Counter(row["decoder"] for row in matrix), {"raw": 6, "greedy": 6}
        )
        self.assertEqual(len({row["label_mapping"] for row in matrix}), 6)
        self.assertTrue(all(row["input_tokens"] == 180 for row in matrix))
        self.assertTrue(all("optimality_gap" in row for row in matrix))
        for row in matrix:
            self.assertAlmostEqual(row["mean_valid_action_mass"], 0.5)
        self.assertTrue(
            all(sum(row["raw_greedy_label_counts"].values()) == 18 for row in matrix)
        )

        summary = summarize_results(matrix)
        self.assertEqual(len(summary["by_decoder"]), 2)
        self.assertTrue(
            all(row["label_mapping_return_range"] == 0 for row in summary["by_decoder"])
        )

    def test_summary_keeps_pipeline_order_paired_by_label_mapping(self):
        fields = {
            "decoder_value_gap": 0,
            "optimality_gap": 0,
            "task_success": 0,
            "stakeholder_cost": 0,
            "mean_state_occupancy_tv": 0,
            "total_variation": 0,
            "censored_raw_mass": 0,
            "optimal_action_censor_rate": 0,
            "mean_valid_action_mass": 1,
            "occupancy_weighted_logit_tie_rate": 0,
        }
        rows = []
        for mapping, left, right in (("A-B-C", 0.7, 0.5), ("B-A-C", 0.2, 0.3)):
            rows.extend(
                (
                    {
                        "decoder": "temperature_0.5_then_top_p_0.8",
                        "label_mapping": mapping,
                        "expected_return": left,
                        **fields,
                    },
                    {
                        "decoder": "top_p_0.8_then_temperature_0.5",
                        "label_mapping": mapping,
                        "expected_return": right,
                        **fields,
                    },
                )
            )

        effect = summarize_results(rows)["operator_order_effect"]
        self.assertEqual(effect["paired_label_mappings"], 2)
        self.assertAlmostEqual(effect["mean_return_delta"], 0.05)
        self.assertAlmostEqual(effect["maximum_absolute_return_delta"], 0.2)

    def test_incomplete_trace_is_rejected(self):
        environment = benchmark_environments()[0]
        trace = LogitTrace(
            model_id="synthetic",
            actions=environment.actions,
            labels=LABELS,
        )
        with self.assertRaisesRegex(ValueError, "six label permutations"):
            run_experiment(environment, trace, {"raw": ()})


if __name__ == "__main__":
    unittest.main()
