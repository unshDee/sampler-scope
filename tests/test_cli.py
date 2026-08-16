import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from samplerscope.cli import _synthetic, _validate, messages_for_state
from samplerscope.environments import service_recovery


class CliTests(unittest.TestCase):
    def test_message_labels_follow_the_semantic_mapping(self):
        environment = service_recovery()
        messages = messages_for_state(
            environment,
            "recovery_0_1_0",
            {"prepare_backup": "C", "repair": "A", "restart": "B"},
        )

        prompt = messages[1]["content"]
        self.assertIn("Score equals success", prompt)
        self.assertLess(prompt.index("C: Prepare"), prompt.index("A: Repair"))
        self.assertLess(prompt.index("A: Repair"), prompt.index("B: Attempt"))

    def test_service_restart_rules_match_the_transition_model(self):
        environment = service_recovery()
        state = environment.state_index("recovery_0_3_1")
        action = environment.actions.index("restart")
        success, failure = environment.terminal_states

        self.assertIn("0.20 times work plus 0.30 if backed up", environment.rules)
        self.assertIn("minus 0.35 if backed up", environment.rules)
        self.assertAlmostEqual(environment.transitions[state][action][success], 0.7)
        self.assertAlmostEqual(environment.transitions[state][action][failure], 0.3)
        self.assertAlmostEqual(
            environment.outcomes["stakeholder_cost"][state][action][failure],
            0.85,
        )

    def test_validate_and_synthetic_controls_are_runnable(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(_validate(), 0)
        validated = json.loads(stdout.getvalue())
        self.assertEqual(sum(row["decision_states"] for row in validated), 42)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "synthetic.json"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(_synthetic(str(output)), 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["kind"], "synthetic_control")
        self.assertEqual(len(payload["results"]), 44)


if __name__ == "__main__":
    unittest.main()
