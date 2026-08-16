import unittest

from samplerscope.mdp import evaluate_policy


class MdpTests(unittest.TestCase):
    def test_terminal_mass_and_return_are_exact(self):
        transitions = (
            ((0, 1, 0), (0, 0, 1)),
            ((0, 0, 0), (0, 0, 0)),
            ((0, 0, 0), (0, 0, 0)),
        )
        rewards = (
            ((0, 1, 0), (0, 0, -1)),
            ((0, 0, 0), (0, 0, 0)),
            ((0, 0, 0), (0, 0, 0)),
        )
        result = evaluate_policy(
            initial=(1, 0, 0),
            transitions=transitions,
            rewards=rewards,
            policy=((0.75, 0.25), (0, 0), (0, 0)),
            horizon=2,
            terminal_states={1, 2},
        )

        self.assertAlmostEqual(result.expected_return, 0.5)
        self.assertEqual(result.state_occupancy[0], (1.0, 0.0, 0.0))
        self.assertEqual(result.state_occupancy[1], (0.0, 0.75, 0.25))
        self.assertEqual(result.state_occupancy[2], (0.0, 0.75, 0.25))
        self.assertEqual(result.terminal_hitting_probabilities, {1: 0.75, 2: 0.25})

    def test_stationary_policy_compounds_over_steps(self):
        transitions = (
            ((0, 1, 0, 0), (0, 0, 0, 1)),
            ((0, 0, 1, 0), (0, 0, 0, 1)),
            ((0, 0, 0, 0), (0, 0, 0, 0)),
            ((0, 0, 0, 0), (0, 0, 0, 0)),
        )
        rewards = (
            ((0, 0, 0, 0), (0, 0, 0, -1)),
            ((0, 0, 2, 0), (0, 0, 0, -1)),
            ((0, 0, 0, 0), (0, 0, 0, 0)),
            ((0, 0, 0, 0), (0, 0, 0, 0)),
        )
        result = evaluate_policy(
            initial=(1, 0, 0, 0),
            transitions=transitions,
            rewards=rewards,
            policy=((0.5, 0.5), (0.25, 0.75), (0, 0), (0, 0)),
            horizon=2,
            terminal_states={2, 3},
        )

        self.assertAlmostEqual(result.expected_return, -0.625)
        self.assertEqual(result.state_occupancy[-1], (0.0, 0.0, 0.125, 0.875))
        self.assertEqual(result.terminal_hitting_probabilities, {2: 0.125, 3: 0.875})

    def test_zero_horizon_only_counts_initial_terminal_mass(self):
        result = evaluate_policy(
            initial=(0.4, 0.6),
            transitions=(((1, 0),), ((0, 0),)),
            rewards=(((0, 0),), ((0, 0),)),
            policy=((1,), (0,)),
            horizon=0,
            terminal_states={1},
        )
        self.assertEqual(result.expected_return, 0.0)
        self.assertEqual(result.state_occupancy, ((0.4, 0.6),))
        self.assertEqual(result.terminal_hitting_probabilities, {1: 0.6})

    def test_invalid_shapes_and_probabilities_are_rejected(self):
        valid = {
            "initial": (1, 0),
            "transitions": (((1, 0),), ((0, 0),)),
            "rewards": (((0, 0),), ((0, 0),)),
            "policy": ((1,), (0,)),
            "horizon": 1,
            "terminal_states": {1},
        }
        cases = (
            {**valid, "initial": (0.8, 0.1)},
            {**valid, "policy": ((0.5,), (0,))},
            {**valid, "transitions": (((0.5, 0),), ((0, 0),))},
            {**valid, "rewards": (((0,),), ((0, 0),))},
            {**valid, "terminal_states": {2}},
            {**valid, "horizon": -1},
        )
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                evaluate_policy(**case)


if __name__ == "__main__":
    unittest.main()
