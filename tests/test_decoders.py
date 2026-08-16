import math
import unittest

from samplerscope.decoders import (
    apply_stack,
    decode,
    distribution_metrics,
    greedy,
    min_p,
    softmax,
    temperature,
    top_k,
    top_p,
)


class DecoderTests(unittest.TestCase):
    def setUp(self):
        self.raw = (0.6, 0.25, 0.15)
        self.logits = tuple(math.log(value) for value in self.raw)

    def test_softmax_is_stable(self):
        result = softmax((10_000, 9_999))
        self.assertAlmostEqual(sum(result), 1.0)
        self.assertGreater(result[0], result[1])

    def test_decoder_transforms(self):
        self.assertEqual(greedy(self.raw), (1.0, 0.0, 0.0))
        self.assertEqual(min_p(self.raw, 0.5), (1.0, 0.0, 0.0))
        self.assertEqual(top_p(self.raw, 0.6), (1.0, 0.0, 0.0))

        kept = top_k(self.raw, 2)
        self.assertAlmostEqual(kept[0], 0.6 / 0.85)
        self.assertAlmostEqual(kept[1], 0.25 / 0.85)
        self.assertEqual(kept[2], 0.0)

        colder = temperature(self.raw, 0.5)
        self.assertGreater(colder[0], self.raw[0])
        self.assertAlmostEqual(sum(colder), 1.0)

    def test_cutoff_ties_have_explicit_semantics(self):
        tied = softmax((0.0, 0.0, -1.0))
        top = top_k(tied, 1)
        self.assertAlmostEqual(top[0], 0.5)
        self.assertAlmostEqual(top[1], 0.5)
        self.assertEqual(top[2], 0.0)

        self.assertEqual(greedy(tied, (34, 32, 33)), (0.0, 1.0, 0.0))
        nucleus = top_p((0.4, 0.3, 0.3), 0.7, (34, 33, 32))
        self.assertGreater(nucleus[2], 0.0)
        self.assertEqual(nucleus[1], 0.0)

    def test_operator_order_is_preserved(self):
        temperature_then_top_p = apply_stack(
            self.logits, (("temperature", 0.5), ("top_p", 0.8))
        )
        top_p_then_temperature = apply_stack(
            self.logits, (("top_p", 0.8), ("temperature", 0.5))
        )
        self.assertEqual(temperature_then_top_p, (1.0, 0.0, 0.0))
        self.assertGreater(top_p_then_temperature[1], 0.0)
        self.assertNotEqual(temperature_then_top_p, top_p_then_temperature)

    def test_temperature_and_top_k_commute_without_rank_ties(self):
        temperature_then_top_k = apply_stack(
            self.logits, (("temperature", 0.5), ("top_k", 2))
        )
        top_k_then_temperature = apply_stack(
            self.logits, (("top_k", 2), ("temperature", 0.5))
        )
        for first, second in zip(temperature_then_top_k, top_k_then_temperature):
            self.assertAlmostEqual(first, second)

    def test_single_decoder_convenience(self):
        self.assertEqual(decode(self.logits, "greedy"), (1.0, 0.0, 0.0))
        raw = decode(self.logits)
        for observed, expected in zip(raw, self.raw):
            self.assertAlmostEqual(observed, expected)

    def test_distribution_metrics_use_decoded_to_raw_kl(self):
        decoded = (0.625, 0.375, 0.0)
        metrics = distribution_metrics((0.5, 0.3, 0.2), decoded)
        self.assertAlmostEqual(metrics["total_variation"], 0.2)
        self.assertAlmostEqual(metrics["censored_raw_mass"], 0.2)
        self.assertGreater(metrics["jensen_shannon"], 0.0)
        self.assertTrue(math.isfinite(metrics["kl_decoded_raw"]))

    def test_invalid_inputs_are_rejected(self):
        bad_calls = (
            lambda: softmax(()),
            lambda: softmax((0.0, math.inf)),
            lambda: temperature(self.raw, 0),
            lambda: top_k(self.raw, 0),
            lambda: top_p(self.raw, 0),
            lambda: min_p(self.raw, 1.1),
            lambda: decode(self.logits, "missing"),
            lambda: distribution_metrics((0.5, 0.5), (1.0, 0.0, 0.0)),
        )
        for call in bad_calls:
            with self.subTest(call=call), self.assertRaises(ValueError):
                call()


if __name__ == "__main__":
    unittest.main()
