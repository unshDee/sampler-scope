import unittest
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

from preference_elicitation.costs import (
    BudgetExceeded,
    BudgetLedger,
    ModelRates,
    Usage,
    calculate_cost,
    reservation_cost,
)


LUNA = ModelRates(
    input_per_million="0.20",
    cached_input_per_million="0.02",
    cache_write_per_million="0.25",
    output_per_million="1.20",
)
MINI = ModelRates(
    input_per_million="0.15",
    cached_input_per_million="0.075",
    output_per_million="0.60",
)


class CostTests(unittest.TestCase):
    def test_exact_luna_cost(self):
        usage = Usage(
            input_tokens=1_000_000,
            cached_input_tokens=200_000,
            cache_write_tokens=300_000,
            output_tokens=100_000,
            reasoning_tokens=40_000,
        )
        self.assertEqual(calculate_cost(usage, LUNA), Decimal("0.299"))

    def test_exact_4o_mini_cost(self):
        usage = Usage(
            input_tokens=1_000_000,
            cached_input_tokens=200_000,
            output_tokens=100_000,
        )
        self.assertEqual(calculate_cost(usage, MINI), Decimal("0.195"))

    def test_reasoning_tokens_are_not_double_counted(self):
        plain = Usage(input_tokens=100, output_tokens=80)
        reasoned = Usage(input_tokens=100, output_tokens=80, reasoning_tokens=70)
        self.assertEqual(calculate_cost(plain, LUNA), calculate_cost(reasoned, LUNA))

    def test_reservation_uses_utf8_ceiling_and_cache_write_rate(self):
        cost = reservation_cost("\u00e9", 20, LUNA, protocol_overhead_tokens=8)
        self.assertEqual(cost, Decimal("0.0000265"))


class BudgetLedgerTests(unittest.TestCase):
    def test_rejects_over_cap_and_reconciles(self):
        ledger = BudgetLedger("1.00")
        first = ledger.reserve("0.60")
        with self.assertRaises(BudgetExceeded):
            ledger.reserve("0.41")

        self.assertEqual(first, "reservation-1")
        self.assertEqual(ledger.reserved, Decimal("0.60"))

        ledger = BudgetLedger("1.00")
        second = ledger.reserve("0.50", "call-1")
        self.assertEqual(second, "call-1")
        ledger.reconcile(second, "0.20")
        self.assertEqual(ledger.spent, Decimal("0.20"))
        self.assertEqual(ledger.reserved, Decimal("0"))
        self.assertEqual(ledger.available, Decimal("0.80"))

    def test_failed_reconcile_leaves_reservation_intact(self):
        ledger = BudgetLedger("1.00")
        reservation_id = ledger.reserve("0.40", "call-1")

        with self.assertRaises(BudgetExceeded):
            ledger.reconcile(reservation_id, "1.01")

        self.assertEqual(ledger.spent, Decimal("0"))
        self.assertEqual(ledger.reserved, Decimal("0.40"))

    def test_concurrent_reservations_cannot_overcommit(self):
        ledger = BudgetLedger("1.00")

        def try_reserve():
            try:
                return ledger.reserve("0.60")
            except BudgetExceeded:
                return None

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: try_reserve(), range(2)))

        self.assertEqual(sum(result is not None for result in results), 1)
        self.assertEqual(ledger.reserved, Decimal("0.60"))


if __name__ == "__main__":
    unittest.main()
