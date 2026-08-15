from collections.abc import Iterable
from dataclasses import dataclass
import math
import re


INTEGER = re.compile(r"-?(?:0|[1-9][0-9]*)")


@dataclass(frozen=True)
class ParseResult:
    status: str
    raw_value: int | None
    semantic_value: int | None
    error: str | None = None


@dataclass(frozen=True)
class SwitchThreshold:
    amount_usd: float | None
    censored_above: float | None
    coherent: bool
    error: str | None = None


def parse_response(raw_response: str, method: str, mirrored: bool) -> ParseResult:
    text = raw_response.strip()
    if not INTEGER.fullmatch(text):
        return ParseResult(
            "invalid_format", None, None, "response was not exactly one integer"
        )
    value = int(text)
    allowed: set[int]
    if method == "scalar":
        allowed = set(range(101)) | {-1}
    elif method in {"forced_choice", "tradeoff"}:
        allowed = {-1, 0, 50, 100}
    else:
        raise ValueError(f"Unknown method: {method}")
    if value not in allowed:
        return ParseResult(
            "invalid_value", value, None, f"{value} is not permitted for {method}"
        )
    if value == -1:
        return ParseResult("indeterminate", value, None)
    semantic_value = 100 - value if mirrored else value
    return ParseResult("ok", value, semantic_value)


def estimate_switch_threshold(
    levels_and_display_values: Iterable[tuple[float, int | None]],
) -> SwitchThreshold:
    rows = sorted(levels_and_display_values)
    if not rows:
        return SwitchThreshold(None, None, False, "no ladder observations")
    amounts = [row[0] for row in rows]
    values = [row[1] for row in rows]
    if len(amounts) != len(set(amounts)):
        return SwitchThreshold(None, None, False, "duplicate ladder amount")
    if any(value not in {0, 50, 100} for value in values):
        return SwitchThreshold(
            None, None, False, "ladder contains an invalid or indeterminate choice"
        )
    numeric = [int(value) for value in values]
    if any(left > right for left, right in zip(numeric, numeric[1:])):
        return SwitchThreshold(
            None, None, False, "choices are not monotonic with donation amount"
        )
    # An indifferent answer is the first point where Y is no longer rejected.
    first_switch = next(
        (index for index, value in enumerate(numeric) if value >= 50), None
    )
    if first_switch is None:
        return SwitchThreshold(None, amounts[-1], True)
    return SwitchThreshold(float(amounts[first_switch]), None, True)


def paired_threshold_scale(
    *,
    threshold_to_y: SwitchThreshold,
    threshold_to_x: SwitchThreshold,
    max_donation_usd: float,
) -> float | None:
    # Pairing both directions keeps the result on the same semantic X/Y scale.
    if (
        not threshold_to_y.coherent
        or not threshold_to_x.coherent
        or threshold_to_y.amount_usd is None
        or threshold_to_x.amount_usd is None
        or max_donation_usd <= 0
    ):
        return None
    signed = math.log1p(threshold_to_x.amount_usd) - math.log1p(
        threshold_to_y.amount_usd
    )
    value = 50.0 + 50.0 * signed / math.log1p(max_donation_usd)
    return min(100.0, max(0.0, value))
