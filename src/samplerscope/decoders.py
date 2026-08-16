import math
from collections.abc import Sequence

_TOLERANCE = 1e-9


def _finite_values(values: Sequence[float], name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numbers") from error
    if not result:
        raise ValueError(f"{name} cannot be empty")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite numbers")
    return result


def _distribution(
    values: Sequence[float], name: str = "probabilities"
) -> tuple[float, ...]:
    result = _finite_values(values, name)
    if any(value < 0 for value in result):
        raise ValueError(f"{name} cannot contain negative values")
    total = sum(result)
    if not math.isclose(total, 1.0, rel_tol=_TOLERANCE, abs_tol=_TOLERANCE):
        raise ValueError(f"{name} must sum to 1")
    return tuple(value / total for value in result)


def _renormalize(probabilities: Sequence[float]) -> tuple[float, ...]:
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("decoder removed every action")
    return tuple(value / total for value in probabilities)


def softmax(logits: Sequence[float]) -> tuple[float, ...]:
    values = _finite_values(logits, "logits")
    largest = max(values)
    weights = tuple(math.exp(value - largest) for value in values)
    return _renormalize(weights)


def _priorities(values: Sequence[float], tie_breakers: Sequence[int] | None):
    if tie_breakers is None:
        return tuple(range(len(values)))
    priorities = tuple(int(value) for value in tie_breakers)
    if len(priorities) != len(values) or len(set(priorities)) != len(priorities):
        raise ValueError("tie breakers must be unique and match the action count")
    return priorities


def greedy(
    probabilities: Sequence[float], tie_breakers: Sequence[int] | None = None
) -> tuple[float, ...]:
    values = _distribution(probabilities)
    priorities = _priorities(values, tie_breakers)
    winner = min(
        range(len(values)), key=lambda index: (-values[index], priorities[index])
    )
    return tuple(1.0 if index == winner else 0.0 for index in range(len(values)))


def temperature(probabilities: Sequence[float], value: float) -> tuple[float, ...]:
    values = _distribution(probabilities)
    try:
        setting = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("temperature must be a positive finite number") from error
    if not math.isfinite(setting) or setting <= 0:
        raise ValueError("temperature must be a positive finite number")
    scaled = tuple(
        math.log(probability) / setting if probability > 0 else -math.inf
        for probability in values
    )
    largest = max(scaled)
    return _renormalize(
        tuple(math.exp(log_probability - largest) for log_probability in scaled)
    )


def top_k(probabilities: Sequence[float], value: int) -> tuple[float, ...]:
    values = _distribution(probabilities)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("top-k must be an integer")
    if not 1 <= value <= len(values):
        raise ValueError("top-k must be between 1 and the action count")
    threshold = sorted(values, reverse=True)[value - 1]
    return _renormalize(
        tuple(
            probability if probability >= threshold else 0.0 for probability in values
        )
    )


def top_p(
    probabilities: Sequence[float],
    value: float,
    tie_breakers: Sequence[int] | None = None,
) -> tuple[float, ...]:
    values = _distribution(probabilities)
    cutoff = _unit_interval(value, "top-p", allow_zero=False)
    priorities = _priorities(values, tie_breakers)
    ranked = sorted(
        range(len(values)), key=lambda index: (-values[index], priorities[index])
    )
    keep = set()
    cumulative = 0.0
    for index in ranked:
        keep.add(index)
        cumulative += values[index]
        if cumulative >= cutoff:
            break
    return _renormalize(
        tuple(
            probability if index in keep else 0.0
            for index, probability in enumerate(values)
        )
    )


def min_p(probabilities: Sequence[float], value: float) -> tuple[float, ...]:
    values = _distribution(probabilities)
    ratio = _unit_interval(value, "min-p", allow_zero=True)
    threshold = ratio * max(values)
    return _renormalize(
        tuple(
            probability if probability >= threshold else 0.0 for probability in values
        )
    )


def _unit_interval(value: float, name: str, *, allow_zero: bool) -> float:
    try:
        setting = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be between 0 and 1") from error
    lower_bound = setting >= 0 if allow_zero else setting > 0
    if not math.isfinite(setting) or not lower_bound or setting > 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return setting


def _apply(
    probabilities: Sequence[float],
    name: str,
    value: float | None,
    tie_breakers: Sequence[int] | None = None,
):
    if name == "raw":
        if value is not None:
            raise ValueError("raw does not take a value")
        return _distribution(probabilities)
    if name == "greedy":
        if value is not None:
            raise ValueError("greedy does not take a value")
        return greedy(probabilities, tie_breakers)
    if value is None:
        raise ValueError(f"{name} requires a value")
    if name == "temperature":
        return temperature(probabilities, value)
    if name == "top_k":
        return top_k(probabilities, value)
    if name == "top_p":
        return top_p(probabilities, value, tie_breakers)
    if name == "min_p":
        return min_p(probabilities, value)
    raise ValueError(f"unknown decoder: {name}")


def decode(
    logits: Sequence[float],
    method: str = "raw",
    value: float | None = None,
    tie_breakers: Sequence[int] | None = None,
) -> tuple[float, ...]:
    if not isinstance(method, str):
        raise TypeError("decoder name must be a string")
    return _apply(softmax(logits), method, value, tie_breakers)


def apply_stack(
    logits: Sequence[float],
    operators: Sequence[str | tuple[str, float | int | None]],
    tie_breakers: Sequence[int] | None = None,
) -> tuple[float, ...]:
    probabilities = softmax(logits)
    for operator in operators:
        if isinstance(operator, str):
            name, value = operator, None
        elif (
            isinstance(operator, tuple)
            and len(operator) == 2
            and isinstance(operator[0], str)
        ):
            name, value = operator
        else:
            raise ValueError(
                "each decoder step must be a name or a (name, value) tuple"
            )
        probabilities = _apply(probabilities, name, value, tie_breakers)
    return probabilities


def total_variation(first: Sequence[float], second: Sequence[float]) -> float:
    left, right = _matching_distributions(first, second)
    return 0.5 * sum(abs(a - b) for a, b in zip(left, right))


def kl_divergence(first: Sequence[float], second: Sequence[float]) -> float:
    left, right = _matching_distributions(first, second)
    total = 0.0
    for a, b in zip(left, right):
        if a == 0:
            continue
        if b == 0:
            return math.inf
        total += a * math.log(a / b)
    return total


def jensen_shannon(first: Sequence[float], second: Sequence[float]) -> float:
    left, right = _matching_distributions(first, second)
    midpoint = tuple((a + b) / 2 for a, b in zip(left, right))
    return 0.5 * kl_divergence(left, midpoint) + 0.5 * kl_divergence(right, midpoint)


def censored_raw_mass(raw: Sequence[float], decoded: Sequence[float]) -> float:
    baseline, result = _matching_distributions(raw, decoded)
    return sum(probability for probability, kept in zip(baseline, result) if kept == 0)


def _matching_distributions(
    first: Sequence[float], second: Sequence[float]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    left = _distribution(first, "first distribution")
    right = _distribution(second, "second distribution")
    if len(left) != len(right):
        raise ValueError("distributions must have the same length")
    return left, right


def distribution_metrics(
    raw: Sequence[float], decoded: Sequence[float]
) -> dict[str, float]:
    baseline, result = _matching_distributions(raw, decoded)
    return {
        "total_variation": total_variation(baseline, result),
        "jensen_shannon": jensen_shannon(baseline, result),
        "kl_decoded_raw": kl_divergence(result, baseline),
        "censored_raw_mass": censored_raw_mass(baseline, result),
    }
