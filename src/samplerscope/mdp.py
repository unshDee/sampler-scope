import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass

_TOLERANCE = 1e-9


@dataclass(frozen=True)
class FiniteHorizonResult:
    expected_return: float
    state_occupancy: tuple[tuple[float, ...], ...]
    terminal_hitting_probabilities: dict[int, float]


def _numbers(values: Sequence[float], name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numbers") from error
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite numbers")
    return result


def _probabilities(
    values: Sequence[float], name: str, *, allow_zero: bool = False
) -> tuple[float, ...]:
    result = _numbers(values, name)
    if any(value < 0 for value in result):
        raise ValueError(f"{name} cannot contain negative values")
    total = sum(result)
    if allow_zero and math.isclose(total, 0.0, abs_tol=_TOLERANCE):
        return result
    if not math.isclose(total, 1.0, rel_tol=_TOLERANCE, abs_tol=_TOLERANCE):
        raise ValueError(f"{name} must sum to 1")
    return tuple(value / total for value in result)


def evaluate_policy(
    initial: Sequence[float],
    transitions: Sequence[Sequence[Sequence[float]]],
    rewards: Sequence[Sequence[Sequence[float]]],
    policy: Sequence[Sequence[float]],
    horizon: int,
    terminal_states: Collection[int] = (),
) -> FiniteHorizonResult:
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 0:
        raise ValueError("horizon must be a non-negative integer")

    start = _probabilities(initial, "initial distribution")
    state_count = len(start)
    if state_count == 0:
        raise ValueError("initial distribution cannot be empty")

    terminals = _terminal_indices(terminal_states, state_count)
    policy_rows = tuple(tuple(row) for row in policy)
    if len(policy_rows) != state_count:
        raise ValueError("policy must have one row per state")
    if not policy_rows or not policy_rows[0]:
        raise ValueError("policy must contain at least one action")
    action_count = len(policy_rows[0])
    checked_policy = []
    for state, row in enumerate(policy_rows):
        if len(row) != action_count:
            raise ValueError("policy rows must have the same length")
        checked_policy.append(
            _probabilities(row, f"policy row {state}", allow_zero=state in terminals)
        )

    checked_transitions = _check_cube(
        transitions,
        "transitions",
        state_count,
        action_count,
        probabilities=True,
        terminals=terminals,
    )
    checked_rewards = _check_cube(
        rewards,
        "rewards",
        state_count,
        action_count,
        probabilities=False,
        terminals=terminals,
    )

    occupancy = [start]
    current = start
    expected_return = 0.0
    for _ in range(horizon):
        following = [0.0] * state_count
        for state, state_mass in enumerate(current):
            if state in terminals:
                # Keeping terminal mass in place makes the final row a hitting probability.
                following[state] += state_mass
                continue
            for action, action_probability in enumerate(checked_policy[state]):
                flow = state_mass * action_probability
                if flow == 0:
                    continue
                for next_state, transition_probability in enumerate(
                    checked_transitions[state][action]
                ):
                    moved = flow * transition_probability
                    following[next_state] += moved
                    expected_return += (
                        moved * checked_rewards[state][action][next_state]
                    )
        current = tuple(following)
        occupancy.append(current)

    return FiniteHorizonResult(
        expected_return=expected_return,
        state_occupancy=tuple(occupancy),
        terminal_hitting_probabilities={
            state: current[state] for state in sorted(terminals)
        },
    )


def _terminal_indices(states: Collection[int], state_count: int) -> frozenset[int]:
    try:
        values = tuple(states)
    except TypeError as error:
        raise ValueError("terminal states must be a collection of indices") from error
    for state in values:
        if isinstance(state, bool) or not isinstance(state, int):
            raise TypeError("terminal state indices must be integers")
        if not 0 <= state < state_count:
            raise ValueError("terminal state index is out of range")
    return frozenset(values)


def _check_cube(
    cube: Sequence[Sequence[Sequence[float]]],
    name: str,
    state_count: int,
    action_count: int,
    *,
    probabilities: bool,
    terminals: frozenset[int],
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    state_rows = tuple(tuple(actions) for actions in cube)
    if len(state_rows) != state_count:
        raise ValueError(f"{name} must have one row per state")
    checked = []
    for state, actions in enumerate(state_rows):
        if len(actions) != action_count:
            raise ValueError(f"{name} must have one row per state and action")
        checked_actions = []
        for action, row in enumerate(actions):
            values = (
                _probabilities(
                    row,
                    f"{name} row {state}, {action}",
                    allow_zero=state in terminals,
                )
                if probabilities
                else _numbers(row, f"{name} row {state}, {action}")
            )
            if len(values) != state_count:
                raise ValueError(f"{name} rows must have one value per next state")
            checked_actions.append(values)
        checked.append(tuple(checked_actions))
    return tuple(checked)
