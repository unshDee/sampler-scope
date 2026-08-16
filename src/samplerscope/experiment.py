import math
from collections import Counter
from collections.abc import Mapping, Sequence

from samplerscope.decoders import apply_stack, distribution_metrics
from samplerscope.environments import Environment, label_permutations
from samplerscope.mdp import FiniteHorizonResult, evaluate_policy
from samplerscope.trace import LogitTrace

DecoderStep = str | tuple[str, float | int | None]

_SUMMARY_FIELDS = {
    "expected_return": "expected_return",
    "decoder_value_gap": "decoder_value_gap",
    "optimality_gap": "optimality_gap",
    "task_success": "task_success",
    "stakeholder_cost": "stakeholder_cost",
    "state_occupancy_tv": "mean_state_occupancy_tv",
    "policy_total_variation": "total_variation",
    "censored_raw_mass": "censored_raw_mass",
    "optimal_action_censor_rate": "optimal_action_censor_rate",
    "valid_action_mass": "mean_valid_action_mass",
    "logit_tie_rate": "occupancy_weighted_logit_tie_rate",
}


def _optimal_action_indices(environment: Environment) -> tuple[tuple[int, ...], ...]:
    state_count = len(environment.states)
    action_count = len(environment.actions)
    values = [0.0] * state_count
    choices: list[tuple[int, ...]] = [() for _ in range(state_count)]
    rewards = environment.outcomes["return"]

    # Time is part of each state, so one backward pass gives a stationary policy.
    for step in reversed(range(environment.horizon)):
        for state_index, state in enumerate(environment.states):
            if state.terminal is not None or state.step != step:
                continue
            action_values = []
            for action_index in range(action_count):
                action_values.append(
                    sum(
                        probability
                        * (
                            rewards[state_index][action_index][next_index]
                            + values[next_index]
                        )
                        for next_index, probability in enumerate(
                            environment.transitions[state_index][action_index]
                        )
                    )
                )
            best_value = max(action_values)
            choices[state_index] = tuple(
                action
                for action, value in enumerate(action_values)
                if math.isclose(value, best_value, rel_tol=1e-12, abs_tol=1e-12)
            )
            values[state_index] = best_value

    return tuple(choices)


def optimal_policy(environment: Environment) -> tuple[tuple[float, ...], ...]:
    action_count = len(environment.actions)
    rows = []
    for choices in _optimal_action_indices(environment):
        probability = 1.0 / len(choices) if choices else 0.0
        rows.append(
            tuple(
                probability if action in choices else 0.0
                for action in range(action_count)
            )
        )
    return tuple(rows)


def optimal_action_sets(environment: Environment) -> dict[str, tuple[str, ...]]:
    choices = _optimal_action_indices(environment)
    return {
        state.id: tuple(environment.actions[action] for action in choices[index])
        for index, state in enumerate(environment.states)
        if state.terminal is None
    }


def optimal_actions(environment: Environment) -> dict[str, str]:
    return {
        state: actions[0] for state, actions in optimal_action_sets(environment).items()
    }


def synthetic_control_logits(
    environment: Environment, margin: str = "high"
) -> dict[str, dict[str, float]]:
    if margin not in {"high", "low"}:
        raise ValueError("margin must be 'high' or 'low'")
    gap = 6.0 if margin == "high" else 0.08
    best_by_state = optimal_actions(environment)
    rows = {}
    for state in environment.nonterminal_states:
        best = best_by_state[state.id]
        other_logits = iter((0.0, -gap))
        rows[state.id] = {
            action: gap if action == best else next(other_logits)
            for action in environment.actions
        }
    return rows


def _check_logits(
    environment: Environment,
    logits_by_state: Mapping[str, Mapping[str, float]],
) -> None:
    expected_states = {state.id for state in environment.nonterminal_states}
    if set(logits_by_state) != expected_states:
        missing = sorted(expected_states - set(logits_by_state))
        extra = sorted(set(logits_by_state) - expected_states)
        raise ValueError(
            f"logit states do not match environment; missing={missing}, extra={extra}"
        )
    expected_actions = set(environment.actions)
    for state_id, logits in logits_by_state.items():
        if set(logits) != expected_actions or not all(
            math.isfinite(float(value)) for value in logits.values()
        ):
            raise ValueError(f"invalid action logits for state {state_id}")


def _policy_from_logits(
    environment: Environment,
    logits_by_state: Mapping[str, Mapping[str, float]],
    operators: Sequence[DecoderStep],
    token_ids_by_state: Mapping[str, Mapping[str, int]] | None = None,
) -> tuple[tuple[float, ...], ...]:
    rows = []
    for state in environment.states:
        if state.terminal is not None:
            rows.append((0.0, 0.0, 0.0))
            continue
        logits = tuple(
            logits_by_state[state.id][action] for action in environment.actions
        )
        tie_breakers = (
            tuple(
                token_ids_by_state[state.id][action] for action in environment.actions
            )
            if token_ids_by_state
            else None
        )
        rows.append(apply_stack(logits, operators, tie_breakers))
    return tuple(rows)


def _evaluate(
    environment: Environment, policy: Sequence[Sequence[float]]
) -> tuple[dict[str, float], FiniteHorizonResult]:
    values = {}
    main_result = None
    for outcome, rewards in environment.outcomes.items():
        result = evaluate_policy(
            environment.initial,
            environment.transitions,
            rewards,
            policy,
            environment.horizon,
            environment.terminal_states,
        )
        values[outcome] = result.expected_return
        if outcome == "return":
            main_result = result
    if main_result is None:
        raise ValueError("environment has no return outcome")
    return values, main_result


def _decision_weights(
    environment: Environment, result: FiniteHorizonResult
) -> tuple[float, ...]:
    visits = [0.0] * len(environment.states)
    for occupancy in result.state_occupancy[:-1]:
        for state_index, mass in enumerate(occupancy):
            if environment.states[state_index].terminal is None:
                visits[state_index] += mass
    total = sum(visits)
    return tuple(value / total for value in visits)


def _mean_occupancy_tv(
    first: FiniteHorizonResult, second: FiniteHorizonResult
) -> float:
    rows = zip(first.state_occupancy[1:], second.state_occupancy[1:])
    return sum(
        0.5 * sum(abs(left - right) for left, right in zip(a, b)) for a, b in rows
    ) / (len(first.state_occupancy) - 1)


def _operator_json(operators: Sequence[DecoderStep]) -> list[object]:
    return [
        list(operator) if isinstance(operator, tuple) else operator
        for operator in operators
    ]


def analyze_logits(
    environment: Environment,
    logits_by_state: Mapping[str, Mapping[str, float]],
    decoder_stacks: Mapping[str, Sequence[DecoderStep]],
    token_ids_by_state: Mapping[str, Mapping[str, int]] | None = None,
    valid_action_mass_by_state: Mapping[str, float] | None = None,
    best_valid_rank_by_state: Mapping[str, int] | None = None,
) -> list[dict[str, object]]:
    _check_logits(environment, logits_by_state)
    if not decoder_stacks:
        raise ValueError("at least one decoder is required")

    if token_ids_by_state is not None:
        _check_logits(environment, token_ids_by_state)
    state_ids = {state.id for state in environment.nonterminal_states}
    valid_mass = (
        valid_action_mass_by_state
        if valid_action_mass_by_state is not None
        else dict.fromkeys(state_ids, 1.0)
    )
    best_ranks = (
        best_valid_rank_by_state
        if best_valid_rank_by_state is not None
        else dict.fromkeys(state_ids, 1)
    )
    if set(valid_mass) != state_ids or not all(
        0 < value <= 1 for value in valid_mass.values()
    ):
        raise ValueError("invalid valid-action mass rows")
    if set(best_ranks) != state_ids or not all(
        value >= 1 for value in best_ranks.values()
    ):
        raise ValueError("invalid valid-token rank rows")

    raw_policy = _policy_from_logits(
        environment, logits_by_state, (), token_ids_by_state
    )
    raw_values, raw_result = _evaluate(environment, raw_policy)
    weights = _decision_weights(environment, raw_result)
    best_policy = optimal_policy(environment)
    best_actions = _optimal_action_indices(environment)
    optimal_values, _ = _evaluate(environment, best_policy)
    matrix = []

    for decoder_name, operators in decoder_stacks.items():
        policy = _policy_from_logits(
            environment, logits_by_state, operators, token_ids_by_state
        )
        values, result = _evaluate(environment, policy)
        weighted_metrics = {
            name: 0.0
            for name in (
                "total_variation",
                "jensen_shannon",
                "kl_decoded_raw",
                "censored_raw_mass",
            )
        }
        optimal_probability = 0.0
        optimal_censor_rate = 0.0
        weighted_valid_mass = 0.0
        weighted_best_rank = 0.0
        weighted_tie_rate = 0.0
        tie_state_count = 0
        for state_index, state in enumerate(environment.states):
            if state.terminal is not None or weights[state_index] == 0:
                continue
            metrics = distribution_metrics(raw_policy[state_index], policy[state_index])
            for name, value in metrics.items():
                weighted_metrics[name] += weights[state_index] * value
            optimal_probability += weights[state_index] * sum(
                policy[state_index][action] for action in best_actions[state_index]
            )
            if all(
                policy[state_index][action] == 0 for action in best_actions[state_index]
            ):
                optimal_censor_rate += weights[state_index]
            weighted_valid_mass += weights[state_index] * valid_mass[state.id]
            weighted_best_rank += weights[state_index] * best_ranks[state.id]
            tied = len(set(logits_by_state[state.id].values())) < len(
                environment.actions
            )
            tie_state_count += int(tied)
            weighted_tie_rate += weights[state_index] * tied

        matrix.append(
            {
                "environment": environment.name,
                "decoder": decoder_name,
                "operators": _operator_json(operators),
                "expected_return": values["return"],
                "decoder_value_gap": values["return"] - raw_values["return"],
                "optimal_return": optimal_values["return"],
                "optimality_gap": optimal_values["return"] - values["return"],
                "task_success": values["task_success"],
                "task_failure": values["task_failure"],
                "stakeholder_cost": values["stakeholder_cost"],
                "expected_steps": values["steps"],
                "mean_state_occupancy_tv": _mean_occupancy_tv(raw_result, result),
                "mean_optimal_action_probability": optimal_probability,
                "optimal_action_censor_rate": optimal_censor_rate,
                "mean_valid_action_mass": weighted_valid_mass,
                "mean_best_valid_token_rank": weighted_best_rank,
                "raw_logit_tie_state_count": tie_state_count,
                "occupancy_weighted_logit_tie_rate": weighted_tie_rate,
                **weighted_metrics,
            }
        )
    return matrix


def run_experiment(
    environment: Environment,
    trace: LogitTrace,
    decoder_stacks: Mapping[str, Sequence[DecoderStep]],
) -> list[dict[str, object]]:
    trace.validate()
    if trace.actions != environment.actions:
        raise ValueError("trace actions do not match environment actions")

    expected_mappings = {
        tuple(mapping[action] for action in environment.actions)
        for mapping in label_permutations(environment.actions)
    }
    grouped = {}
    for row in trace.rows:
        if row.environment != environment.name:
            continue
        mapping = tuple(row.label_by_action[action] for action in environment.actions)
        grouped.setdefault(mapping, {})[row.state_id] = row
    if set(grouped) != expected_mappings:
        raise ValueError("trace must contain all six label permutations")

    matrix = []
    for mapping in sorted(grouped):
        rows = grouped[mapping]
        logits = {state_id: row.logits_by_action for state_id, row in rows.items()}
        token_ids = {state_id: row.token_id_by_action for state_id, row in rows.items()}
        valid_mass = {state_id: row.valid_action_mass for state_id, row in rows.items()}
        best_ranks = {
            state_id: row.best_valid_token_rank for state_id, row in rows.items()
        }
        analyzed = analyze_logits(
            environment,
            logits,
            decoder_stacks,
            token_ids,
            valid_mass,
            best_ranks,
        )
        label_by_action = dict(zip(environment.actions, mapping, strict=True))
        action_winners = []
        for state_id, state_logits in logits.items():
            scores = tuple(state_logits[action] for action in environment.actions)
            priorities = tuple(
                token_ids[state_id][action] for action in environment.actions
            )
            winner = apply_stack(scores, ("greedy",), priorities).index(1.0)
            action_winners.append(environment.actions[winner])
        action_counts = Counter(action_winners)
        label_counts = Counter(label_by_action[action] for action in action_winners)
        for result in analyzed:
            matrix.append(
                {
                    "model_id": trace.model_id,
                    "label_mapping": "-".join(mapping),
                    "label_by_action": label_by_action,
                    "input_tokens": sum(row.input_tokens for row in rows.values()),
                    "raw_greedy_action_counts": dict(sorted(action_counts.items())),
                    "raw_greedy_label_counts": dict(sorted(label_counts.items())),
                    "raw_dominant_label_fraction": max(label_counts.values())
                    / len(action_winners),
                    **result,
                }
            )
    return matrix


def summarize_results(matrix: Sequence[Mapping[str, object]]) -> dict[str, object]:
    rows = tuple(matrix)
    if not rows:
        raise ValueError("cannot summarize an empty result matrix")

    by_decoder = []
    decoder_names = sorted({str(row["decoder"]) for row in rows})
    for decoder in decoder_names:
        group = [row for row in rows if row["decoder"] == decoder]
        summary: dict[str, object] = {
            "decoder": decoder,
            "label_mappings": len(group),
        }
        for output_name, field in _SUMMARY_FIELDS.items():
            values = [float(row[field]) for row in group]
            summary[f"mean_{output_name}"] = sum(values) / len(values)
        returns = [float(row["expected_return"]) for row in group]
        gaps = [float(row["decoder_value_gap"]) for row in group]
        summary["minimum_expected_return"] = min(returns)
        summary["maximum_expected_return"] = max(returns)
        summary["label_mapping_return_range"] = max(returns) - min(returns)
        summary["minimum_decoder_value_gap"] = min(gaps)
        summary["maximum_decoder_value_gap"] = max(gaps)
        summary["positive_value_gap_count"] = sum(value > 0 for value in gaps)
        summary["negative_value_gap_count"] = sum(value < 0 for value in gaps)
        summary["zero_value_gap_count"] = sum(value == 0 for value in gaps)
        by_decoder.append(summary)

    left_name = "temperature_0.5_then_top_p_0.8"
    right_name = "top_p_0.8_then_temperature_0.5"
    indexed = {(str(row["decoder"]), str(row["label_mapping"])): row for row in rows}
    mappings = sorted(
        str(row["label_mapping"]) for row in rows if row["decoder"] == left_name
    )
    order_effect = None
    if mappings and all((right_name, mapping) in indexed for mapping in mappings):
        deltas = [
            float(indexed[(left_name, mapping)]["expected_return"])
            - float(indexed[(right_name, mapping)]["expected_return"])
            for mapping in mappings
        ]
        order_effect = {
            "left": left_name,
            "right": right_name,
            "paired_label_mappings": len(deltas),
            "mean_return_delta": sum(deltas) / len(deltas),
            "maximum_absolute_return_delta": max(abs(value) for value in deltas),
        }

    return {
        "by_decoder": by_decoder,
        "operator_order_effect": order_effect,
    }
