from collections import defaultdict
from collections.abc import Callable
from itertools import combinations
import json
import math
from pathlib import Path
import random
from statistics import fmean, median
from typing import Any

from .parsing import estimate_switch_threshold, paired_threshold_scale


FRAME_LABELS = {"typical": "A", "ideal": "I", "self": "S"}


def normalized_pull(
    a: float, i: float, s: float, minimum_gap: float = 10.0
) -> dict[str, Any]:
    gap = i - a
    # Tiny anchor gaps make the ratio unstable, so keep them as missing.
    if abs(gap) < minimum_gap:
        return {"value": None, "identifiable": False, "anchor_gap": gap}
    return {"value": (s - a) / gap, "identifiable": True, "anchor_gap": gap}


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile needs at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_omega(
    a_values: list[float],
    i_values: list[float],
    s_values: list[float],
    *,
    samples: int = 2000,
    seed: int = 20260815,
    minimum_gap: float = 10.0,
) -> list[float] | None:
    if not a_values or not i_values or not s_values or samples <= 0:
        return None
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        means = []
        for values in (a_values, i_values, s_values):
            means.append(fmean(rng.choice(values) for _ in range(len(values))))
        estimate = normalized_pull(*means, minimum_gap=minimum_gap)
        if estimate["identifiable"]:
            estimates.append(estimate["value"])
    if not estimates:
        return None
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def bootstrap_statistic(
    values: list[float],
    statistic: Callable[[list[float]], float],
    *,
    samples: int = 2000,
    seed: int = 20260815,
) -> list[float] | None:
    if not values or samples <= 0:
        return None
    rng = random.Random(seed)
    estimates = [
        float(statistic([rng.choice(values) for _ in range(len(values))]))
        for _ in range(samples)
    ]
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2 + 1
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    rx, ry = _ranks(x), _ranks(y)
    mx, my = fmean(rx), fmean(ry)
    numerator = sum((left - mx) * (right - my) for left, right in zip(rx, ry))
    denominator = math.sqrt(
        sum((value - mx) ** 2 for value in rx) * sum((value - my) ** 2 for value in ry)
    )
    return numerator / denominator if denominator else None


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            records.append(record)
    return records


def _tradeoff_observations(
    records: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str, str], list[float]], list[dict[str, Any]]]:
    groups: dict[
        tuple[str, str, str, int], dict[str, list[tuple[float, int | None]]]
    ] = defaultdict(lambda: {"original": [], "mirrored": []})
    for record in records:
        if (
            record.get("method") != "tradeoff"
            or record.get("api_status") != "completed"
        ):
            continue
        key = (
            str(record["model"]),
            str(record["scenario_id"]),
            str(record["frame"]),
            int(record["repetition"]),
        )
        raw = (
            record.get("parsed_raw_value")
            if record.get("parse_status") == "ok"
            else None
        )
        groups[key][str(record["order"])].append((float(record["donation_usd"]), raw))

    observations: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    diagnostics: list[dict[str, Any]] = []
    for (model, scenario, frame, repetition), by_order in sorted(groups.items()):
        # Original subsidizes semantic Y; mirrored subsidizes semantic X.
        threshold_y = estimate_switch_threshold(by_order["original"])
        threshold_x = estimate_switch_threshold(by_order["mirrored"])
        all_amounts = [amount for rows in by_order.values() for amount, _ in rows]
        maximum = max(all_amounts) if all_amounts else 0.0
        score = paired_threshold_scale(
            threshold_to_y=threshold_y,
            threshold_to_x=threshold_x,
            max_donation_usd=maximum,
        )
        if score is not None:
            observations[(model, scenario, frame)].append(score)
        diagnostics.append(
            {
                "model": model,
                "scenario_id": scenario,
                "frame": frame,
                "repetition": repetition,
                "threshold_to_y_usd": threshold_y.amount_usd,
                "threshold_to_x_usd": threshold_x.amount_usd,
                "threshold_to_y_censored_above": threshold_y.censored_above,
                "threshold_to_x_censored_above": threshold_x.censored_above,
                "coherent": threshold_y.coherent and threshold_x.coherent,
                "shared_scale_value": score,
                "errors": [
                    error for error in (threshold_y.error, threshold_x.error) if error
                ],
            }
        )
    return observations, diagnostics


def _response_quality(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for record in records:
        if record.get("api_status") != "completed":
            continue
        key = (str(record.get("model")), str(record.get("method")))
        counts[key]["total"] += 1
        status = str(record.get("parse_status"))
        counts[key][status] += 1
    result = []
    for (model, method), row in sorted(counts.items()):
        total = row["total"]
        invalid = row["invalid_format"] + row["invalid_value"]
        result.append(
            {
                "model": model,
                "method": method,
                "total": total,
                "invalid_rate": invalid / total,
                "indeterminate_rate": row["indeterminate"] / total,
            }
        )
    return result


def _order_sensitivity(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[float]] = defaultdict(list)
    for record in records:
        if (
            record.get("api_status") != "completed"
            or record.get("parse_status") != "ok"
        ):
            continue
        method = str(record["method"])
        if method == "tradeoff" and float(record.get("donation_usd", -1)) != 0:
            continue
        key = (
            str(record["model"]),
            str(record["scenario_id"]),
            method,
            str(record["frame"]),
            str(record["order"]),
        )
        groups[key].append(float(record["parsed_semantic_value"]))

    result = []
    bases = sorted({key[:4] for key in groups})
    for base in bases:
        original = groups.get((*base, "original"), [])
        mirrored = groups.get((*base, "mirrored"), [])
        if not original or not mirrored:
            continue
        result.append(
            {
                "model": base[0],
                "scenario_id": base[1],
                "method": base[2],
                "frame": base[3],
                "mirrored_minus_original": fmean(mirrored) - fmean(original),
                "note": "Tradeoff order sensitivity uses only the USD 0 level."
                if base[2] == "tradeoff"
                else None,
            }
        )
    return result


def analyze_records(
    records: list[dict[str, Any]], *, bootstrap_samples: int = 2000
) -> dict[str, Any]:
    observations: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for record in records:
        if (
            record.get("method") in {"scalar", "forced_choice"}
            and record.get("api_status") == "completed"
            and record.get("parse_status") == "ok"
        ):
            key = (
                str(record["model"]),
                str(record["scenario_id"]),
                str(record["method"]),
                str(record["frame"]),
            )
            observations[key].append(float(record["parsed_semantic_value"]))

    tradeoff, diagnostics = _tradeoff_observations(records)
    for (model, scenario, frame), values in tradeoff.items():
        observations[(model, scenario, "tradeoff", frame)].extend(values)

    bases = sorted({key[:3] for key in observations})
    estimates: list[dict[str, Any]] = []
    omega_index: dict[tuple[str, str, str], float] = {}
    for model, scenario, method in bases:
        frame_values = {
            FRAME_LABELS[frame]: observations.get((model, scenario, method, frame), [])
            for frame in FRAME_LABELS
        }
        if any(not values for values in frame_values.values()):
            continue
        means = {label: fmean(values) for label, values in frame_values.items()}
        omega = normalized_pull(means["A"], means["I"], means["S"])
        ci = bootstrap_omega(
            frame_values["A"],
            frame_values["I"],
            frame_values["S"],
            samples=bootstrap_samples,
            seed=20260815 + sum(ord(char) for char in f"{model}{scenario}{method}"),
        )
        row = {
            "model": model,
            "scenario_id": scenario,
            "method": method,
            "A": means["A"],
            "I": means["I"],
            "S": means["S"],
            "omega_hat": omega["value"],
            "omega_identifiable": omega["identifiable"],
            "anchor_gap": omega["anchor_gap"],
            "omega_bootstrap_95_ci": ci,
            "frame_mean_bootstrap_95_ci": {
                label: bootstrap_statistic(
                    values,
                    fmean,
                    samples=bootstrap_samples,
                    seed=20260815
                    + sum(ord(char) for char in f"{model}{scenario}{method}{label}"),
                )
                for label, values in frame_values.items()
            },
            "valid_observations": {
                label: len(values) for label, values in frame_values.items()
            },
        }
        estimates.append(row)
        if omega["identifiable"]:
            omega_index[(model, scenario, method)] = omega["value"]

    cross_method = []
    models = sorted({key[0] for key in omega_index})
    methods = sorted({key[2] for key in omega_index})
    for model in models:
        for left, right in combinations(methods, 2):
            scenarios = sorted(
                {key[1] for key in omega_index if key[0] == model and key[2] == left}
                & {key[1] for key in omega_index if key[0] == model and key[2] == right}
            )
            left_values = [
                omega_index[(model, scenario, left)] for scenario in scenarios
            ]
            right_values = [
                omega_index[(model, scenario, right)] for scenario in scenarios
            ]
            cross_method.append(
                {
                    "model": model,
                    "left_method": left,
                    "right_method": right,
                    "scenario_count": len(scenarios),
                    "spearman": spearman(left_values, right_values),
                    "median_absolute_difference": (
                        median(abs(a - b) for a, b in zip(left_values, right_values))
                        if scenarios
                        else None
                    ),
                }
            )

    method_summary = []
    for model in models:
        for method in methods:
            values = [
                value
                for (m, _, meth), value in omega_index.items()
                if m == model and meth == method
            ]
            if values:
                method_summary.append(
                    {
                        "model": model,
                        "method": method,
                        "scenario_count": len(values),
                        "median_omega_hat": median(values),
                        "mean_omega_hat": fmean(values),
                        "median_omega_bootstrap_95_ci": bootstrap_statistic(
                            values,
                            median,
                            samples=bootstrap_samples,
                            seed=20260815
                            + sum(ord(char) for char in f"{model}{method}median"),
                        ),
                        "mean_omega_bootstrap_95_ci": bootstrap_statistic(
                            values,
                            fmean,
                            samples=bootstrap_samples,
                            seed=20260815
                            + sum(ord(char) for char in f"{model}{method}mean"),
                        ),
                    }
                )

    model_differences = []
    if len(models) > 1:
        for left_model, right_model in combinations(models, 2):
            shared = sorted(
                {
                    (scenario, method)
                    for model, scenario, method in omega_index
                    if model == left_model
                }
                & {
                    (scenario, method)
                    for model, scenario, method in omega_index
                    if model == right_model
                }
            )
            for scenario, method in shared:
                model_differences.append(
                    {
                        "left_model": left_model,
                        "right_model": right_model,
                        "scenario_id": scenario,
                        "method": method,
                        "right_minus_left_omega": (
                            omega_index[(right_model, scenario, method)]
                            - omega_index[(left_model, scenario, method)]
                        ),
                    }
                )

    return {
        "record_count": len(records),
        "scenario_estimates": estimates,
        "method_summary": method_summary,
        "cross_method": cross_method,
        "response_quality": _response_quality(records),
        "order_sensitivity": _order_sensitivity(records),
        "model_differences": model_differences,
        "tradeoff_diagnostics": diagnostics,
    }


def write_analysis(
    raw_path: str | Path,
    output_path: str | Path,
    *,
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    result = analyze_records(read_jsonl(raw_path), bootstrap_samples=bootstrap_samples)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
