from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


FRAMES = {"typical", "ideal", "self"}
METHODS = {"scalar", "forced_choice", "tradeoff"}
ORDERS = {"original", "mirrored"}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Scenario:
    id: str
    semantic_x: str
    semantic_y: str
    mirror_x: str
    mirror_y: str

    def displayed(self, order: str) -> tuple[str, str, bool]:
        if order == "original":
            return self.semantic_x, self.semantic_y, False
        if order == "mirrored":
            return self.mirror_x, self.mirror_y, True
        raise ConfigError(f"Unknown option order: {order}")


@dataclass(frozen=True)
class ModelConfig:
    id: str
    role: str
    reasoning_effort: str | None
    rates: dict[str, float]


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    name: str
    scenarios: tuple[Scenario, ...]
    scenario_version: str
    frames: tuple[str, ...]
    methods: dict[str, dict[str, Any]]
    orders: tuple[str, ...]
    models: tuple[ModelConfig, ...]
    service_tier: str
    seed: int
    prompt_version: str
    max_concurrency: int
    max_retries: int
    max_output_tokens: int
    budget_usd: float
    total_project_budget_usd: float
    token_estimate: dict[str, float]
    pricing_verified_on: str
    digest: str
    source_hashes: dict[str, str]


def _load_json_yaml(path: Path) -> dict[str, Any]:
    # The config files use JSON syntax so offline planning needs no YAML package.
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"{path} must use JSON-compatible YAML: line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain an object at the top level")
    return data


def _positive_int(data: dict[str, Any], key: str, *, allow_zero: bool = False) -> int:
    value = data.get(key)
    lower = 0 if allow_zero else 1
    if not isinstance(value, int) or isinstance(value, bool) or value < lower:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ConfigError(f"{key} must be a {qualifier} integer")
    return value


def _nonempty_strings(values: Any, key: str) -> tuple[str, ...]:
    if not isinstance(values, list) or not values:
        raise ConfigError(f"{key} must be a non-empty list")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ConfigError(f"{key} must contain non-empty strings")
    if len(values) != len(set(values)):
        raise ConfigError(f"{key} must not contain duplicates")
    return tuple(values)


def _load_scenarios(path: Path) -> tuple[str, dict[str, Scenario], dict[str, Any]]:
    document = _load_json_yaml(path)
    if document.get("schema_version") != 1:
        raise ConfigError("scenarios schema_version must be 1")
    version = document.get("scenario_version")
    if not isinstance(version, str) or not version:
        raise ConfigError("scenario_version must be a non-empty string")
    rows = document.get("scenarios")
    if not isinstance(rows, list) or not rows:
        raise ConfigError("scenarios must be a non-empty list")

    scenarios: dict[str, Scenario] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ConfigError("each scenario must be an object")
        scenario_id = row.get("id")
        outcomes = row.get("outcomes")
        mirror = row.get("mirror_wording")
        confounds = row.get("confounds", [])
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ConfigError("each scenario needs a non-empty id")
        if scenario_id in scenarios:
            raise ConfigError(f"duplicate scenario id: {scenario_id}")
        if not isinstance(outcomes, dict) or not isinstance(mirror, dict):
            raise ConfigError(
                f"{scenario_id}: outcomes and mirror_wording must be objects"
            )
        texts = [outcomes.get("x"), outcomes.get("y"), mirror.get("x"), mirror.get("y")]
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ConfigError(f"{scenario_id}: all outcome wordings must be non-empty")
        if mirror["x"] != outcomes["y"] or mirror["y"] != outcomes["x"]:
            raise ConfigError(f"{scenario_id}: mirror_wording must reverse X and Y")
        if not isinstance(confounds, list) or any(
            not isinstance(item, str) for item in confounds
        ):
            raise ConfigError(f"{scenario_id}: confounds must be a list of strings")
        scenarios[scenario_id] = Scenario(
            id=scenario_id,
            semantic_x=outcomes["x"],
            semantic_y=outcomes["y"],
            mirror_x=mirror["x"],
            mirror_y=mirror["y"],
        )
    return version, scenarios, document


def load_experiment(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    raw = _load_json_yaml(config_path)
    if raw.get("schema_version") != 1:
        raise ConfigError("experiment schema_version must be 1")

    scenario_file = raw.get("scenario_file")
    if not isinstance(scenario_file, str) or not scenario_file:
        raise ConfigError("scenario_file must be a non-empty path")
    scenario_path = (config_path.parent / scenario_file).resolve()
    scenario_version, all_scenarios, scenario_document = _load_scenarios(scenario_path)

    scenario_ids = _nonempty_strings(raw.get("scenario_ids"), "scenario_ids")
    unknown = sorted(set(scenario_ids) - set(all_scenarios))
    if unknown:
        raise ConfigError(f"unknown scenario ids: {', '.join(unknown)}")
    scenarios = tuple(all_scenarios[item] for item in scenario_ids)

    frames = _nonempty_strings(raw.get("frames"), "frames")
    if set(frames) != FRAMES:
        raise ConfigError(f"frames must contain exactly: {', '.join(sorted(FRAMES))}")
    orders = _nonempty_strings(raw.get("orders"), "orders")
    if set(orders) != ORDERS:
        raise ConfigError(f"orders must contain exactly: {', '.join(sorted(ORDERS))}")

    methods = raw.get("methods")
    if not isinstance(methods, dict) or set(methods) != METHODS:
        raise ConfigError(f"methods must contain exactly: {', '.join(sorted(METHODS))}")
    for method, settings in methods.items():
        if not isinstance(settings, dict):
            raise ConfigError(f"{method} settings must be an object")
        _positive_int(settings, "repetitions")
    levels = methods["tradeoff"].get("donation_amounts_usd")
    if (
        not isinstance(levels, list)
        or len(levels) < 2
        or any(
            not isinstance(level, (int, float)) or isinstance(level, bool) or level < 0
            for level in levels
        )
        or levels != sorted(set(levels))
        or levels[0] != 0
    ):
        raise ConfigError(
            "tradeoff donation_amounts_usd must be unique, increasing, and start at 0"
        )

    model_rows = raw.get("models")
    if not isinstance(model_rows, list) or not model_rows:
        raise ConfigError("models must be a non-empty list")
    models: list[ModelConfig] = []
    for row in model_rows:
        if not isinstance(row, dict):
            raise ConfigError("each model must be an object")
        model_id = row.get("id")
        role = row.get("role")
        rates = row.get("rates_usd_per_million")
        if not isinstance(model_id, str) or not model_id:
            raise ConfigError("each model needs a non-empty id")
        if not isinstance(role, str) or not role:
            raise ConfigError(f"{model_id}: role must be a non-empty string")
        if not isinstance(rates, dict):
            raise ConfigError(f"{model_id}: rates_usd_per_million must be an object")
        for rate_name in ("input", "cached_input", "output"):
            rate = rates.get(rate_name)
            if not isinstance(rate, (int, float)) or isinstance(rate, bool) or rate < 0:
                raise ConfigError(f"{model_id}: missing or invalid {rate_name} rate")
        cache_write = rates.get("cache_write", 0)
        if (
            not isinstance(cache_write, (int, float))
            or isinstance(cache_write, bool)
            or cache_write < 0
        ):
            raise ConfigError(f"{model_id}: invalid cache_write rate")
        reasoning = row.get("reasoning_effort")
        if reasoning is not None and (not isinstance(reasoning, str) or not reasoning):
            raise ConfigError(f"{model_id}: reasoning_effort must be a string when set")
        if model_id == "gpt-5.6-luna" and reasoning != "none":
            raise ConfigError("gpt-5.6-luna must explicitly use reasoning_effort none")
        if model_id == "gpt-4o-mini-2024-07-18" and reasoning is not None:
            raise ConfigError("gpt-4o-mini-2024-07-18 must omit reasoning_effort")
        models.append(
            ModelConfig(
                id=model_id,
                role=role,
                reasoning_effort=reasoning,
                rates={key: float(value) for key, value in rates.items()},
            )
        )
    if len({model.id for model in models}) != len(models):
        raise ConfigError("model ids must be unique")
    if sum(model.role == "primary" for model in models) != 1:
        raise ConfigError("models must contain exactly one primary role")

    token_estimate = raw.get("token_estimate")
    if not isinstance(token_estimate, dict):
        raise ConfigError("token_estimate must be an object")
    for key in ("chars_per_token", "request_overhead", "output_per_call"):
        value = token_estimate.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ConfigError(f"token_estimate.{key} must be positive")

    name = raw.get("name")
    prompt_version = raw.get("prompt_version")
    service_tier = raw.get("service_tier")
    pricing_verified_on = raw.get("verified_on")
    for key, value in (
        ("name", name),
        ("prompt_version", prompt_version),
        ("service_tier", service_tier),
        ("verified_on", pricing_verified_on),
    ):
        if not isinstance(value, str) or not value:
            raise ConfigError(f"{key} must be a non-empty string")

    if service_tier != "default":
        raise ConfigError(
            "service_tier must be default so configured standard rates apply"
        )

    budget = raw.get("budget_usd")
    total_budget = raw.get("total_project_budget_usd")
    if not isinstance(budget, (int, float)) or isinstance(budget, bool) or budget <= 0:
        raise ConfigError("budget_usd must be positive")
    if (
        not isinstance(total_budget, (int, float))
        or isinstance(total_budget, bool)
        or total_budget < budget
    ):
        raise ConfigError("total_project_budget_usd must be at least budget_usd")

    canonical = json.dumps(
        {"experiment": raw, "scenarios": scenario_document},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    source_hashes = {
        "experiment": sha256(config_path.read_bytes()).hexdigest(),
        "scenarios": sha256(scenario_path.read_bytes()).hexdigest(),
    }
    max_output_tokens = _positive_int(raw, "max_output_tokens")
    if max_output_tokens < 16:
        raise ConfigError("max_output_tokens must be at least 16 for the Responses API")

    return ExperimentConfig(
        path=config_path,
        name=name,
        scenarios=scenarios,
        scenario_version=scenario_version,
        frames=frames,
        methods=methods,
        orders=orders,
        models=tuple(models),
        service_tier=service_tier,
        seed=_positive_int(raw, "seed", allow_zero=True),
        prompt_version=prompt_version,
        max_concurrency=_positive_int(raw, "max_concurrency"),
        max_retries=_positive_int(raw, "max_retries", allow_zero=True),
        max_output_tokens=max_output_tokens,
        budget_usd=float(budget),
        total_project_budget_usd=float(total_budget),
        token_estimate={key: float(value) for key, value in token_estimate.items()},
        pricing_verified_on=pricing_verified_on,
        digest=sha256(canonical).hexdigest(),
        source_hashes=source_hashes,
    )
