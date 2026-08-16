import hashlib
import hmac
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

TRACE_FORMAT = "samplerscope.logits"
TRACE_VERSION = 2


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def hash_token_ids(token_ids: list[int] | tuple[int, ...]) -> str:
    return hashlib.sha256(_json_bytes(list(token_ids))).hexdigest()


@dataclass(frozen=True)
class TraceRow:
    environment: str
    state_id: str
    label_by_action: dict[str, str]
    logits_by_action: dict[str, float]
    token_id_by_action: dict[str, int]
    valid_action_mass: float
    best_valid_token_rank: int
    rendered_prompt: str
    input_token_ids: tuple[int, ...]
    prompt_sha256: str
    input_sha256: str
    input_tokens: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceRow":
        return cls(
            environment=str(data["environment"]),
            state_id=str(data["state_id"]),
            label_by_action={
                str(k): str(v) for k, v in data["label_by_action"].items()
            },
            logits_by_action={
                str(k): float(v) for k, v in data["logits_by_action"].items()
            },
            token_id_by_action={
                str(k): int(v) for k, v in data["token_id_by_action"].items()
            },
            valid_action_mass=float(data["valid_action_mass"]),
            best_valid_token_rank=int(data["best_valid_token_rank"]),
            rendered_prompt=str(data["rendered_prompt"]),
            input_token_ids=tuple(int(value) for value in data["input_token_ids"]),
            prompt_sha256=str(data["prompt_sha256"]),
            input_sha256=str(data["input_sha256"]),
            input_tokens=int(data["input_tokens"]),
        )


@dataclass
class LogitTrace:
    model_id: str
    actions: tuple[str, ...]
    labels: tuple[str, ...]
    rows: list[TraceRow] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    def _payload(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "actions": list(self.actions),
            "labels": list(self.labels),
            "metadata": self.metadata,
            "rows": [asdict(row) for row in self.rows],
        }

    def payload_sha256(self) -> str:
        return hashlib.sha256(_json_bytes(self._payload())).hexdigest()

    def validate(self) -> None:
        if not self.actions or len(self.actions) != len(set(self.actions)):
            raise ValueError("actions must be non-empty and unique")
        if len(self.actions) != len(self.labels) or len(self.labels) != len(
            set(self.labels)
        ):
            raise ValueError("labels must be unique and match the action count")

        expected_actions = set(self.actions)
        expected_labels = set(self.labels)
        seen = set()
        for row in self.rows:
            if set(row.label_by_action) != expected_actions:
                raise ValueError(f"bad action mapping for state {row.state_id}")
            if set(row.label_by_action.values()) != expected_labels:
                raise ValueError(f"bad label permutation for state {row.state_id}")
            if set(row.logits_by_action) != expected_actions:
                raise ValueError(f"missing logits for state {row.state_id}")
            if set(row.token_id_by_action) != expected_actions or len(
                set(row.token_id_by_action.values())
            ) != len(expected_actions):
                raise ValueError(f"bad token ids for state {row.state_id}")
            if row.input_tokens < 1 or not all(
                math.isfinite(v) for v in row.logits_by_action.values()
            ):
                raise ValueError(f"invalid trace values for state {row.state_id}")
            if not 0 < row.valid_action_mass <= 1 or row.best_valid_token_rank < 1:
                raise ValueError(
                    f"invalid vocabulary diagnostics for state {row.state_id}"
                )
            if row.input_tokens != len(row.input_token_ids):
                raise ValueError(f"bad input length for state {row.state_id}")
            if (
                hash_prompt(row.rendered_prompt) != row.prompt_sha256
                or hash_token_ids(row.input_token_ids) != row.input_sha256
            ):
                raise ValueError(f"row hashes do not match state {row.state_id}")

            permutation = tuple(row.label_by_action[action] for action in self.actions)
            key = (row.environment, row.state_id, permutation)
            if key in seen:
                raise ValueError(f"duplicate trace row for state {row.state_id}")
            seen.add(key)

    def save(self, path: str | Path) -> None:
        self.validate()
        payload = self._payload()
        envelope = {
            "format": TRACE_FORMAT,
            "version": TRACE_VERSION,
            "sha256": self.payload_sha256(),
            "trace": payload,
        }
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(_json_bytes(envelope) + b"\n")

    @classmethod
    def load(cls, path: str | Path) -> "LogitTrace":
        envelope = json.loads(Path(path).read_text(encoding="utf-8"))
        if (
            envelope.get("format") != TRACE_FORMAT
            or envelope.get("version") != TRACE_VERSION
        ):
            raise ValueError("unsupported trace format")

        payload = envelope["trace"]
        actual_hash = hashlib.sha256(_json_bytes(payload)).hexdigest()
        if not hmac.compare_digest(str(envelope.get("sha256", "")), actual_hash):
            raise ValueError("trace checksum does not match its contents")

        trace = cls(
            model_id=str(payload["model_id"]),
            actions=tuple(str(value) for value in payload["actions"]),
            labels=tuple(str(value) for value in payload["labels"]),
            rows=[TraceRow.from_dict(row) for row in payload["rows"]],
            metadata={str(k): str(v) for k, v in payload.get("metadata", {}).items()},
        )
        trace.validate()
        return trace
