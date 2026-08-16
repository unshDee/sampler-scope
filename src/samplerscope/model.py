import contextlib
import hashlib
import itertools
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from samplerscope.trace import TraceRow, hash_prompt, hash_token_ids

Messages = Sequence[Mapping[str, str]]
MessageBuilder = Callable[[Mapping[str, str]], Messages]

_CHECKPOINT_FILES = (
    "config.json",
    "generation_config.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def _flat_input_ids(encoded: Any) -> tuple[int, ...]:
    values = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], (list, tuple)):
        if len(values) != 1:
            raise ValueError("expected one prompt")
        values = values[0]
    return tuple(int(value) for value in values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity(
    model_path: str, expected_revision: str | None = None
) -> dict[str, str]:
    root = Path(model_path)
    weight_files = sorted(root.glob("*.safetensors"))
    if not weight_files:
        raise ValueError("model directory has no safetensors checkpoint")

    files = [root / name for name in _CHECKPOINT_FILES if (root / name).is_file()]
    files.extend(weight_files)
    manifest = {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in files
    }
    tree_hash = hashlib.sha256(
        json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()

    metadata_file = root / ".cache/huggingface/download/config.json.metadata"
    revision = None
    if metadata_file.is_file():
        candidate = metadata_file.read_text(encoding="utf-8").splitlines()[0]
        if len(candidate) == 40 and all(
            character in "0123456789abcdef" for character in candidate
        ):
            revision = candidate
    if expected_revision and revision != expected_revision:
        raise ValueError(
            f"local checkpoint revision {revision!r} does not match {expected_revision!r}"
        )

    return {
        "model_revision": revision or "unavailable",
        "checkpoint_tree_sha256": tree_hash,
        "checkpoint_bytes": str(sum(item["bytes"] for item in manifest.values())),
        "checkpoint_files": ",".join(sorted(manifest)),
    }


def _vocabulary_diagnostics(logits: Any, token_ids: Sequence[int]) -> tuple[float, int]:
    if hasattr(logits, "detach"):
        scores = logits.float()
        selected = scores[list(token_ids)]
        log_mass = selected.logsumexp(0) - scores.logsumexp(0)
        mass = float(log_mass.exp().item())
        rank = int((scores > selected.max()).sum().item()) + 1
        return mass, rank

    values = tuple(float(value) for value in logits.tolist())
    largest = max(values)
    denominator = sum(math.exp(value - largest) for value in values)
    selected = tuple(values[token_id] for token_id in token_ids)
    numerator = sum(math.exp(value - largest) for value in selected)
    return numerator / denominator, 1 + sum(value > max(selected) for value in values)


class LocalModel:
    def __init__(self, tokenizer: Any, model: Any, device: str | None = None):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    @classmethod
    def from_pretrained(
        cls, model_path: str, device: str | None = None
    ) -> "LocalModel":
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "local model runs require torch and transformers"
            ) from error

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"

        # local_files_only keeps an experiment run from downloading a different checkpoint.
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        model.to(device)
        model.eval()
        return cls(tokenizer, model, device)

    def render(self, messages: Messages) -> tuple[str, tuple[int, ...]]:
        prompt = self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self.tokenizer(prompt, add_special_tokens=False)
        return prompt, _flat_input_ids(encoded)

    def action_token_ids(
        self,
        messages: Messages,
        labels: Sequence[str],
    ) -> tuple[str, tuple[int, ...], dict[str, int]]:
        if not labels or len(labels) != len(set(labels)):
            raise ValueError("action labels must be non-empty and unique")

        prompt, prompt_ids = self.render(messages)
        token_by_label = {}
        for label in labels:
            combined = self.tokenizer(prompt + label, add_special_tokens=False)
            combined_ids = _flat_input_ids(combined)
            if (
                combined_ids[:-1] != prompt_ids
                or len(combined_ids) != len(prompt_ids) + 1
            ):
                raise ValueError(
                    f"action label {label!r} is not one token in the rendered context"
                )
            token_by_label[label] = combined_ids[-1]

        if len(set(token_by_label.values())) != len(token_by_label):
            raise ValueError("action labels resolve to duplicate token ids")
        return prompt, prompt_ids, token_by_label

    def _score(
        self,
        messages: Messages,
        labels: Sequence[str],
    ) -> tuple[
        dict[str, float],
        str,
        tuple[int, ...],
        dict[str, int],
        float,
        int,
    ]:
        prompt, prompt_ids, token_by_label = self.action_token_ids(messages, labels)
        encoded = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        if self.device:
            encoded = {name: value.to(self.device) for name, value in encoded.items()}

        try:
            import torch

            inference = torch.inference_mode()
        except ImportError:
            # Test doubles do not need torch. Real local models do.
            inference = contextlib.nullcontext()

        with inference:
            output = self.model(**encoded, use_cache=False)
        next_logits = output.logits[0, -1]
        valid_mass, best_rank = _vocabulary_diagnostics(
            next_logits, tuple(token_by_label.values())
        )
        return (
            {
                label: float(next_logits[token_id].item())
                for label, token_id in token_by_label.items()
            },
            prompt,
            prompt_ids,
            token_by_label,
            valid_mass,
            best_rank,
        )

    def trace_permutations(
        self,
        environment: str,
        state_id: str,
        actions: Sequence[str],
        labels: Sequence[str],
        message_builder: MessageBuilder,
    ) -> list[TraceRow]:
        if not actions or len(actions) != len(set(actions)):
            raise ValueError("actions must be non-empty and unique")
        if len(actions) != len(labels):
            raise ValueError("actions and labels must have the same length")

        rows = []
        for permutation in itertools.permutations(labels):
            label_by_action = dict(zip(actions, permutation, strict=True))
            (
                label_logits,
                prompt,
                prompt_ids,
                token_by_label,
                valid_mass,
                best_rank,
            ) = self._score(
                message_builder(label_by_action),
                permutation,
            )
            rows.append(
                TraceRow(
                    environment=environment,
                    state_id=state_id,
                    label_by_action=label_by_action,
                    logits_by_action={
                        action: label_logits[label_by_action[action]]
                        for action in actions
                    },
                    token_id_by_action={
                        action: token_by_label[label_by_action[action]]
                        for action in actions
                    },
                    valid_action_mass=valid_mass,
                    best_valid_token_rank=best_rank,
                    rendered_prompt=prompt,
                    input_token_ids=prompt_ids,
                    prompt_sha256=hash_prompt(prompt),
                    input_sha256=hash_token_ids(prompt_ids),
                    input_tokens=len(prompt_ids),
                )
            )
        return rows
