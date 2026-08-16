import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from samplerscope.model import LocalModel, checkpoint_identity
from samplerscope.trace import LogitTrace, TraceRow, hash_prompt, hash_token_ids


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def to(self, _device):
        return self

    def tolist(self):
        return self.values

    def __getitem__(self, index):
        if isinstance(index, tuple):
            row, column = index
            return FakeTensor(self.values[row][column])
        value = self.values[index]
        return (
            FakeScalar(value) if isinstance(value, (int, float)) else FakeTensor(value)
        )


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return "|".join(message["content"] for message in messages) + "|answer:"

    def __call__(self, text, add_special_tokens, return_tensors=None):
        assert not add_special_tokens
        ids = [ord(character) for character in text]
        return (
            {"input_ids": FakeTensor([ids])} if return_tensors else {"input_ids": ids}
        )


class FakeModel:
    def __call__(self, input_ids, use_cache):
        assert not use_cache
        vocab = [float(value) for value in range(128)]
        return SimpleNamespace(
            logits=FakeTensor([[vocab for _ in input_ids.values[0]]])
        )


class TraceTests(unittest.TestCase):
    def test_action_labels_use_exact_rendered_context(self):
        adapter = LocalModel(FakeTokenizer(), FakeModel())
        messages = [{"role": "user", "content": "Pick A, B, or C"}]

        _, _, token_ids = adapter.action_token_ids(messages, ("A", "B", "C"))

        self.assertEqual(token_ids, {"A": 65, "B": 66, "C": 67})
        with self.assertRaisesRegex(ValueError, "not one token"):
            adapter.action_token_ids(messages, ("A", "BC"))

    def test_all_label_permutations_are_traced(self):
        adapter = LocalModel(FakeTokenizer(), FakeModel())
        actions = ("left", "right", "wait")
        labels = ("A", "B", "C")

        rows = adapter.trace_permutations(
            "fork",
            "start",
            actions,
            labels,
            lambda mapping: [
                {
                    "role": "user",
                    "content": ", ".join(
                        f"{label}={action}" for action, label in mapping.items()
                    ),
                }
            ],
        )

        self.assertEqual(len(rows), 6)
        self.assertEqual(
            {tuple(row.label_by_action[action] for action in actions) for row in rows},
            {
                ("A", "B", "C"),
                ("A", "C", "B"),
                ("B", "A", "C"),
                ("B", "C", "A"),
                ("C", "A", "B"),
                ("C", "B", "A"),
            },
        )
        self.assertEqual(
            rows[0].logits_by_action,
            {"left": 65.0, "right": 66.0, "wait": 67.0},
        )
        self.assertEqual(
            rows[0].token_id_by_action,
            {"left": 65, "right": 66, "wait": 67},
        )
        self.assertGreater(rows[0].valid_action_mass, 0)
        self.assertEqual(rows[0].best_valid_token_rank, 61)

    def test_trace_round_trip_and_checksum(self):
        row = TraceRow(
            environment="fork",
            state_id="start",
            label_by_action={"left": "A", "right": "B"},
            logits_by_action={"left": 1.25, "right": -0.5},
            token_id_by_action={"left": 10, "right": 11},
            valid_action_mass=0.25,
            best_valid_token_rank=4,
            rendered_prompt="prompt",
            input_token_ids=(1, 2, 3),
            prompt_sha256=hash_prompt("prompt"),
            input_sha256=hash_token_ids((1, 2, 3)),
            input_tokens=3,
        )
        trace = LogitTrace(
            "fake",
            ("left", "right"),
            ("A", "B"),
            [row],
            {"revision": "test"},
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            trace.save(path)

            self.assertEqual(LogitTrace.load(path), trace)
            saved = json.loads(path.read_text())
            self.assertEqual(saved["sha256"], trace.payload_sha256())
            saved["trace"]["rows"][0]["input_tokens"] = 4
            path.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, "checksum"):
                LogitTrace.load(path)

    def test_checkpoint_identity_verifies_the_local_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors").write_bytes(b"weights")
            (root / "config.json").write_text("{}", encoding="utf-8")
            metadata = root / ".cache/huggingface/download"
            metadata.mkdir(parents=True)
            revision = "1" * 40
            (metadata / "config.json.metadata").write_text(
                f"{revision}\netag\n0\n", encoding="utf-8"
            )

            identity = checkpoint_identity(str(root), revision)
            self.assertEqual(identity["model_revision"], revision)
            self.assertEqual(len(identity["checkpoint_tree_sha256"]), 64)
            with self.assertRaisesRegex(ValueError, "does not match"):
                checkpoint_identity(str(root), "2" * 40)


if __name__ == "__main__":
    unittest.main()
