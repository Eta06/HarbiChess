"""Pure native-boundary receipt tests; no model training, jobs or remote calls."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "controller",
    Path(__file__).with_name("a100-mc-whole-training-controller-v3-18000.py"),
)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class Tests(unittest.TestCase):
    def fixture(self, root):
        native = root / "run/checkpoints/epoch-00000040"
        native.mkdir(parents=True)
        journal = root / "run/journal"
        journal.mkdir()
        (journal / "epoch-00000040.json.gz").write_bytes(b"synthetic")
        state = {
            "epoch": 40,
            "actor_steps": 10240,
            "fresh_transitions": 1310720,
            "replay_buffer": "closed-empty",
            "ppo_pass": "closed",
            "optimizer_accepted_updates": 5,
            "optimizer_attempted_updates": 10,
            "optimizer_rejected_updates": 5,
            "sample_chain_sha256": "b" * 64,
        }
        for file, value in [
            ("actor.json", json.dumps(state)),
            ("model.safetensors", "synthetic-model"),
        ]:
            (native / file).write_text(value)
        metadata = {
            "source_commit": "a" * 40,
            "config": {"seed": 20261205},
            "absolute_deadline_epoch": 2000,
            "max_epochs": 40,
            "checkpoint_interval": 1,
            "inputs": {"initial_weights": {"sha256": "e" * 64}},
        }
        (root / "run/metadata.json").write_text(json.dumps(metadata))
        manifest = {
            "schema": "torch-fullgame-native-cuda-v1",
            "source_commit": "a" * 40,
            "state": state,
            "run_config": {
                "config": {"seed": 20261205},
                "input_sha256": {"initial_weights": "e" * 64},
            },
            "artifacts": {
                name: c.sha(native / name)
                for name in ("actor.json", "model.safetensors")
            },
        }
        (native / "checkpoint.json").write_text(json.dumps(manifest))
        # Final progress intentionally has actor_steps, not fresh_transitions,
        # matching real runner terminal receipt and retained dev wrapper bug.
        progress = {
            "status": "completed",
            "epoch": 40,
            "actor_steps": 10240,
            "closed_boundary": True,
            "absolute_deadline_epoch": 2000,
            "finished_epoch": 1999,
            "source_commit": "a" * 40,
        }
        progress.update(
            {
                name: state[name]
                for name in (
                    "optimizer_accepted_updates",
                    "optimizer_attempted_updates",
                    "optimizer_rejected_updates",
                    "sample_chain_sha256",
                )
            }
        )
        (root / "run/progress.json").write_text(json.dumps(progress))
        return native, state

    def test_final_progress_without_fresh_transitions_uses_native_counter(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, state = self.fixture(root)
            self.assertEqual(
                c.verify_boundary(root, 20261205, "a" * 40, 40, 2000), state
            )

    def test_native_bit_artifact_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            native, _ = self.fixture(root)
            (native / "model.safetensors").write_text("corrupt")
            with self.assertRaises(AssertionError):
                c.verify_boundary(root, 20261205, "a" * 40, 40, 2000)

    def test_incorrect_seed_or_original_deadline_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root)
            for seed, deadline in ((20261206, 2000), (20261205, 2001)):
                with self.assertRaises(AssertionError):
                    c.verify_boundary(root, seed, "a" * 40, 40, deadline)

    def test_terminal_progress_source_or_native_chain_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root)
            path = root / "run/progress.json"
            original = json.loads(path.read_text())
            for field, value in (
                ("source_commit", "c" * 40),
                ("sample_chain_sha256", "c" * 64),
                ("optimizer_accepted_updates", 6),
            ):
                changed = dict(original)
                changed[field] = value
                path.write_text(json.dumps(changed))
                with self.assertRaises(AssertionError):
                    c.verify_boundary(root, 20261205, "a" * 40, 40, 2000)


if __name__ == "__main__":
    unittest.main()
