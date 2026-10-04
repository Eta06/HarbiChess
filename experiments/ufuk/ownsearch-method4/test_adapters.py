"""Synthetic native-schema/counter rejection only; no queries or optimizer."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "own_controller", Path(__file__).with_name("ownv1_training_controller.py")
)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class AdapterTests(unittest.TestCase):
    def test_mc_completed_latency_barrier_rejects_running_late_or_mutated_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ["replay-20261205", "replay-20261206", "eligibility", "cuda-parity", "latency"]
            rows = []
            for name in names:
                path = root / (name + "-process-result.json")
                path.write_text(
                    json.dumps({"returncode": 0, "finished_epoch": 10, "deadline_epoch": 20})
                )
                rows.append({"path": str(path), "sha256": c.sha(path)})
            latency = root / "latency.json"
            # Gate failure is allowed after timing COMPLETED; scheduler reads no outcomes.
            latency.write_text('{"speed_pass": false}')
            barrier = {
                "process_receipts": rows,
                "coordinator_sha256": (
                    "686265b09284d1e8c879406510910991d2d8f1a71bf60b610372eac00e76e100"
                ),
                "latency_receipt": {"path": str(latency), "sha256": c.sha(latency)},
            }
            c.validate_mc_barrier(barrier, 11)
            with self.assertRaises(AssertionError):
                c.validate_mc_barrier(barrier, 9)
            rows[-1]["sha256"] = "0" * 64
            with self.assertRaises(AssertionError):
                c.validate_mc_barrier(barrier, 11)
            rows[-1]["sha256"] = c.sha(rows[-1]["path"])
            path = Path(rows[-1]["path"])
            path.write_text(
                json.dumps({"returncode": -15, "finished_epoch": 10, "deadline_epoch": 20})
            )
            rows[-1]["sha256"] = c.sha(path)
            with self.assertRaises(AssertionError):
                c.validate_mc_barrier(barrier, 11)

    def test_rejects_mc_native_instead_of_relabeling40(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            native = run / "checkpoints/epoch-00000024"
            native.mkdir(parents=True)
            (native / "actor.json").write_text("{}")
            (native / "checkpoint.json").write_text(
                json.dumps({"schema": "torch-fullgame-native-cuda-v1"})
            )
            with self.assertRaises(AssertionError):
                c.validate_native(run, 24, "a" * 40)

    def test_ownv1_exact_epoch_counters_and_closed_search_rng_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            native = run / "checkpoints/epoch-00000024"
            native.mkdir(parents=True)
            state = {
                "epoch": 24,
                "actor_steps": 6144,
                "fresh_transitions": 786432,
                "pending_search_schedule": "closed-empty",
                "replay_buffer": "closed-empty",
                "training_pass": "closed",
                "search_rngs": [None] * 128,
            }
            manifest = {
                "schema": "torch-ownsearch-native-cuda-v1",
                "run_config": {"schema": "torch-fresh-sparse-ownsearch-v1"},
                "source_commit": "a" * 40,
                "state": state,
                "artifacts": {},
            }

            def save():
                (native / "actor.json").write_text(json.dumps(state))
                for name in (
                    "model.safetensors",
                    "base.safetensors",
                    "behavior.safetensors",
                    "training.pt",
                    "last-frozen-epoch.json.gz",
                ):
                    (native / name).write_bytes(
                        b"synthetic-hash-only-controller-test-not-real-native"
                    )
                manifest["artifacts"] = {
                    name: c.sha(native / name)
                    for name in (
                        "model.safetensors",
                        "base.safetensors",
                        "behavior.safetensors",
                        "training.pt",
                        "actor.json",
                        "last-frozen-epoch.json.gz",
                    )
                }
                (native / "checkpoint.json").write_text(json.dumps(manifest))

            save()
            c.validate_native(run, 24, "a" * 40)
            state["fresh_transitions"] = 1310720
            save()
            with self.assertRaises(AssertionError):
                c.validate_native(run, 24, "a" * 40)
            state["fresh_transitions"] = 786432
            state["search_rngs"] = [None] * 127
            save()
            with self.assertRaises(AssertionError):
                c.validate_native(run, 24, "a" * 40)


if __name__ == "__main__":
    unittest.main()
