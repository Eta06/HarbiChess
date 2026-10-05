"""Pure launcher guard tests; child processes are never started here."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from launch_pst import (
    check_parent_guard,
    digest,
    make_command,
    sha,
    validate_clock,
    validate_proof_receipt,
    validate_ram_root,
)


class ClockTests(unittest.TestCase):
    def test_phase_clock_is_original_and_nonextendable(self):
        protocol = {
            "phase_clocks": {
                "proof": {"first_epoch": 10.0, "deadline_epoch": 610.0},
                "fit": {"first_epoch": 700.0, "deadline_epoch": 2500.0},
            },
            "deadline_by_seed": {"20262905": 2500.0, "20262906": 2500.0},
        }
        self.assertEqual(validate_clock(protocol, "proof", 10.0), (10.0, 610.0))
        with self.assertRaises(TimeoutError):
            validate_clock(protocol, "proof", 610.0)
        self.assertEqual(validate_clock(protocol, "fit", 1800.0), (700.0, 2500.0))
        protocol["phase_clocks"]["fit"] = {
            "first_epoch": 609.0,
            "deadline_epoch": 2409.0,
        }
        protocol["deadline_by_seed"] = {
            "20262905": 2409.0,
            "20262906": 2409.0,
        }
        with self.assertRaises(ValueError):
            validate_clock(protocol, "fit", 1800.0)
        protocol["phase_clocks"]["fit"] = {
            "first_epoch": 700.0,
            "deadline_epoch": 2500.0,
        }
        protocol["deadline_by_seed"] = {
            "20262905": 2500.0,
            "20262906": 2500.0,
        }
        protocol["deadline_by_seed"]["20262906"] = 2501.0
        with self.assertRaises(ValueError):
            validate_clock(protocol, "fit", 1800.0)

    def test_ram_root_must_be_new_child_of_dev_shm(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as parent:
            candidate = Path(parent) / "new"
            self.assertEqual(validate_ram_root(candidate), candidate.resolve())
            candidate.mkdir()
            with self.assertRaises(ValueError):
                validate_ram_root(candidate)
        with tempfile.TemporaryDirectory() as not_shm, self.assertRaises(ValueError):
            validate_ram_root(Path(not_shm) / "new")


class GuardTests(unittest.TestCase):
    def test_parent_runtime_scopes_memory_artifacts_per_seed(self):
        class FakeRuntime:
            def __init__(self):
                self.paths = []

            def guard(self, deadline, source, artifacts):
                self.paths.append(Path(artifacts))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ram"
            seed = root / "20262905"
            root.mkdir()
            seed.mkdir()
            fake = FakeRuntime()
            check_parent_guard(fake, 100, Path(tmp), root, seed)
            self.assertEqual(fake.paths, [seed])
            with patch("launch_pst.TOTAL_ARTIFACT_BYTES", 1):
                (root / "x").write_bytes(b"12")
                with self.assertRaises(RuntimeError):
                    check_parent_guard(fake, 100, Path(tmp), root, seed)

    def test_command_preserves_output_scope_and_resume_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol_path = root / "protocol.json"
            protocol_path.write_text("{}")
            resume = root / "step-4.native.gz"
            resume.write_bytes(b"checkpoint")
            class Args:
                protocol = protocol_path
                protocol_sha256 = "pinned"
                source_repo = root / "source"
                cpu_core = 1
                phase = "proof"

            protocol = {
                "config_paths": {"20262905": str(root / "config.json")},
                "journal_paths": {"20262905": str(root / "journal.json.gz")},
            }
            cmd = make_command(
                Args(), protocol, 20262905, 42.0,
                output=root / "seed" / "split", stop=8,
                resume=(resume, "checkpoint-sha"),
            )
            self.assertIn("--phase", cmd)
            self.assertIn("proof", cmd)
            self.assertEqual(cmd[cmd.index("--output") + 1], str(root / "seed" / "split"))
            self.assertEqual(cmd[cmd.index("--resume-sha256") + 1], "checkpoint-sha")

    def test_fit_requires_complete_on_time_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "proof.json"
            import hashlib
            import json

            receipt = {
                "schema": "classical-own-pst-proof-result-v1",
                "status": "PASS-two-seed-pause-resume-and-strict-loads",
                "protocol_sha256": "protocol-sha",
                "strict_native_loads": 6,
                "phase_first_epoch": 10.0,
                "phase_deadline_epoch": 610.0,
                "started_epoch": 12.0,
                "finished_epoch": 15.0,
                "seeds": {
                    "20262905": {
                        "whole_equals_pause_resume": True,
                        "strict_loaded_steps": [0, 4, 8],
                        "state_sha256_by_step": {
                            str(step): digest({"step": step}) for step in (0, 4, 8)
                        },
                    },
                    "20262906": {
                        "whole_equals_pause_resume": True,
                        "strict_loaded_steps": [0, 4, 8],
                        "state_sha256_by_step": {
                            str(step): digest({"step": step}) for step in (0, 4, 8)
                        },
                    },
                },
            }
            path.write_text(json.dumps(receipt))
            proof_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            proof_root = path.parent
            (proof_root / "owner-process.json").write_text(
                json.dumps({
                    "phase": "proof",
                    "protocol_sha256": "protocol-sha",
                    "original_deadline_epoch": 610.0,
                    "started_epoch": 12.0,
                    "launcher_sha256": sha(Path(__file__).with_name("launch_pst.py")),
                })
            )
            for seed in ("20262905", "20262906"):
                for branch in ("whole", "split"):
                    branch_path = proof_root / seed / branch
                    branch_path.mkdir(parents=True)
                    (branch_path / "fit-contract.json").write_text(
                        json.dumps({"protocol_sha256": "protocol-sha"})
                    )
            protocol = {
                "phase_clocks": {
                    "proof": {"first_epoch": 10.0, "deadline_epoch": 610.0}
                },
            }
            def synthetic_state(branch, step, contract):
                return {"step": step}

            with patch("launch_pst.read_state", side_effect=synthetic_state):
                validate_proof_receipt(path, proof_sha, protocol, "protocol-sha", 16.0)
            receipt["strict_native_loads"] = 5
            path.write_text(json.dumps(receipt))
            with self.assertRaises(ValueError), patch(
                "launch_pst.read_state", side_effect=synthetic_state
            ):
                validate_proof_receipt(
                    path, hashlib.sha256(path.read_bytes()).hexdigest(),
                    protocol, "protocol-sha", 16.0,
                )


if __name__ == "__main__":
    unittest.main()
