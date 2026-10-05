"""Actual fresh processes through SC CLI with explicit synthetic data seam only."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
SOURCE = Path("/workspace/work/harbichess/cpu-additive-source-6fcc8b4")
MAIN = Path("/workspace/HarbiChess/experiments/ufuk")
WEIGHTS = Path(
    "/workspace/work/harbichess/a100/restoration/local-rehearsal-content/harbichess-inputs/initial-e8.safetensors"
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_unit8_whole4pause_freshresume8_exact_and_all6_strict_native_loads():
    from qualify import equal

    started = time.time()
    deadline = started + 120
    with tempfile.TemporaryDirectory(prefix="harbichess-sc-unit-", dir="/dev/shm") as directory:
        temp = Path(directory)
        journal = temp / "synthetic-journal-not-real-data.json"
        journal.write_text('{"scope":"synthetic common-data seam; no real actor data"}\n')
        actor = temp / "actor-config.json"
        config = {
            "seed": 20263005,
            "source_commit": "6fcc8b476d25495d1c9c413e55b2c7ba4794013e",
            "model_sha256": sha(WEIGHTS),
            "anchor_model_sha256": sha(WEIGHTS),
            "anchor_helper_sha256": "1" * 64,
            "anchor_target": "frozen-e8-wdl-probabilities-mover-perspective-v1",
            "producer_sha256": sha(MAIN / "cpu-fresh-selfplay-v2/journal_v2.py"),
            "torch_version": str(torch.__version__),
            "exclusion_book_sha256": "9" * 64,
            "excluded_training_position_keys": ["8/8/8/8/8/8/4k3/7K w - -"],
            "test_scope": "SYNTHETIC-DATA-SEAM-NOT-REAL-CLI-QUALIFICATION",
        }
        actor.write_text(json.dumps(config))
        protocol = json.loads((ROOT / "protocol-template.json").read_text())
        protocol.update(
            {
                "status": "SYNTHETIC-NATIVE-UNIT-NOT-CANDIDATE",
                "unit_fixture_sha256": sha(ROOT / "unit_cli_fixture.py"),
                "source_commit": config["source_commit"],
                "trainer_sha256": sha(ROOT / "train.py"),
                "feature_helper_sha256": sha(MAIN / "cpu-residual-value-v1/features.py"),
                "journal_helper_sha256": config["producer_sha256"],
                "anchor_helper_sha256": config["anchor_helper_sha256"],
                "initial_e8_sha256": sha(WEIGHTS),
                "seeds": [20263005],
                "seed_pairing": {
                    "20263005": {"journal_sha256": sha(journal), "actor_config_sha256": sha(actor)}
                },
                "deadline_epoch_by_seed": {"20263005": deadline},
                "sc_helper_sha256": sha(ROOT / "own_search_consistency.py"),
                "local_support_sha256": sha(ROOT / "features.py"),
                "protected_position_keys": config["excluded_training_position_keys"],
                "protected_position_keys_sha256": hashlib.sha256(
                    json.dumps(
                        config["excluded_training_position_keys"],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
                "protected_position_book_sha256": config["exclusion_book_sha256"],
                "learning_rate": 0.00002,
            }
        )
        protocol_path = temp / "protocol.json"
        protocol_path.write_text(json.dumps(protocol))
        command = [
            sys.executable,
            str(ROOT / "unit_cli_fixture.py"),
            "--weights",
            str(WEIGHTS),
            "--protocol",
            str(protocol_path),
            "--journal",
            str(journal),
            "--actor-config",
            str(actor),
            "--journal-helper",
            str(MAIN / "cpu-fresh-selfplay-v2/journal_v2.py"),
            "--feature-helper",
            str(MAIN / "cpu-residual-value-v1/features.py"),
            "--source-commit",
            config["source_commit"],
            "--seed",
            "20263005",
            "--deadline-epoch",
            str(deadline),
        ]
        env = {
            **os.environ,
            "PYTHONPATH": str(SOURCE / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
        }
        commands = []

        def run(extra):
            invocation = [*command, *extra]
            result = subprocess.run(
                invocation,
                cwd=SOURCE,
                env=env,
                capture_output=True,
                text=True,
                timeout=max(0.1, deadline - time.time()),
            )
            commands.append(
                {
                    "command": invocation,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            )
            if result.returncode:
                (ROOT / "unit-native-failure.json").write_text(json.dumps(commands, indent=2))
            assert result.returncode == 0, result.stderr

        for output, stop, resume in [("whole", 8, None), ("split", 4, None), ("split", 8, 4)]:
            extra = ["--output", str(temp / output), "--stop-at", str(stop)]
            if resume is not None:
                extra += ["--resume", str(temp / output / f"checkpoints/step-{resume:08d}")]
            run(extra)
        hashes = []
        serialization = []
        for step in (0, 4, 8):
            whole = temp / "whole" / f"checkpoints/step-{step:08d}"
            split = temp / "split" / f"checkpoints/step-{step:08d}"
            for name in ("training.pt", "checkpoint.json"):
                serialization.append(
                    {
                        "step": step,
                        "file": name,
                        "raw_bytes_equal": (whole / name).read_bytes()
                        == (split / name).read_bytes(),
                        "whole_sha256": sha(whole / name),
                        "split_sha256": sha(split / name),
                    }
                )
                hashes.append({"step": step, "file": name, "sha256": sha(whole / name)})
            a = torch.load(whole / "training.pt", weights_only=True)
            b = torch.load(split / "training.pt", weights_only=True)
            semantic_equal = equal(a, b)
            if not semantic_equal:
                (ROOT / "unit-native-state-difference.json").write_text(
                    json.dumps(
                        {
                            "step": step,
                            "fields_equal": {key: equal(a[key], b[key]) for key in a},
                            "commands": commands,
                        },
                        indent=2,
                    )
                )
            assert semantic_equal and a["accepted"] == step
            for output in ("whole", "split"):
                run(
                    [
                        "--output",
                        str(temp / output),
                        "--resume",
                        str(temp / output / f"checkpoints/step-{step:08d}"),
                        "--audit-only",
                    ]
                )
        receipt = {
            "schema": "SC-synthetic-common-data-unit-native-proof-v1",
            "status": "PASS-unit-not-real-data-CLI-qualification",
            "whole_steps": 8,
            "pause_steps": 4,
            "fresh_resume_steps": 8,
            "strict_loads": 6,
            "all3_boundary_complete_payload_storage_bits_exact": True,
            "raw_serialization_comparisons": serialization,
            "bindings": hashes,
            "commands": commands,
            "fixture_sha256": sha(ROOT / "unit_cli_fixture.py"),
            "trainer_sha256": sha(ROOT / "train.py"),
            "sc_helper_sha256": sha(ROOT / "own_search_consistency.py"),
            "qualifier_sha256": sha(ROOT / "qualify.py"),
            "started_epoch": started,
            "deadline_epoch": deadline,
            "finished_epoch": time.time(),
            "real_journals_or_selfplay_or_neural_actor_executed": False,
        }
        (ROOT / "unit-native-proof.json").write_text(json.dumps(receipt, indent=2) + "\n")
