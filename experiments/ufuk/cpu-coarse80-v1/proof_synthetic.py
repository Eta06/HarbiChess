"""Synthetic full-state proof only. Does not load project data or call an evaluator."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from learner80 import OPTIMIZER, Learner, decode_native, encode_native

HERE = Path(__file__).resolve().parent


def fixture():
    groups = {}
    for game in range(8):
        rows = []
        for row in range(4):
            x = [0.0] * 80
            x[(game * 7 + row * 3) % 80] = 1.0
            x[(game * 7 + row * 3 + 1) % 80] = -1.0
            rows.append((x, (game - row) * 0.001, max(-1.0, min(1.0, (game - 3) / 4))))
        groups[f"synthetic-game-{game}"] = rows
    contract = {
        "schema": "ownq-coarse80-contract-v1",
        "seed": 81023,
        "optimizer": OPTIMIZER,
        "labels_sha256": "0" * 64,
        "prior_sha256": "1" * 64,
        "search_producer_sha256": "4" * 64,
        "source_commit": "2" * 40,
        "closure_sha256": "3" * 64,
    }
    return groups, contract


def run():
    groups, contract = fixture()
    whole = Learner(contract["seed"], contract)
    whole.advance(groups, 8)
    expected = encode_native(whole.native())
    with tempfile.TemporaryDirectory(prefix="coarse80-synthetic-proof-") as temp:
        root = Path(temp)
        groups_path = root / "groups.json"
        contract_path = root / "contract.json"
        native_path = root / "native.gz"
        output_path = root / "resumed.gz"
        groups_path.write_text(json.dumps(groups, separators=(",", ":")))
        contract_path.write_text(json.dumps(contract, sort_keys=True, separators=(",", ":")))
        split = Learner(contract["seed"], contract)
        split.advance(groups, 4)
        native_path.write_bytes(encode_native(split.native()))
        subprocess.run(
            [
                sys.executable,
                str(HERE / "resume_worker.py"),
                "--groups",
                str(groups_path),
                "--contract",
                str(contract_path),
                "--native",
                str(native_path),
                "--out",
                str(output_path),
                "--target",
                "8",
            ],
            check=True,
            cwd=HERE,
            timeout=30,
        )
        resumed = output_path.read_bytes()
        if resumed != expected:
            raise AssertionError("fresh-process 4→8 bytes differ from whole 8")
        state = decode_native(resumed, contract)
        for _ in range(6):
            clone = decode_native(resumed, contract)
            if encode_native(clone) != resumed or clone != state:
                raise AssertionError("strict native load changed bytes/state")
    return {
        "schema": "coarse80-synthetic-native-proof-v1",
        "synthetic_only": True,
        "data_sha256": "synthetic-fixture-not-production-data",
        "whole_updates": 8,
        "pause_updates": 4,
        "fresh_process_resume_updates": 4,
        "six_strict_fresh_loads": True,
        "whole_resume_native_byte_equal": True,
        "candidate_updated": any(value != 0.0 for value in state["theta"]),
        "production_labels_loaded": False,
        "inference_or_search_run": False,
    }


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
