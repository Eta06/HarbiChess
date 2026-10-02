"""Warmed end-to-end CPU thread/batch benchmark with real policy/value inference."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch

from harbichess.backends.torch_backend import TorchPolicyValueBackend
from harbichess.backends.torch_network import load_weights, sha256
from harbichess.chess.actions import legal_action_indices
from harbichess.chess.encoding import BoardEncoder
from harbichess.chess.rules import PythonChessRules


def benchmark(weights: Path, *, legacy_mlx: bool = False, repeats: int = 30) -> dict:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    network = load_weights(weights, legacy_mlx=legacy_mlx)
    rules = PythonChessRules()
    state = rules.initial_state()
    encoded = BoardEncoder(rules).encode(state)
    actions = legal_action_indices(rules.board(state))
    rows = []
    for threads in (1, 2, 4):
        torch.set_num_threads(threads)
        backend = TorchPolicyValueBackend(network)
        for batch in (1, 8, 32):
            positions, legal = [encoded] * batch, [actions] * batch
            for _ in range(5):
                backend.evaluate_masked(positions, legal)
            elapsed = []
            for _ in range(repeats):
                started = time.perf_counter()
                backend.evaluate_masked(positions, legal)
                elapsed.append(time.perf_counter() - started)
            rows.append(
                {
                    "threads": threads,
                    "batch": batch,
                    "repeats": repeats,
                    "positions_per_second": repeats * batch / sum(elapsed),
                    "mean_batch_ms": statistics.mean(elapsed) * 1000,
                    "p95_batch_ms": sorted(elapsed)[int((len(elapsed) - 1) * 0.95)] * 1000,
                }
            )
    return {
        "weights_sha256": sha256(weights),
        "torch": torch.__version__,
        "device": "cpu",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "encoder": "shared schema 1; encoding outside timed region",
        "timing": "backend input construction, masked forward and host outputs included",
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--legacy-mlx", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = benchmark(args.weights, legacy_mlx=args.legacy_mlx)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
