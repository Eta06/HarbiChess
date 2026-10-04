"""Frozen CPU18-probe latency qualification. Run only after training/arenas quiesce."""

import argparse
import json
import time
import statistics
from pathlib import Path
import chess
import torch
from harbichess.backends.torch_network import load_weights, sha256
from harbichess.chess.actions import legal_action_indices
from harbichess.chess.encoding import BoardEncoder

PROBE_SHA = "1089fd0cca308c24bb040a840352d0ad18aa85a7bdcd17608133456bf08bed9f"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--initial", type=Path, required=True)
    p.add_argument("--final-20261205", type=Path, required=True)
    p.add_argument("--final-20261206", type=Path, required=True)
    p.add_argument("--probes", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--quiescent", action="store_true", required=True)
    a = p.parse_args()
    start = time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if sha256(a.probes) != PROBE_SHA:
        raise ValueError("probehash mismatch")
    paths = {
        "initial": a.initial,
        "20261205": a.final_20261205,
        "20261206": a.final_20261206,
    }
    models = {k: load_weights(v).eval() for k, v in paths.items()}
    if any(m.config != models["initial"].config for m in models.values()):
        raise ValueError("architecture changed")
    inputs = []
    for row in json.loads(a.probes.read_text())["positions"]:
        board = chess.Board(row["root_fen"])
        for move in row["moves"]:
            board.push_uci(move)
        if board.fen() != row["fen"]:
            raise ValueError("probehistory mismatch")
        inputs.append(
            (
                torch.tensor(BoardEncoder().encode_board(board).values).reshape(
                    1, 8, 8, 104
                ),
                torch.tensor([legal_action_indices(board)]),
            )
        )
    if len(inputs) != 18:
        raise ValueError("18probes required")
    samples = {k: [] for k in models}
    with torch.no_grad():
        for _ in range(10):
            for x, actions in inputs:
                for m in models.values():
                    m.masked_policy_value(x, actions)
        setup = time.perf_counter() - start
        for repeat in range(200):
            order = list(models)
            order = order[repeat % 3 :] + order[: repeat % 3]
            for x, actions in inputs:
                for k in order:
                    t = time.perf_counter_ns()
                    policy, wdl = models[k].masked_policy_value(x, actions)
                    dt = time.perf_counter_ns() - t
                    if (
                        not torch.isfinite(policy).all()
                        or not torch.isfinite(wdl).all()
                    ):
                        raise ValueError("nonfinite inference")
                    samples[k].append(dt)
    medians = {k: statistics.median(v) for k, v in samples.items()}
    ratios = {k: medians[k] / medians["initial"] for k in medians if k != "initial"}
    result = {
        "schema": 1,
        "quiescent_declared": True,
        "probe_sha256": PROBE_SHA,
        "weights_sha256": {k: sha256(v) for k, v in paths.items()},
        "median_nanoseconds": medians,
        "ratios_vs_initial": ratios,
        "gates": {k: r <= 1.10 for k, r in ratios.items()},
        "speed_pass": all(r <= 1.10 for r in ratios.values()),
        "setup_warmup_seconds_excluded": setup,
        "whole_seconds_after_import": time.perf_counter() - start,
        "cpu_threads": 1,
        "warmup_rounds": 10,
        "timed_rounds": 200,
        "script_sha256": sha256(Path(__file__)),
        "samples_nanoseconds": samples,
        "scope": "Batch1 masked FP32 CPU policy+WDL; encoding/setup excluded; rotated3arms; quiescence must be enforced externally; not game throughput.",
    }
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "samples_nanoseconds"}))


if __name__ == "__main__":
    main()
