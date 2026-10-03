"""Frozen all-legal-reference diagnostic of equal-simulation root breadth."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import resource
import subprocess
import time
from pathlib import Path

import torch

from harbichess.backends.torch_backend import TorchPolicyValueBackend
from harbichess.backends.torch_network import load_weights, sha256
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove, ChessState
from harbichess.search.batching import SharedBatchEvaluator
from harbichess.search.evaluator import NeuralPositionEvaluator
from harbichess.search.full_gumbel import FullGumbelConfig, FullGumbelMCTS


class CountingEvaluator:
    def __init__(self, evaluator):
        self.evaluator, self.calls = evaluator, 0

    def evaluate(self, state):
        self.calls += 1
        return self.evaluator.evaluate(state)


def run(reference: Path, weights: Path, output: Path) -> dict:
    started = time.perf_counter()
    torch.set_num_threads(1)
    if sha256(weights) != "18f2aae5a4dca317229b87af17ee393a4e786db3369e54aaf32c926a811cb5ae":
        raise ValueError("frozen model mismatch")
    audit = json.loads((reference / "integrity-audit.json").read_text())
    for filename in ("positions.jsonl", "panel.json"):
        if sha256(reference / filename) != audit["hashes"][filename]:
            raise ValueError("frozen reference mismatch")
    positions = [json.loads(s) for s in (reference / "positions.jsonl").read_text().splitlines()]
    if len(positions) != 128:
        raise ValueError("frozen panel count mismatch")
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "collector_sha256": sha256(Path(__file__)),
        "model_sha256": sha256(weights),
        "reference_sha256": sha256(reference / "positions.jsonl"),
        "seed": 20261013,
        "simulations": 16,
        "widths": [16, 4],
        "wall_budget_seconds": 300,
        "new_reference_calls": 0,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rules = PythonChessRules()
    bridge = SharedBatchEvaluator(
        TorchPolicyValueBackend(load_weights(weights)), max_batch_size=1, max_wait_seconds=0
    )
    evaluator = CountingEvaluator(NeuralPositionEvaluator(bridge, rules=rules))
    rows, status, reason = [], "completed", "all128 positions measured"
    try:
        with (output / "positions.jsonl").open("x") as stream:
            for index, row in enumerate(positions):
                if time.perf_counter() - started >= 300:
                    raise TimeoutError("registered wall cap")
                state = ChessState(row["root_fen"], tuple(ChessMove(m) for m in row["moves"]))
                legal = {m.uci for m in rules.legal_moves(state)}
                if set(row["candidate_scores"]) != legal:
                    raise ValueError("reference legal coverage mismatch")
                arms = {}
                for width in (16, 4) if index % 2 == 0 else (4, 16):
                    search = FullGumbelMCTS(
                        evaluator,
                        rules=rules,
                        config=FullGumbelConfig(
                            simulations=16, max_considered_actions=width, gumbel_scale=0
                        ),
                    )
                    before, wall = evaluator.calls, time.perf_counter()
                    result = search.search(state, rng=random.Random(f"20261013:{index}"))
                    selected = result.selected_action.uci
                    if selected not in legal or sum(m.visits for m in result.moves) != 16:
                        raise ValueError("search legal/visit mismatch")
                    arms[str(width)] = {
                        "selected": selected,
                        "expected_score": row["candidate_scores"][selected]["expected_score"],
                        "soft_target_expected_score": sum(
                            p * row["candidate_scores"][m.uci]["expected_score"]
                            for m, p in result.action_weights
                        ),
                        "visits": {m.move.uci: m.visits for m in result.moves},
                        "policy": {m.uci: p for m, p in result.action_weights},
                        "priors": {m.move.uci: m.prior for m in result.moves},
                        "nn_calls": evaluator.calls - before,
                        "wall_seconds": time.perf_counter() - wall,
                    }
                item = {
                    "index": index,
                    "family": row["family"],
                    "root_fen": row["root_fen"],
                    "moves": row["moves"],
                    "reference_row_sha256": hashlib.sha256(
                        json.dumps(row, sort_keys=True).encode()
                    ).hexdigest(),
                    "arms": arms,
                    "effect": arms["4"]["expected_score"] - arms["16"]["expected_score"],
                }
                best = max(s["expected_score"] for s in row["candidate_scores"].values())
                ranked = sorted(arms["16"]["priors"], key=lambda m: (-arms["16"]["priors"][m], m))
                item["prior_near_best_coverage"] = {
                    str(k): any(
                        row["candidate_scores"][m]["expected_score"] >= best - 0.05
                        for m in ranked[:k]
                    )
                    for k in (4, 16)
                }
                rows.append(item)
                stream.write(json.dumps(item) + "\n")
                stream.flush()
    except Exception as error:
        status, reason = "incomplete", f"{type(error).__name__}: {error}"
    finally:
        bridge.close()
    effects = [sum(r["effect"] for r in rows if r["family"] == f) / 8 for f in range(16)]
    rng = random.Random(20261013)
    samples = sorted(sum(rng.choices(effects, k=16)) / 16 for _ in range(10000))
    mean, radius = sum(effects) / 16, math.sqrt(2 * math.log(40) / 16)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result = {
        **metadata,
        "status": status,
        "reason": reason,
        "positions_completed": len(rows),
        "mean_selected_expected_score_effect": mean if len(rows) == 128 else None,
        "conditional_bootstrap_95": [samples[249], samples[9749]] if len(rows) == 128 else None,
        "hoeffding_95": [max(-1, mean - radius), min(1, mean + radius)]
        if len(rows) == 128
        else None,
        "mechanism_gate_passed": len(rows) == 128 and mean >= 0.01 and samples[249] > 0,
        "family_effects": effects,
        "nn_calls": evaluator.calls,
        "wall_seconds": time.perf_counter() - started,
        "parent_cpu_seconds": usage.ru_utime + usage.ru_stime,
        "peak_rss_kib": usage.ru_maxrss,
        "scope": (
            "Reused development panel and finite references; "
            "no played strength or new training evidence."
        ),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "weights", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = run(args.reference, args.weights, args.output)
    print(json.dumps(result))
    if result["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
