"""Audit own-search soft targets against all legal budgeted engine root choices."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import resource
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import chess
import torch

from harbichess.backends.torch_backend import TorchPolicyValueBackend
from harbichess.backends.torch_network import load_weights, sha256
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove, ChessState
from harbichess.evaluation.teacher_probe import StockfishReference
from harbichess.search.batching import SharedBatchEvaluator
from harbichess.search.evaluator import NeuralPositionEvaluator
from harbichess.search.full_gumbel import FullGumbelConfig, FullGumbelMCTS


def freeze(dataset: Path) -> list[dict]:
    manifest = json.loads((dataset / "dataset.json").read_text())
    groups = defaultdict(lambda: defaultdict(list))
    for path in sorted(dataset.glob("validation*.json.gz")):
        if sha256(path) != manifest["files"][path.name]:
            raise ValueError("native data checksum mismatch")
        data = json.loads(gzip.decompress(path.read_bytes()))
        job = data["job"]
        groups[job["family"]][job["actor"]].append((job["game"], path.name, data))
    rng, panel = random.Random(20261010), []
    for family in sorted(groups):
        for actor in ("engine-engine", "neural-engine"):
            games = sorted(groups[family][actor], key=lambda x: x[0])[:4]
            if len(games) != 4:
                raise ValueError("not enough distinct source games")
            for index, (game, name, data) in enumerate(games):
                turn = "w" if index % 2 == 0 else "b"
                eligible = [r for r in data["rows"] if r["fen"].split()[1] == turn]
                if not eligible:
                    raise ValueError("source game lacks preregistered side")
                row = rng.choice(eligible)
                board = chess.Board(row["root_fen"])
                for uci in row["moves"]:
                    board.push_uci(uci)
                if board.fen() != row["fen"] or board.outcome(claim_draw=True):
                    raise ValueError("invalid or terminal frozen history")
                panel.append(
                    {
                        "family": family,
                        "game": game,
                        "actor": actor,
                        "neural_actor_color": data["job"]["neural_color"],
                        "source_file": name,
                        "source_file_sha256": manifest["files"][name],
                        "root_fen": row["root_fen"],
                        "moves": row["moves"],
                        "fen": row["fen"],
                    }
                )
    if len(panel) != 128 or len(groups) != 16:
        raise ValueError("frozen panel does not match registered count")
    return panel


def run(dataset: Path, weights: Path, stockfish: Path, output: Path, wall_seconds: float) -> dict:
    torch.set_num_threads(1)
    started = time.perf_counter()
    panel = freeze(dataset)
    output.mkdir(parents=True, exist_ok=False)
    (output / "panel.json").write_text(json.dumps(panel, indent=2) + "\n")
    metadata = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "collector_sha256": sha256(Path(__file__)),
        "panel_sha256": sha256(output / "panel.json"),
        "dataset_sha256": sha256(dataset / "dataset.json"),
        "weights_sha256": sha256(weights),
        "stockfish_sha256": sha256(stockfish),
        "seed": 20261010,
        "search": {"simulations": 64, "max_actions": 16, "gumbel": 1, "value_scale": 0.1},
        "wall_budget_seconds": wall_seconds,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rules = PythonChessRules()
    bridge = SharedBatchEvaluator(
        TorchPolicyValueBackend(load_weights(weights)), max_batch_size=1, max_wait_seconds=0
    )
    search = FullGumbelMCTS(
        NeuralPositionEvaluator(bridge, rules=rules),
        rules=rules,
        config=FullGumbelConfig(simulations=64, gumbel_scale=1),
    )
    rows, status, reason = [], "completed", "all frozen positions measured"
    with (
        (output / "queries.jsonl").open("x") as queries,
        (output / "positions.jsonl").open("x") as positions,
    ):
        reference = StockfishReference(stockfish, rules, started + wall_seconds, queries)
        try:
            for index, item in enumerate(panel):
                state = ChessState(item["root_fen"], tuple(ChessMove(m) for m in item["moves"]))
                result = search.search(state, rng=random.Random(f"20261010:{index}"))
                raw = {m.move.uci: m.prior for m in result.moves}
                target = {m.uci: p for m, p in result.action_weights}
                legal = {m.uci for m in rules.legal_moves(state)}
                if set(raw) != set(target) or set(raw) != legal:
                    raise ValueError("raw/target legal action mismatch")
                if any(abs(sum(p.values()) - 1) > 1e-6 for p in (raw, target)):
                    raise ValueError("policy probabilities not normalized")
                scores = {
                    uci: reference.analyse(state, 32768, ChessMove(uci)) for uci in sorted(legal)
                }
                expected = {
                    arm: sum(p[m] * scores[m]["expected_score"] for m in p)
                    for arm, p in (("raw", raw), ("target", target))
                }
                cp = {
                    arm: sum(p[m] * math.tanh(scores[m]["cp_or_mate_score"] / 400) for m in p)
                    for arm, p in (("raw", raw), ("target", target))
                }
                best = max(s["expected_score"] for s in scores.values())
                row = {
                    **item,
                    "raw": raw,
                    "target": target,
                    "search_selected": result.selected_action.uci,
                    "visits": {m.move.uci: m.visits for m in result.moves},
                    "child_mean_values": {m.move.uci: m.mean_value for m in result.moves},
                    "candidate_scores": scores,
                    "expected_score": expected,
                    "effect": expected["target"] - expected["raw"],
                    "cp_tanh_diagnostic": cp,
                    "entropy": {
                        arm: -sum(v * math.log(max(v, 1e-300)) for v in p.values())
                        for arm, p in (("raw", raw), ("target", target))
                    },
                    "kl_target_raw": sum(
                        target[m] * math.log(max(target[m], 1e-300) / max(raw[m], 1e-300))
                        for m in legal
                    ),
                    "near_best_mass": {
                        arm: sum(
                            v for m, v in p.items() if scores[m]["expected_score"] >= best - 0.05
                        )
                        for arm, p in (("raw", raw), ("target", target))
                    },
                }
                if not math.isfinite(row["effect"]):
                    raise ValueError("nonfinite target effect")
                rows.append(row)
                positions.write(json.dumps(row) + "\n")
                positions.flush()
                queries.flush()
                if len(rows) % 16 == 0:
                    print(
                        json.dumps(
                            {"positions": len(rows), "elapsed": time.perf_counter() - started}
                        ),
                        flush=True,
                    )
        except Exception as error:
            status, reason = "incomplete", f"{type(error).__name__}: {error}"
        finally:
            reference.close()
            bridge.close()
    effects = [sum(r["effect"] for r in rows if r["family"] == f) / 8 for f in range(16)]
    mean = sum(effects) / 16
    rng = random.Random(20261010)
    boot = sorted(sum(rng.choices(effects, k=16)) / 16 for _ in range(10000))
    radius = math.sqrt(2 * math.log(40) / 16)
    report = {
        **metadata,
        "status": status,
        "reason": reason,
        "positions_completed": len(rows),
        "family_effects": effects,
        "mean_expected_score_effect": mean if len(rows) == 128 else None,
        "conditional_bootstrap_95": [boot[249], boot[9749]] if len(rows) == 128 else None,
        "hoeffding_95": [max(-1, mean - radius), min(1, mean + radius)]
        if len(rows) == 128
        else None,
        "mechanism_gate_passed": len(rows) == 128 and mean >= 0.01 and boot[249] > 0,
        "reference_queries": reference.calls,
        "reference_actual_nodes": reference.nodes,
        "self_cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime
        + resource.getrusage(resource.RUSAGE_SELF).ru_stime,
        "self_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "wall_seconds": time.perf_counter() - started,
        "role_stm_counts": dict(Counter(r["actor"] + "/" + r["fen"].split()[1] for r in rows)),
        "scope": (
            "Fixed16 development families/native finite-budget all-legal references; "
            "soft-target diagnostic, not game strength or promotion."
        ),
    }
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "weights", "stockfish", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--wall-seconds", type=float, default=1800)
    args = parser.parse_args()
    if args.wall_seconds <= 0:
        parser.error("positive wall budget required")
    result = run(args.dataset, args.weights, args.stockfish, args.output, args.wall_seconds)
    print(json.dumps(result))
    if result["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
