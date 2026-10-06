"""Fixed prospective CPU legal-root timing admission; not a strength experiment."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import chess
import profile_support
import torch
from profile_support import publish_result
from search import BudgetSearch
from value import MixedValue

from harbichess.training.cgroup_budget import CgroupMemoryBudget
from harbichess.training.torch_search_acting_run import clean_source


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--first-epoch", type=float, required=True)
    args = parser.parse_args()
    p = json.loads(args.protocol.read_text())
    clean_source(p["source_commit"])
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        raise ValueError("CPU-only")
    end = args.first_epoch + 600
    assert args.first_epoch == p["profile_first_epoch"]
    assert end == p["profile_deadline_epoch"] <= p["operator_hard_deadline_epoch"]
    assert args.first_epoch <= time.time() < end
    budget = CgroupMemoryBudget(p["memory_max_bytes"])

    def guard():
        if time.time() >= end:
            raise TimeoutError("original profiling clock exhausted")
        budget.check()

    from native_admission import admit

    admissions = [admit(p, seed) for seed in p["match_seeds"]]
    book = Path(p["book_path"])
    assert sha(book) == p["book_sha256"]
    openings = json.loads(book.read_text())["splits"]["arena"][:4]
    rows = []
    ratios = []
    try:
        for seed in p["match_seeds"]:
            for role in ["learned", "prior", "e8"]:
                info = p["models"][str(seed)][role]
                path = Path(info["path"])
                assert sha(path) == info["sha256"]
                evaluator = MixedValue(path, p)
                engine = BudgetSearch(
                    evaluator,
                    nodes=p["search_nodes"],
                    quiescence_plies=p["quiescence_plies"],
                    max_depth=p["max_depth"],
                    guard=guard,
                )
                for i, row in enumerate(openings):
                    board = chess.Board()
                    for uci in row["opening"]["moves"]:
                        board.push_uci(uci)
                    before = board.fen(), tuple(board.move_stack)
                    first = time.perf_counter()
                    r = engine.search(board)
                    elapsed = time.perf_counter() - first
                    assert board.is_legal(r.move) and r.root_actions == board.legal_moves.count()
                    assert r.nodes <= p["search_nodes"] and before == (
                        board.fen(),
                        tuple(board.move_stack),
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "role": role,
                            "opening": i,
                            "wall_seconds": elapsed,
                            "nodes": r.nodes,
                            "static_evaluations": r.evaluations,
                            "completed_depth": r.completed_depth,
                            "root_actions": r.root_actions,
                            "model_sha256": info["sha256"],
                        }
                    )
        guard()
        ratios = []
        for seed in p["match_seeds"]:
            elapsed = {
                role: sum(
                    r["wall_seconds"] for r in rows if r["seed"] == seed and r["role"] == role
                )
                for role in ["learned", "prior", "e8"]
            }
            ratios.append(
                {
                    "seed": seed,
                    "learned_vs_original_fastprior": elapsed["learned"] / elapsed["prior"],
                    "learned_vs_original_E8": elapsed["learned"] / elapsed["e8"],
                }
            )
        if any(
            r[k] > 1.10
            for r in ratios
            for k in ["learned_vs_original_fastprior", "learned_vs_original_E8"]
        ):
            raise ValueError("actual trained24 latency exceeds fixed1.10 control")
        status = "PASS-selective-Q-trained24-profile-not-strength"
    except Exception as exc:
        status = "failed-preserved"
        rows.append({"error": repr(exc)})
    result = {
        "profile_support_sha256": sha(Path(profile_support.__file__)),
        "status": status,
        "rows": rows,
        "native_admissions": admissions,
        "latency_ratios": ratios,
        "ratio_statistic": "sum of four fixed search calls per seed and role",
        "protocol_sha256": sha(args.protocol),
        "helper_sha256": sha(Path(__file__)),
        "original_first_epoch": args.first_epoch,
        "original_deadline_epoch": end,
        "finished_epoch": time.time(),
        "GPU_used": False,
        "strength_success_claimed": False,
        "scope": "24 fixed CPU moves, not games or strength",
    }
    result = publish_result(args.output, result, end)
    status = result["status"]
    rows = result["rows"]
    if status.startswith("failed"):
        raise RuntimeError(rows[-1])
    print(json.dumps({"status": status, "moves": len(rows)}))


if __name__ == "__main__":
    main()
