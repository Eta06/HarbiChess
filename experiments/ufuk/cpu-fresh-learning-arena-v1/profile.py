"""Fixed 24-search CPU admission for the six freshly learned models."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import chess
import torch
from contracts import load_fit_manifest, verify_fit_inputs, verify_fixed_arena_protocol
from search import BudgetSearch
from value import NeuralValue

from harbichess.training.cgroup_budget import CgroupMemoryBudget
from harbichess.training.torch_search_acting_run import clean_source


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--first-epoch", type=float, required=True)
    args = ap.parse_args()
    p = json.loads(args.protocol.read_text())
    verify_fixed_arena_protocol(p)
    clean_source(p["source_commit"])
    if p.get("status") != "registered" or args.output.exists():
        raise ValueError("registered protocol and publish-once output required")
    fit_path, fit = load_fit_manifest(p)
    verify_fit_inputs(p, fit)
    if time.time() >= args.first_epoch + p["profile_seconds"]:
        raise TimeoutError("fixed profile clock expired")
    profile_end = args.first_epoch + p["profile_seconds"]
    if (
        p["profile_first_epoch"] != args.first_epoch
        or p["profile_deadline_epoch"] != profile_end
    ):
        raise ValueError("profile clock differs from registered original clocks")
    if profile_end > p["operator_hard_deadline_epoch"]:
        raise ValueError("profile exceeds hard deadline")
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        raise ValueError("CPU-only arena profile")
    budget = CgroupMemoryBudget(p["memory_max_bytes"])

    def guard():
        if time.time() >= profile_end:
            raise TimeoutError("original profile deadline exhausted")
        budget.check()

    book_path = Path(p["book_path"])
    if sha(book_path) != p["book_sha256"]:
        raise ValueError("opening book SHA differs")
    openings = json.loads(book_path.read_text())["splits"]["arena"][:4]
    rows = []
    for seed in p["match_seeds"]:
        for role in ("mc", "sc", "full"):
            rec = fit["fits"][str(seed)][role]
            if sha(rec["path"]) != rec["sha256"]:
                raise ValueError("profile candidate SHA differs")
            ev = NeuralValue(Path(rec["path"]))
            search = BudgetSearch(
                ev,
                nodes=p["search_nodes"],
                quiescence_plies=p["quiescence_plies"],
                max_depth=p["max_depth"],
                guard=guard,
            )
            for i, row in enumerate(openings):
                guard()
                board = chess.Board()
                for move in row["opening"]["moves"]:
                    board.push_uci(move)
                before = board.fen(), tuple(board.move_stack)
                tick = time.perf_counter()
                result = search.search(board)
                elapsed = time.perf_counter() - tick
                if not board.is_legal(result.move) or before != (
                    board.fen(),
                    tuple(board.move_stack),
                ):
                    raise ValueError("profile returned illegal move or mutated root")
                if (
                    result.nodes > p["search_nodes"]
                    or result.root_actions != board.legal_moves.count()
                ):
                    raise ValueError("profile search budget/root count differs")
                rows.append(
                    {
                        "seed": seed,
                        "role": role,
                        "opening": i,
                        "model_sha256": rec["sha256"],
                        "nodes": result.nodes,
                        "evaluations": result.evaluations,
                        "completed_depth": result.completed_depth,
                        "root_actions": result.root_actions,
                        "wall_seconds": elapsed,
                    }
                )
    if len(rows) != 24:
        raise ValueError("profile must execute exactly 24 fixed neural searches")
    guard()
    result = {
        "schema": "fresh-own-learning-arena-profile-v1",
        "status": "PASS-qualification-not-strength",
        "rows": rows,
        "protocol_sha256": sha(args.protocol),
        "fit_provenance_sha256": sha(fit_path),
        "helper_sha256": sha(Path(__file__)),
        "original_first_epoch": args.first_epoch,
        "original_deadline_epoch": profile_end,
        "finished_epoch": time.time(),
        "GPU_used": False,
        "strength_success_claimed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"status": result["status"], "searches": len(rows)}))


if __name__ == "__main__":
    main()
