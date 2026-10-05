"""Fixed paired CPU neural-value alpha-beta tournaments, independent of training."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import chess
import chess.engine
import torch
from contracts import (
    load_fit_manifest,
    model_record,
    verify_fit_inputs,
    verify_fixed_arena_protocol,
)
from search import BudgetSearch
from value import NeuralValue

from harbichess.evaluation.portable_arena import paired_summary
from harbichess.training.cgroup_budget import CgroupMemoryBudget
from harbichess.training.torch_search_acting_run import clean_source


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "candidate", "book", "stockfish", "output", "progress"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args()
    p = json.loads(args.protocol.read_text())
    verify_fixed_arena_protocol(p)
    clean_source(p["source_commit"])
    _, fit_manifest = load_fit_manifest(p)
    verify_fit_inputs(p, fit_manifest)
    p["models"] = {
        str(seed): {
            role: model_record(p, fit_manifest, seed, role)
            for role in ("e8", "mc", "sc", "full")
        }
        for seed in p["match_seeds"]
    }
    assert args.seed in p["match_seeds"] and time.time() < args.deadline_epoch
    assert (
        sha(args.book) == p["book_sha256"]
        and sha(args.stockfish) == p["stockfish_sha256"]
    )
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        raise ValueError("CPU-only prospective control")
    budget = CgroupMemoryBudget(p["memory_max_bytes"])

    def guard():
        if time.time() >= args.deadline_epoch:
            raise TimeoutError("original tournament clock exhausted")
        budget.check()

    initial = p["models"][str(args.seed)]
    admitted = {r["sha256"] for r in initial.values()}
    assert sha(args.candidate) in admitted
    other = None if args.opponent == "SF512" else Path(args.opponent)
    if other is not None:
        assert sha(other) in admitted
    book = json.loads(args.book.read_text())["splits"]["arena"]
    candidate = BudgetSearch(
        NeuralValue(args.candidate),
        nodes=p["search_nodes"],
        quiescence_plies=p["quiescence_plies"],
        max_depth=p["max_depth"],
        guard=guard,
    )
    opponent = (
        None
        if other is None
        else BudgetSearch(
            NeuralValue(other),
            nodes=p["search_nodes"],
            quiescence_plies=p["quiescence_plies"],
            max_depth=p["max_depth"],
            guard=guard,
        )
    )
    engine = None
    games = []
    first = time.time()
    args.output.parent.mkdir(exist_ok=True)
    with args.progress.open("x") as progress:

        def event(x):
            progress.write(json.dumps(x) + "\n")
            progress.flush()

        try:
            if other is None:
                engine = chess.engine.SimpleEngine.popen_uci(
                    str(args.stockfish), timeout=15
                )
                engine.configure({"Threads": 1, "Hash": 16})
            for pair, row in enumerate(book[: p["opening_pairs"]]):
                opening = row["opening"]["moves"]
                for color in (chess.WHITE, chess.BLACK):
                    board = chess.Board()
                    for uci in opening:
                        board.push_uci(uci)
                    assert board.is_valid()
                    start = time.perf_counter()
                    times = []
                    sf = []
                    searches = []
                    event(
                        {
                            "type": "game_start",
                            "pair": pair,
                            "color": "white" if color else "black",
                        }
                    )
                    while (
                        board.outcome(claim_draw=True) is None
                        and board.ply() < p["max_plies"]
                    ):
                        guard()
                        move_start = time.perf_counter()
                        candidate_turn = board.turn == color
                        if candidate_turn or opponent is not None:
                            result = (candidate if candidate_turn else opponent).search(
                                board
                            )
                            if result.move is None or not board.is_legal(result.move):
                                raise ValueError(
                                    "search did not return a legal nonterminal move"
                                )
                            move = result.move
                            searches.append(
                                {
                                    "ply": board.ply() + 1,
                                    "candidate": candidate_turn,
                                    "nodes": result.nodes,
                                    "evaluations": result.evaluations,
                                    "completed_depth": result.completed_depth,
                                    "root_actions": result.root_actions,
                                    "legal_root_actions": board.legal_moves.count(),
                                    "value": result.value,
                                    "selected_move": move.uci(),
                                    "wall_seconds": time.perf_counter() - move_start,
                                }
                            )
                        else:
                            result = engine.play(
                                board,
                                chess.engine.Limit(nodes=p["stockfish_nodes"]),
                                game=(pair, color),
                                info=chess.engine.INFO_BASIC,
                            )
                            move = result.move
                            assert isinstance(result.info.get("nodes"), int)
                            sf.append(
                                {
                                    "ply": board.ply() + 1,
                                    "nodes": result.info["nodes"],
                                    "wall_seconds": time.perf_counter() - move_start,
                                }
                            )
                        board.push(move)
                        times.append(time.perf_counter() - move_start)
                        event({"type": "move", "ply": board.ply(), "uci": move.uci()})
                    outcome = board.outcome(claim_draw=True)
                    score = (
                        0.5
                        if outcome is None or outcome.winner is None
                        else float(outcome.winner == color)
                    )
                    game = {
                        "opening_pair": pair,
                        "opening": opening,
                        "candidate_color": "white" if color else "black",
                        "plies": board.ply(),
                        "termination": "max_plies"
                        if outcome is None
                        else outcome.termination.name.lower(),
                        "score": score,
                        "moves": [m.uci() for m in board.move_stack],
                        "wall_seconds": time.perf_counter() - start,
                        "move_wall_seconds": times,
                        "stockfish_nodes_by_move": sf,
                        "search_by_move": searches,
                    }
                    games.append(game)
                    event({"type": "game_end", "game": game})
        finally:
            if engine is not None:
                engine.quit()
    guard()
    result = {
        "schema": "cpu-all-root-quiescent-alpha-beta-arena-v1",
        "source_commit": p["source_commit"],
        "protocol_sha256": sha(args.protocol),
        "helper_sha256": sha(Path(__file__)),
        "search_helper_sha256": sha(Path(__file__).with_name("search.py")),
        "value_helper_sha256": sha(Path(__file__).with_name("value.py")),
        "candidate_sha256": sha(args.candidate),
        "opponent_sha256": None if other is None else sha(other),
        "stockfish_sha256": sha(args.stockfish) if other is None else None,
        "search_nodes_per_move": p["search_nodes"],
        "quiescence_plies": p["quiescence_plies"],
        "stockfish_nodes": p["stockfish_nodes"] if other is None else None,
        "stockfish_threads": 1 if other is None else None,
        "stockfish_hash_mib": 16 if other is None else None,
        "opening_source_sha256": sha(args.book),
        "seed": args.seed,
        "max_plies": p["max_plies"],
        "summary": paired_summary(games, args.seed),
        "games": games,
        "GPU_used": False,
        "original_deadline_epoch": args.deadline_epoch,
        "started_epoch": first,
        "finished_epoch": time.time(),
        "promotion_ready": False,
        "scope": "known development roots, not formal strength confirmation",
        "budget_comparability": (
            "alpha-beta recursive nodes include terminal and quiescence; "
            "Stockfish actualnodes separately. Not equal compute or "
            "direct node-throughput comparison."
        ),
    }
    guard()
    with args.output.open("x") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    print(
        json.dumps(
            {"status": "completed-games-not-strength", "summary": result["summary"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
