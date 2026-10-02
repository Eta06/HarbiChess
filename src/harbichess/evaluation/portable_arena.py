"""Color/opening-paired CPU/CUDA diagnostics against neural/random/Stockfish baselines."""

from __future__ import annotations

import argparse
import math
import random
import subprocess
import time
from pathlib import Path

import chess
import chess.engine
import torch

from harbichess.backends.torch_backend import TorchPolicyValueBackend
from harbichess.backends.torch_network import load_weights, sha256
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove, Side
from harbichess.search.batching import SharedBatchEvaluator
from harbichess.search.evaluator import NeuralPositionEvaluator
from harbichess.search.full_gumbel import FullGumbelConfig, FullGumbelMCTS

OPENINGS = (
    ("e2e4", "e7e5", "g1f3", "b8c6"),
    ("e2e4", "c7c5", "g1f3", "d7d6"),
    ("d2d4", "d7d5", "c2c4", "e7e6"),
    ("c2c4", "e7e5", "b1c3", "g8f6"),
    ("g1f3", "d7d5", "g2g3", "g8f6"),
    ("d2d4", "g8f6", "c2c4", "g7g6"),
    ("e2e4", "e7e6", "d2d4", "d7d5"),
    ("e2e4", "c7c6", "d2d4", "d7d5"),
)


def paired_summary(games: list[dict], seed: int) -> dict:
    pairs = [(games[i]["score"] + games[i + 1]["score"]) / 2 for i in range(0, len(games), 2)]
    score = sum(pairs) / len(pairs)
    rng = random.Random(seed)
    bootstrap = sorted(sum(rng.choices(pairs, k=len(pairs))) / len(pairs) for _ in range(10000))
    radius = math.sqrt(math.log(40) / (2 * len(pairs)))
    return {
        "wins": sum(g["score"] == 1 for g in games),
        "draws": sum(g["score"] == 0.5 for g in games),
        "losses": sum(g["score"] == 0 for g in games),
        "score": score,
        "capped_games": sum(g["termination"] == "max_plies" for g in games),
        "pair_scores": pairs,
        "bootstrap_pair_95": [bootstrap[249], bootstrap[9749]],
        "hoeffding_pair_95": [max(0, score - radius), min(1, score + radius)],
        "uncertainty_scope": "opening families assumed independent; fixed suite, no general Elo",
        "capped_scoring": "unknown games count 0.5 for diagnostic only; caps listed separately",
    }


def arena(
    candidate: Path,
    *,
    opponent: Path | str,
    stockfish: Path | None = None,
    nodes: int = 32,
    simulations: int = 16,
    max_plies: int = 192,
    seed: int = 20261002,
    threads: int = 1,
    wall_seconds: float = 900,
    opening_pairs: int = 8,
) -> dict:
    if (
        not 1 <= opening_pairs <= len(OPENINGS)
        or min(nodes, simulations, max_plies, threads) <= 0
        or wall_seconds <= 0
    ):
        raise ValueError("positive budgets and 1..8 opening pairs required")
    torch.set_num_threads(threads)
    rules = PythonChessRules()
    bridges = []
    engine = None
    games = []
    engine_id = None
    started = time.perf_counter()
    deadline = started + wall_seconds
    config = FullGumbelConfig(
        simulations=simulations, max_considered_actions=min(16, simulations), gumbel_scale=0.0
    )

    def search(weights):
        bridge = SharedBatchEvaluator(
            TorchPolicyValueBackend(load_weights(weights)), max_batch_size=1, max_wait_seconds=0
        )
        bridges.append(bridge)
        return FullGumbelMCTS(
            NeuralPositionEvaluator(bridge, rules=rules), rules=rules, config=config
        )

    candidate_search = search(candidate)
    other_search = search(opponent) if isinstance(opponent, Path) else None
    try:
        if opponent == "stockfish":
            if stockfish is None:
                raise ValueError("Stockfish executable required")
            engine = chess.engine.SimpleEngine.popen_uci(str(stockfish), timeout=15)
            engine.configure({"Threads": 1, "Hash": 16})
            engine_id = engine.id
        for pair, opening in enumerate(OPENINGS[:opening_pairs]):
            for color in (Side.WHITE, Side.BLACK):
                rng = random.Random(f"{seed}:{pair}:{color}")
                state = rules.initial_state()
                for move in opening:
                    state = rules.apply(state, ChessMove(move))
                game_started = time.perf_counter()
                while rules.outcome(state, claim_draw=True) is None and state.ply < max_plies:
                    if time.perf_counter() >= deadline:
                        raise TimeoutError("arena wall budget exhausted")
                    if rules.view(state).side_to_move == color:
                        move = candidate_search.search(state, rng=rng).selected_action
                    elif other_search:
                        move = other_search.search(state, rng=rng).selected_action
                    elif engine:
                        result = engine.play(
                            rules.board(state), chess.engine.Limit(nodes=nodes), game=(pair, color)
                        )
                        move = ChessMove(result.move.uci())
                    else:
                        move = rng.choice(sorted(rules.legal_moves(state), key=lambda m: m.uci))
                    state = rules.apply(state, move)
                outcome = rules.outcome(state, claim_draw=True)
                games.append(
                    {
                        "opening_pair": pair,
                        "opening": opening,
                        "candidate_color": color,
                        "plies": state.ply,
                        "termination": outcome.termination if outcome else "max_plies",
                        "score": (outcome.value_for(color) + 1) / 2 if outcome else 0.5,
                        "moves": [m.uci for m in state.moves],
                        "wall_seconds": time.perf_counter() - game_started,
                    }
                )
    finally:
        if engine:
            engine.quit()
        statistics = [bridge.statistics for bridge in bridges]
        for bridge in bridges:
            bridge.close()
    elapsed = time.perf_counter() - started
    return {
        "candidate_sha256": sha256(candidate),
        "opponent": str(opponent),
        "opponent_sha256": sha256(opponent) if isinstance(opponent, Path) else None,
        "engine_id": engine_id,
        "stockfish_sha256": sha256(stockfish) if engine_id else None,
        "stockfish_nodes": nodes if engine_id else None,
        "stockfish_threads": 1 if engine_id else None,
        "stockfish_hash_mib": 16 if engine_id else None,
        "neural_simulations_per_move": simulations,
        "neural_threads": threads,
        "seed": seed,
        "max_plies": max_plies,
        "wall_seconds": elapsed,
        "evaluated_positions": sum(s.positions for s in statistics),
        "positions_per_wall_second": sum(s.positions for s in statistics) / elapsed,
        "summary": paired_summary(games, seed),
        "games": games,
        "promotion_ready": False,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }


def main() -> None:
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("opponent", help="random, stockfish or portable weight file")
    parser.add_argument("--stockfish", type=Path)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--nodes", type=int, default=32)
    parser.add_argument("--max-plies", type=int, default=192)
    parser.add_argument("--wall-seconds", type=float, default=900)
    parser.add_argument("--opening-pairs", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    opponent = args.opponent if args.opponent in ("random", "stockfish") else Path(args.opponent)
    result = arena(
        args.candidate,
        opponent=opponent,
        stockfish=args.stockfish,
        simulations=args.simulations,
        nodes=args.nodes,
        max_plies=args.max_plies,
        wall_seconds=args.wall_seconds,
        opening_pairs=args.opening_pairs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
