"""Bounded executor for a preselected own-search root list (not invoked here)."""

from __future__ import annotations

import hashlib

import chess


def replay_position(root_fen: str, prefix_uci: list[str]):
    """Rebuild complete known history; never infer repetition from FEN alone."""
    if not isinstance(root_fen, str) or not isinstance(prefix_uci, list):
        raise ValueError("root FEN and full UCI prefix are required")
    board = chess.Board(root_fen)
    if not board.is_valid():
        raise ValueError("invalid root position")
    for token in prefix_uci:
        move = chess.Move.from_uci(token)
        if move not in board.legal_moves:
            raise ValueError("illegal history prefix")
        board.push(move)
    if board.is_game_over(claim_draw=True):
        raise ValueError("terminal/claimable-draw root excluded")
    return board


def run_selected(rows, search_factory, *, expected_count=1024, nodes=8192, progress=None):
    """Run every selected row exactly once and emit bounded mover-POV targets.

    `rows` must already be the frozen, outcome-blind selector output. The
    factory must close over the immutable human-prior evaluator and return the
    pinned root search implementation. No result-based early stopping occurs.
    """
    rows = list(rows)
    if len(rows) != expected_count:
        raise ValueError("exact preregistered root count required")
    ids = [row.get("row_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("unique stable root IDs required")
    results = []
    for ordinal, row in enumerate(rows):
        board = replay_position(row["root_fen"], row["prefix_uci"])
        history = {"root_fen": row["root_fen"], "prefix_uci": row["prefix_uci"]}
        history_sha = hashlib.sha256(
            (row["root_fen"] + "\n" + " ".join(row["prefix_uci"])).encode()
        ).hexdigest()
        result = search_factory().search(board.copy(stack=True))
        if (
            result.nodes < 1
            or result.nodes > nodes
            or result.evaluations < 0
            or result.evaluations > result.nodes
            or not 0 <= result.completed_depth <= 8
            or not -2.0 <= result.value <= 2.0
        ):
            raise ValueError("search receipt outside pinned bounds")
        target = max(-1.0, min(1.0, float(result.value)))
        results.append(
            {
                "ordinal": ordinal,
                "row_id": row["row_id"],
                "trajectory_id": row["trajectory_id"],
                "history_sha256": history_sha,
                "history": history,
                "root_mover": "white" if board.turn else "black",
                "root_fen4": " ".join(board.fen().split()[:4]),
                "raw_q_mover": float(result.value),
                "target_mover": target,
                "exact_mate_score_range": abs(float(result.value)) > 1.0,
                "zero_value_is_not_a_draw_certificate": float(result.value) == 0.0,
                "nodes": result.nodes,
                "evaluations": result.evaluations,
                "completed_depth": result.completed_depth,
                "root_actions": result.root_actions,
            }
        )
        if progress is not None and (ordinal + 1) % 64 == 0:
            progress(ordinal + 1)
    return results
