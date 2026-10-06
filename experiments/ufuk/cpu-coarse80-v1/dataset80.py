"""Strict conversion of registered own-search value rows into piece-square features."""

from __future__ import annotations

import hashlib
import math

from coarse80 import features_from_board

EXPECTED_ROW_KEYS = {
    "ordinal",
    "row_id",
    "trajectory_id",
    "history_sha256",
    "history",
    "root_mover",
    "root_fen4",
    "raw_q_mover",
    "target_mover",
    "exact_mate_score_range",
    "zero_value_is_not_a_draw_certificate",
    "nodes",
    "evaluations",
    "completed_depth",
    "root_actions",
}


def history_sha(root_fen, prefix):
    return hashlib.sha256((root_fen + "\n" + " ".join(prefix)).encode()).hexdigest()


def replay_legal_position(root_fen, prefix):
    """Reconstruct exact history with python-chess legality checks at each ply."""
    import chess

    board = chess.Board(root_fen)
    if not board.is_valid():
        raise ValueError("valid root position required")
    for text in prefix:
        move = chess.Move.from_uci(text)
        if move not in board.legal_moves:
            raise ValueError("illegal move in registered full history")
        board.push(move)
    return board


def labels_to_groups(payload, *, replay_position, classical_features, prior_weights, prior_scale):
    roots = payload.get("roots")
    if payload.get("schema") != "own-search-deeper-value-labels-v1" or not isinstance(roots, list):
        raise ValueError("registered own-search scalar labels required")
    if len(roots) != 1024:
        raise ValueError("exactly 1,024 registered Q roots required")
    if len(prior_weights) != 18 or not math.isfinite(prior_scale) or prior_scale <= 0:
        raise ValueError("pinned 18-feature prior constants required")
    seen = set()
    groups = {}
    for ordinal, row in enumerate(roots):
        if set(row) != EXPECTED_ROW_KEYS or row["ordinal"] != ordinal:
            raise ValueError("Q label row schema/order differs")
        if not isinstance(row["row_id"], str) or row["row_id"] in seen:
            raise ValueError("unique source row IDs required")
        seen.add(row["row_id"])
        history = row["history"]
        if set(history) != {"root_fen", "prefix_uci"}:
            raise ValueError("full root plus UCI prefix required")
        root_fen, prefix = history["root_fen"], history["prefix_uci"]
        if (
            not isinstance(root_fen, str)
            or not isinstance(prefix, list)
            or history_sha(root_fen, prefix) != row["history_sha256"]
        ):
            raise ValueError("full-history digest differs")
        board = replay_position(root_fen, prefix)
        if " ".join(board.fen().split()[:4]) != row["root_fen4"]:
            raise ValueError("replayed FEN differs")
        if ("white" if board.turn else "black") != row["root_mover"]:
            raise ValueError("mover perspective differs")
        if board.outcome(claim_draw=True) is not None:
            raise ValueError("terminal Q root must not be trained as nonterminal")
        legal_count = len(list(board.legal_moves))
        raw = float(row["raw_q_mover"])
        target = max(-1.0, min(1.0, raw))
        if (
            not math.isfinite(raw)
            or abs(raw) > 2.0
            or target != row["target_mover"]
            or row["exact_mate_score_range"] != (abs(raw) > 1.0)
            or row["zero_value_is_not_a_draw_certificate"] != (raw == 0.0)
            or not isinstance(row["nodes"], int)
            or not 0 < row["nodes"] <= 8192
            or type(row["root_actions"]) is not int
            or row["root_actions"] != legal_count
            or row["nodes"] < legal_count + 1
            or not isinstance(row["evaluations"], int)
            or not 0 <= row["evaluations"] <= row["nodes"]
            or not isinstance(row["completed_depth"], int)
            or not 1 <= row["completed_depth"] <= 8
        ):
            raise ValueError("bounded frozen own-Q target packet differs")
        x, _ = features_from_board(board)
        state = tuple(float(v) for v in classical_features(board))
        if len(state) != 18 or not all(math.isfinite(v) for v in state):
            raise ValueError("finite exact18 human-prior inputs required")
        prior_logit = sum(w * v for w, v in zip(prior_weights, state, strict=True)) / prior_scale
        groups.setdefault(row["trajectory_id"], []).append((x, prior_logit, target))
    if sum(len(rows) for rows in groups.values()) != 1024:
        raise ValueError("Q row coverage differs")
    return groups
