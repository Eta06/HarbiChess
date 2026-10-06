"""Own prior8192 bestmove labels on the EXACT original frozen TRAIN selection."""

import hashlib
import math

import chess
from action_model import features


def run_selected(
    rows,
    search_factory,
    *,
    expected_count=1024,
    nodes=8192,
    progress=None,
    state_features=None,
    reference=None,
):
    if len(rows) != expected_count or len({r["row_id"] for r in rows}) != len(rows):
        raise ValueError("fixed unique selected roots")
    if state_features is None:
        raise ValueError("pinned classical18 context helper required")
    output = []
    for ordinal, row in enumerate(rows):
        board = chess.Board(row["root_fen"])
        if not board.is_valid():
            raise ValueError("valid original root")
        for text in row["prefix_uci"]:
            move = chess.Move.from_uci(text)
            if move not in board.legal_moves:
                raise ValueError("full legal history required")
            board.push(move)
        if board.is_game_over(claim_draw=True):
            raise ValueError("terminal root excluded")
        legal = sorted(board.legal_moves, key=lambda m: m.uci())
        original_fen, original_stack = board.fen(), tuple(board.move_stack)
        result = search_factory().search(board.copy(stack=True))
        if board.fen() != original_fen or tuple(board.move_stack) != original_stack:
            raise ValueError("root mutated")
        if (
            result.move not in legal
            or result.root_actions != len(legal)
            or not len(legal) + 1 <= result.nodes <= nodes
            or not 0 <= result.evaluations <= result.nodes
            or not 1 <= result.completed_depth <= 8
            or not math.isfinite(result.value)
            or not -2 <= result.value <= 2
        ):
            raise ValueError("selected action/complete root budget packet")
        h = hashlib.sha256(
            (row["root_fen"] + "\n" + " ".join(row["prefix_uci"])).encode()
        ).hexdigest()
        record = dict(
            ordinal=ordinal,
            row_id=row["row_id"],
            trajectory_id=row["trajectory_id"],
            history_sha256=h,
            history=dict(root_fen=row["root_fen"], prefix_uci=row["prefix_uci"]),
            root_mover="white" if board.turn else "black",
            root_fen4=" ".join(board.fen().split()[:4]),
            selected_uci=result.move.uci(),
            actions=[m.uci() for m in legal],
            features=[features(board, m) for m in legal],
            state18=list(state_features(board)),
            target=legal.index(result.move),
            raw_q_mover=float(result.value),
            nodes=result.nodes,
            evaluations=result.evaluations,
            completed_depth=result.completed_depth,
            root_actions=result.root_actions,
        )
        if len(record["state18"]) != 18:
            raise ValueError("exact18 context")
        if reference is not None:
            previous = reference[row["row_id"]]
            for key in [
                "history_sha256",
                "nodes",
                "evaluations",
                "completed_depth",
                "root_actions",
            ]:
                if record[key] != previous[key]:
                    raise ValueError("original value reanalysis reproducibility")
            if record["raw_q_mover"].hex() != float(previous["raw_q_mover"]).hex():
                raise ValueError("raw q bits differ")
        output.append(record)
        if progress is not None and (ordinal + 1) % 64 == 0:
            progress(ordinal + 1)
    return output
