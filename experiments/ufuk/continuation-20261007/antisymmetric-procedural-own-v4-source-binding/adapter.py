"""Feature-only representation bridge AFTER ROOT-qualified full raw trace replay."""

import hashlib
import json

import chess
from model import FEATURE_SCHEMA, board_indices, paired_board_indices

DATA_SCHEMA = "own-same-board-antisymmetric-sparse-training-data-v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def derive_rows(rows, validated_boards, provenance):
    """No raw receipt authorizer: upstream must replay/protect every exposure."""
    if len(rows) != len(validated_boards) or not rows:
        raise ValueError("aligned nonempty fully replayed same-data rows")
    output = []
    for row, board in zip(rows, validated_boards, strict=True):
        if not board.is_valid() or board.outcome(claim_draw=True) is not None:
            raise ValueError("nonterminal legal actual same-board input")
        if row["indices"] != board_indices(board):
            raise ValueError("original mover features must match replayed actual board")
        a, o = paired_board_indices(board)
        packet = dict(row, indices=a, opposite_indices=o)
        packet["paired_encoding_same_board"] = True
        packet["fullhistory_sha256"] = hashlib.sha256(
            canonical(
                dict(root_fen=board.root().fen(), history_uci=[m.uci() for m in board.move_stack])
            )
        ).hexdigest()
        packet["fen4"] = " ".join(board.fen().split()[:4])
        packet["mover"] = "white" if board.turn == chess.WHITE else "black"
        output.append(packet)
    return dict(
        schema=DATA_SCHEMA,
        feature_schema=FEATURE_SCHEMA,
        original_rows_sha256=hashlib.sha256(canonical(rows)).hexdigest(),
        target_and_prior_unchanged=True,
        paired_view_is_legal_transition=False,
        upstream_provenance=provenance,
        rows=output,
    )
