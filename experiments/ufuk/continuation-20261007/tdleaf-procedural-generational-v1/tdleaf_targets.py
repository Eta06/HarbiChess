"""Pure, versioned search-leaf TD(lambda) targets; no game/search/model calls."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping

import chess

LAMBDA = 0.5
SCHEMA = "own-tdleaf-search-leaf-sequence-targets-v2"
KNOWN = "completed-own-terminal"
UNKNOWN = {"unknown-ply-cap", "unknown-row-budget-prefix"}


def _canon(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value: object) -> str:
    return hashlib.sha256(_canon(value)).hexdigest()


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _board(root_fen: str, history: list[str]) -> chess.Board:
    if root_fen != chess.STARTING_FEN or not isinstance(history, list):
        raise ValueError("full standard-start history required")
    board = chess.Board(root_fen)
    for token in history:
        move = chess.Move.from_uci(token)
        if move not in board.legal_moves:
            raise ValueError("illegal chronological game history")
        board.push(move)
    return board


def make_episode(rows: Iterable[Mapping], *, terminal_white_wdl: int | None) -> dict:
    """Convert captured PV leaves to white-oriented λ-return targets.

    `search_root_value` is the source search's mover-oriented score. The model
    value at a static PV leaf is mover-oriented for that leaf. The parity sign
    is checked before constructing targets. Known terminal WDL is only admitted
    for completed rule-replayed episodes; UNKNOWN tails bootstrap from the last
    logged pre-action search score. Protected or unplayed episodes are excluded
    wholesale.
    """
    items = [dict(row) for row in rows]
    if not items:
        raise ValueError("nonempty search-leaf episode required")
    root_ids = {r.get("root_id") for r in items}
    statuses = {r.get("episode_status") for r in items}
    if len(root_ids) != 1 or not isinstance(next(iter(root_ids)), str):
        raise ValueError("one stable root_id required")
    if len(statuses) != 1 or not isinstance(next(iter(statuses)), str):
        raise ValueError("one explicit final episode status required")
    status = next(iter(statuses))
    items.sort(key=lambda r: r.get("local_ply", -1))

    excluded = False
    for index, row in enumerate(items):
        if row.get("local_ply") != index:
            raise ValueError("contiguous chronological local plies required")
        if row.get("root_fen") != chess.STARTING_FEN:
            raise ValueError("standard-start full history required")
        board = _board(row["root_fen"], row.get("history_uci"))
        if board.outcome(claim_draw=True) is not None:
            raise ValueError("pre-action root must be nonterminal")
        mover = "white" if board.turn else "black"
        if row.get("mover") != mover or row.get("history_sha256") != _sha(
            {"root_fen": row["root_fen"], "history_uci": row["history_uci"]}
        ):
            raise ValueError("root history/mover binding differs")
        selected_uci = row.get("selected_action_uci")
        selected_move = chess.Move.from_uci(selected_uci)
        if selected_move not in board.legal_moves:
            raise ValueError("selected search action must be legal at the recorded root")
        if row.get("played_action") is not True or row.get("train_eligible") is not True:
            excluded = True
        action_uci = row.get("played_action_uci")
        if row.get("played_action") is True:
            if action_uci != selected_uci:
                raise ValueError("played move must equal the selected search action")
            action = chess.Move.from_uci(action_uci)
            if action not in board.legal_moves:
                raise ValueError("played action must be legal at the recorded root")
        elif action_uci is not None:
            raise ValueError("unplayed selected action cannot be recorded as a game transition")
        if row.get("protected_search_aliases") != []:
            if not isinstance(row.get("protected_search_aliases"), list):
                raise ValueError("protected search aliases must be an explicit list")
            if row["protected_search_aliases"]:
                excluded = True

        pv = row.get("pv_uci")
        if not isinstance(pv, list) or not pv or pv[0] != selected_uci:
            raise ValueError("full PV UCI list required")
        leaf_board = board.copy(stack=True)
        for token in pv:
            move = chess.Move.from_uci(token)
            if move not in leaf_board.legal_moves:
                raise ValueError("illegal captured PV continuation")
            leaf_board.push(move)
        if row.get("leaf_fen") != leaf_board.fen() or row.get("leaf_history_sha256") != _sha(
            {"root_fen": row["root_fen"], "history_uci": row["history_uci"] + pv}
        ):
            raise ValueError("captured PV leaf full-history identity differs")
        leaf_outcome = leaf_board.outcome(claim_draw=True)
        leaf_kind = row.get("leaf_kind")
        if leaf_outcome is None:
            if leaf_kind != "static" or row.get("leaf_terminal_mover_wdl") is not None:
                raise ValueError("nonterminal PV endpoint must be a static model leaf")
            leaf_value = _number(row.get("leaf_value_mover"), "leaf_value_mover")
            if not -1 <= leaf_value <= 1:
                raise ValueError("static leaf value range differs")
            parity = -1.0 if len(pv) % 2 else 1.0
            root_score = _number(row.get("search_root_value"), "search_root_value")
            if not -2.0 <= root_score <= 2.0:
                raise ValueError("root search score is outside the source evaluator range")
            expected = parity * leaf_value
            if not math.isclose(root_score, expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("root score does not equal signed selected PV leaf value")
        else:
            if leaf_kind != "terminal":
                raise ValueError("terminal PV endpoint must be marked terminal")
            expected_leaf_wdl = (
                0
                if leaf_outcome.winner is None
                else (1 if leaf_outcome.winner == leaf_board.turn else -1)
            )
            if row.get("leaf_terminal_mover_wdl") != expected_leaf_wdl:
                raise ValueError("terminal PV endpoint WDL/perspective differs")
            if row.get("leaf_value_mover") is not None:
                raise ValueError("terminal leaves must not carry a learned NN value")

        if index + 1 < len(items):
            following = items[index + 1]
            if row.get("played_action") is not True:
                raise ValueError("an unplayed source row must end its episode")
            expected_history = row["history_uci"] + [action_uci]
            if following.get("history_uci") != expected_history:
                raise ValueError("game successor history does not follow the played action")

    if excluded:
        return {
            "schema": SCHEMA,
            "root_id": next(iter(root_ids)),
            "status": "excluded-protected-or-unplayed-episode",
            "lambda": LAMBDA,
            "source_rows": len(items),
            "rows": [],
        }
    if status not in UNKNOWN | {KNOWN}:
        raise ValueError("episode status must preserve known terminal or UNKNOWN cap")
    if status == KNOWN:
        if type(terminal_white_wdl) is not int or terminal_white_wdl not in {-1, 0, 1}:
            raise ValueError("completed own terminal needs exact White WDL")
        last = items[-1]
        last_board = _board(last["root_fen"], last["history_uci"])
        last_board.push_uci(last["played_action_uci"])
        outcome = last_board.outcome(claim_draw=True)
        if outcome is None:
            raise ValueError("known episode must terminate after its final played action")
        actual = 0 if outcome.winner is None else (1 if outcome.winner == chess.WHITE else -1)
        if terminal_white_wdl != actual:
            raise ValueError("terminal WDL disagrees with replayed legal game result")
        endpoint = float(terminal_white_wdl)
        endpoint_kind = "exact-own-terminal-wdl"
    else:
        if terminal_white_wdl is not None:
            raise ValueError("UNKNOWN cap must not carry a draw or terminal label")
        last = items[-1]
        last_board = _board(last["root_fen"], last["history_uci"])
        last_board.push_uci(last["played_action_uci"])
        if last_board.outcome(claim_draw=True) is not None:
            raise ValueError("terminal game endpoint cannot be mislabeled UNKNOWN")
        endpoint = None
        endpoint_kind = "last-recorded-preaction-search-bootstrap"

    white_predictions = []
    gradient_signs = []
    for row in items:
        root_score = max(-1.0, min(1.0, _number(row["search_root_value"], "search_root_value")))
        root_sign = 1.0 if row["mover"] == "white" else -1.0
        white_predictions.append(root_sign * root_score)
        if row["leaf_kind"] == "static":
            # White score = root-to-White sign x (-1)^PV plies x leaf mover score.
            gradient_signs.append(root_sign * (-1.0 if len(row["pv_uci"]) % 2 else 1.0))
        else:
            gradient_signs.append(None)

    targets_white = [0.0] * len(items)
    if endpoint is not None:
        targets_white[-1] = endpoint
    else:
        targets_white[-1] = white_predictions[-1]
    for index in range(len(items) - 2, -1, -1):
        targets_white[index] = (1.0 - LAMBDA) * white_predictions[
            index + 1
        ] + LAMBDA * targets_white[index + 1]

    target_rows = []
    for i, (row, target_white, sign) in enumerate(
        zip(items, targets_white, gradient_signs, strict=True)
    ):
        target_leaf = None if sign is None else sign * target_white
        target_rows.append(
            {
                "source_row_id": f"{row['root_id']}:{i}",
                "local_ply": i,
                "root_history_sha256": row["history_sha256"],
                "pv_leaf_history_sha256": row["leaf_history_sha256"],
                "mover": row["mover"],
                "pv_uci": row["pv_uci"],
                "pv_length": len(row["pv_uci"]),
                "gradient_sign_leaf_to_white": sign,
                "search_root_white_prediction": white_predictions[i],
                "tdleaf_target_white": target_white,
                "tdleaf_target_leaf_mover": target_leaf,
                "gradient_mask": sign is not None,
                "leaf_kind": row["leaf_kind"],
                "endpoint": endpoint_kind,
            }
        )
    return {
        "schema": SCHEMA,
        "root_id": next(iter(root_ids)),
        "status": "targeted-known-terminal" if status == KNOWN else "targeted-unknown-bootstrap",
        "lambda": LAMBDA,
        "target_method": "searched-PV-leaf-TD(lambda)-white-return-v2",
        "source_rows": len(items),
        "endpoint": endpoint_kind,
        "terminal_white_wdl": terminal_white_wdl,
        "rows": target_rows,
    }


def transform_episodes(episodes: Iterable[tuple[Iterable[Mapping], int | None]]) -> list[dict]:
    return [make_episode(rows, terminal_white_wdl=wdl) for rows, wdl in episodes]
