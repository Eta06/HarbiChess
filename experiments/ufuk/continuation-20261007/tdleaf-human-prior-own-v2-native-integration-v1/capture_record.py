"""Create one journal row from the already completed captured search result."""

from __future__ import annotations

import hashlib
import json

import chess


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def capture_row(
    *,
    root_id: str,
    local_ply: int,
    root_board: chess.Board,
    played_action: chess.Move | None,
    result,
    episode_status: str,
    train_eligible: bool = True,
    protected_search_aliases: list[int] | None = None,
) -> dict:
    """Bind a result to a full played-root/PV leaf history without re-search.

    This is an adapter for a future journal producer, not a game runner. The
    search result must have already been completed and selected as the action.
    """
    if not isinstance(root_id, str) or not root_id or local_ply < 0:
        raise ValueError("stable root ID and nonnegative local ply required")
    if root_board.root().fen() != chess.STARTING_FEN:
        raise ValueError("only full standard-start histories are admitted")
    selected_action = result.move
    if selected_action is None or selected_action not in root_board.legal_moves:
        raise ValueError("completed search must select a legal root action")
    if type(train_eligible) is not bool:
        raise ValueError("train eligibility must be explicit")
    protected = list(protected_search_aliases or [])
    if any(type(alias) is not int for alias in protected):
        raise ValueError("protected aliases must be integer board projections")
    if played_action is None:
        if train_eligible and not protected:
            raise ValueError("unplayed selected move must be explicitly excluded")
    elif played_action != selected_action or played_action not in root_board.legal_moves:
        raise ValueError("the played action must equal the completed search move")
    pv = list(result.pv)
    if not pv or pv[0] != selected_action:
        raise ValueError("completed PV must start with the search-selected move")
    if result.leaf.ply != len(pv):
        raise ValueError("captured leaf ply differs from full PV length")
    history = [move.uci() for move in root_board.move_stack]
    leaf = root_board.copy(stack=True)
    for move in pv:
        if move not in leaf.legal_moves:
            raise ValueError("captured PV is illegal under complete root history")
        leaf.push(move)
    if result.leaf.kind == "static":
        if leaf.outcome(claim_draw=True) is not None or not -1 <= result.leaf.value <= 1:
            raise ValueError("static endpoint must be nonterminal and in evaluator range")
        leaf_value = float(result.leaf.value)
        terminal_mover_wdl = None
    elif result.leaf.kind == "terminal":
        outcome = leaf.outcome(claim_draw=True)
        if outcome is None:
            raise ValueError("terminal search endpoint failed rule replay")
        leaf_value = None
        terminal_mover_wdl = (
            0 if outcome.winner is None else (1 if outcome.winner == leaf.turn else -1)
        )
    else:
        raise ValueError("unknown PV leaf kind")

    root_fen = chess.STARTING_FEN
    return {
        "root_id": root_id,
        "local_ply": local_ply,
        "episode_status": episode_status,
        "root_fen": root_fen,
        "history_uci": history,
        "history_sha256": sha({"root_fen": root_fen, "history_uci": history}),
        "mover": "white" if root_board.turn else "black",
        "selected_action_uci": selected_action.uci(),
        "played_action_uci": played_action.uci() if played_action is not None else None,
        "played_action": played_action is not None,
        "train_eligible": bool(train_eligible),
        "protected_search_aliases": protected,
        "search_root_value": float(result.value),
        "search_nodes": result.nodes,
        "search_evaluations": result.evaluations,
        "search_completed_depth": result.completed_depth,
        "search_root_actions": result.root_actions,
        "pv_uci": [move.uci() for move in pv],
        "leaf_kind": result.leaf.kind,
        "leaf_value_mover": leaf_value,
        "leaf_search_value_mover": float(result.leaf.value),
        "leaf_terminal_mover_wdl": terminal_mover_wdl,
        "leaf_fen": leaf.fen(),
        "leaf_history_sha256": sha(
            {"root_fen": root_fen, "history_uci": history + [m.uci() for m in pv]}
        ),
    }
