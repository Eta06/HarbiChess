"""Strict own-search legal-child ranking/Q-anchor data contract."""

from __future__ import annotations

import hashlib
import json
import math

import chess

DATA_SCHEMA = "own-kingbucket-afterstate-action-ranking-data-v1"
PROVENANCE_SCHEMA = "NNUE-own1024-afterstate-action-ranking-provenance-v1"
PHASE = "own-afterstate-action-ranking-v1"
NATIVE_SCHEMA = "own-kingbucket-nnue16-action-ranking-full-native-cpu-v1"
TARGET_METHOD = "own-selected-action-ranking-with-q-anchor-v1"
CONTRACT_SCHEMA = "own-kingbucket-afterstate-action-ranking-contract-v1"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _history_sha(root_fen: str, moves: list[str]) -> str:
    return _sha(_canonical({"root_fen": root_fen, "prefix_uci": moves}))


def _board_alias(board: chess.Board) -> int:
    token = min(board.board_fen(), board.mirror().board_fen()).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little", signed=True)


def admit_contract(contract_bytes: bytes, dataset_bytes: bytes, provenance_bytes: bytes) -> dict:
    contract = json.loads(contract_bytes)
    data = json.loads(dataset_bytes)
    provenance = json.loads(provenance_bytes)
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("phase") != PHASE
        or contract.get("native_schema") != NATIVE_SCHEMA
        or contract.get("dataset_schema") != DATA_SCHEMA
        or contract.get("target_method") != TARGET_METHOD
        or contract.get("initializer_kind") != "named-parent-weights-only"
        or contract.get("optimizer_mode") != "fresh-adam"
        or contract.get("initial_accepted_step") != 0
        or contract.get("legacy_native_resume_allowed") is not False
        or contract.get("batch_roots") != 16
        or contract.get("temperature") != 0.25
        or contract.get("selected_action_cross_entropy_weight") != 0.1
        or contract.get("dataset_sha256") != _sha(dataset_bytes)
        or contract.get("target_provenance_sha256") != _sha(provenance_bytes)
    ):
        raise ValueError("exact action-ranking phase/objective/native contract required")
    rows = data.get("rows", [])
    if (
        data.get("schema") != DATA_SCHEMA
        or data.get("phase") != PHASE
        or data.get("target_method") != TARGET_METHOD
        or data.get("temperature") != 0.25
        or data.get("selected_action_cross_entropy_weight") != 0.1
        or data.get("candidate_policy")
        != "all-legal-child-classes; only-selected-action-onehot; no-sibling-outcomes"
        or len(rows) != 1024
    ):
        raise ValueError("exact 1,024-root action-ranking data required")
    ids = [row.get("source_row_id") for row in rows]
    if any(not isinstance(value, str) for value in ids) or len(set(ids)) != 1024:
        raise ValueError("unique original source row IDs required")
    for row in rows:
        history = row.get("source_history_uci")
        root_fen = row.get("source_root_fen")
        q_target = row.get("q_target")
        candidates = row.get("candidates")
        selected = row.get("selected_index")
        if (
            row.get("selected_action_played") is not True
            or not isinstance(history, list)
            or root_fen != chess.STARTING_FEN
            or row.get("source_history_sha256") != _history_sha(root_fen, history)
            or type(selected) is not int
            or not isinstance(candidates, list)
            or not candidates
            or selected < 0
            or selected >= len(candidates)
            or isinstance(q_target, bool)
            or not isinstance(q_target, int | float)
            or not math.isfinite(float(q_target))
            or not -1 <= float(q_target) <= 1
        ):
            raise ValueError("selected own-Q root provenance and target required")
        board = chess.Board(root_fen)
        for token in history:
            move = chess.Move.from_uci(token)
            if move not in board.legal_moves:
                raise ValueError("illegal chronological root history")
            board.push(move)
        if board.outcome(claim_draw=True) is not None:
            raise ValueError("root action-ranking position must be nonterminal")
        legal = [move.uci() for move in board.legal_moves]
        if [candidate.get("action_uci") for candidate in candidates] != legal:
            raise ValueError("all legal actions in exact board order are required")
        if candidates[selected]["action_uci"] != row.get("selected_action_uci"):
            raise ValueError("one-hot search action must match selected played action")
        if (
            row.get("target_kind")
            not in {"negated-selected-root-search-q", "exact-terminal-afterstate-wdl"}
            or row.get("source_outcome_status") not in {"UNKNOWN", "known-own-terminal"}
            or not math.isfinite(float(row.get("raw_q_mover")))
            or abs(float(row["raw_q_mover"])) > 2
        ):
            raise ValueError("explicit own-Q/censor provenance required")
        observed = row.get("observed_child_aliases")
        expected_observed = sorted(
            candidate["visited_alias"]
            for candidate in candidates
            if candidate.get("terminal") is False
        )
        if (
            not isinstance(observed, list)
            or observed != sorted(set(observed))
            or observed != sorted(set(expected_observed))
            or type(row.get("search_alias_count")) is not int
            or row["search_alias_count"] < len(observed)
            or not isinstance(row.get("search_alias_sha256"), str)
            or len(row["search_alias_sha256"]) != 64
        ):
            raise ValueError("all nonterminal legal child aliases must be bound to source trace")
        for candidate in candidates:
            action = chess.Move.from_uci(candidate["action_uci"])
            child = board.copy(stack=True)
            child.push(action)
            child_history = [*history, action.uci()]
            if (
                candidate.get("afterstate_history_uci") != child_history
                or candidate.get("afterstate_history_sha256")
                != _history_sha(root_fen, child_history)
                or candidate.get("after_mover") != ("white" if child.turn else "black")
            ):
                raise ValueError("full legal child history/mover mismatch")
            outcome = child.outcome(claim_draw=True)
            if outcome is not None:
                value = (
                    0.0
                    if outcome.winner is None
                    else (1.0 if outcome.winner == child.turn else -1.0)
                )
                if (
                    candidate.keys()
                    != {
                        "action_uci",
                        "after_mover",
                        "afterstate_history_uci",
                        "afterstate_history_sha256",
                        "terminal",
                        "child_wdl",
                    }
                    or candidate.get("terminal") is not True
                    or candidate.get("child_wdl") != value
                    or any(key in candidate for key in ("indices", "prior_logit", "visited_alias"))
                ):
                    raise ValueError(
                        "terminal child must use exact constant WDL, with no inference"
                    )
            elif (
                candidate.keys()
                != {
                    "action_uci",
                    "after_mover",
                    "afterstate_history_uci",
                    "afterstate_history_sha256",
                    "terminal",
                    "indices",
                    "prior_logit",
                    "visited_alias",
                }
                or candidate.get("terminal") is not False
                or type(candidate.get("visited_alias")) is not int
                or candidate.get("visited_alias") not in row.get("observed_child_aliases", [])
                or not isinstance(candidate.get("indices"), list)
                or not 1 <= len(candidate["indices"]) <= 32
                or candidate["indices"] != sorted(set(candidate["indices"]))
                or any(
                    type(index) is not int or not 0 <= index < 12288
                    for index in candidate["indices"]
                )
                or not math.isfinite(float(candidate.get("prior_logit")))
                or candidate.get("visited_alias") != _board_alias(child)
                or any(key in candidate for key in ("child_wdl", "target", "q_target", "outcome"))
            ):
                raise ValueError("nonterminal child must be traced; no fabricated sibling label")
        selected_candidate = candidates[selected]
        if row["target_kind"] == "exact-terminal-afterstate-wdl":
            if (
                selected_candidate["terminal"] is not True
                or q_target != selected_candidate["child_wdl"]
            ):
                raise ValueError("selected terminal child must use exact child-side WDL Q anchor")
        elif selected_candidate["terminal"] is not False or float(q_target) != -max(
            -1.0, min(1.0, float(row["raw_q_mover"]))
        ):
            raise ValueError("selected child Q anchor must exactly negate clipped root Q")
    if (
        provenance.get("schema") != PROVENANCE_SCHEMA
        or provenance.get("phase") != PHASE
        or provenance.get("target_method") != TARGET_METHOD
        or provenance.get("output_train_rows") != 1024
        or provenance.get("dataset_sha256") != _sha(dataset_bytes)
        or provenance.get("teacher_labels_used") is not False
        or provenance.get("native_phase_required") != PHASE
        or provenance.get("native_schema_required") != NATIVE_SCHEMA
        or provenance.get("legacy_native_resume_allowed") is not False
        or provenance.get("source_afterstate_provenance_sha256") is None
        or provenance.get("source_receipt_sha256") != provenance.get("collection_receipt_sha256")
        or provenance.get("parent_candidate", {}).get("sha256")
        != contract.get("parent_candidate_sha256")
        or provenance.get("seed") != contract.get("seed")
        or provenance.get("inputs") != contract.get("raw_collection_inputs")
        or provenance.get("unknown_q_rows")
        != sum(row.get("source_outcome_status") == "UNKNOWN" for row in rows)
        or provenance.get("unknown_source_rows_used_as_q") != (provenance.get("unknown_q_rows") > 0)
    ):
        raise ValueError("ranking source/parent/native lineage differs")
    trace = provenance.get("trace")
    if not isinstance(trace, list) or len(trace) != len(rows):
        raise ValueError("target trace must cover every root in data order")
    for item, row in zip(trace, rows, strict=True):
        if (
            item.get("source_row_id") != row["source_row_id"]
            or item.get("selected_index") != row["selected_index"]
            or item.get("q_target") != row["q_target"]
            or item.get("candidate_count") != len(row["candidates"])
            or item.get("observed_child_aliases_sha256")
            != _sha(_canonical(row["observed_child_aliases"]))
            or item.get("search_alias_count") != row["search_alias_count"]
            or item.get("search_alias_sha256") != row["search_alias_sha256"]
            or item.get("source_outcome_status") != row["source_outcome_status"]
        ):
            raise ValueError("ranking trace target/candidate coverage mismatch")
    return contract
