import hashlib
import json
import math
import sys
from pathlib import Path

import chess
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
import contract  # noqa: E402
from convert import afterstate_target  # noqa: E402


def canonical(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    )


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def history_sha(moves):
    return sha(canonical({"root_fen": chess.STARTING_FEN, "prefix_uci": moves})[:-1])


def fixture(terminal=False):
    rows, trace = [], []
    for index in range(1024):
        raw_q = ((index % 5) - 2) / 2
        target = -raw_q
        history_hash = history_sha([])
        after_hash = history_sha(["e2e4"])
        kind = "negated-selected-root-search-q"
        row_terminal = False
        if terminal and index == 0:
            kind, row_terminal, target = "exact-terminal-afterstate-wdl", True, -1.0
        row = {
            "source_row_id": f"game-{index}:0",
            "source_game_id": f"game-{index}",
            "source_local_ply": 0,
            "source_root_fen": chess.STARTING_FEN,
            "source_history_uci": [],
            "source_history_sha256": history_hash,
            "indices": [1, 3],
            "prior_logit": 0.0,
            "raw_q_mover": raw_q,
            "target": target,
            "target_kind": kind,
            "before_mover": "white",
            "after_mover": "black",
            "selected_action_uci": "e2e4",
            "selected_action_played": True,
            "afterstate_terminal": row_terminal,
            "afterstate_fen4": "8/8/8/8/8/8/8/8 b - -",
            "afterstate_history_uci": ["e2e4"],
            "afterstate_history_sha256": after_hash,
        }
        rows.append(row)
        trace.append(
            {
                "source_row_id": row["source_row_id"],
                "source_history_sha256": history_hash,
                "afterstate_history_sha256": after_hash,
                "root_search_q_mover": raw_q,
                "clipped_root_q_mover": max(-1.0, min(1.0, raw_q)),
                "target": target,
                "target_kind": kind,
                "selected_action_played": True,
                "afterstate_terminal": row_terminal,
                "source_outcome_status": "UNKNOWN",
            }
        )
    data = canonical(
        {
            "schema": contract.DATA_SCHEMA,
            "phase": contract.PHASE,
            "target_method": contract.TARGET_METHOD,
            "rows": rows,
        }
    )
    provenance = canonical(
        {
            "schema": contract.PROVENANCE_SCHEMA,
            "target_method": contract.TARGET_METHOD,
            "output_train_rows": 1024,
            "teacher_labels_used": False,
            "native_phase_required": contract.PHASE,
            "native_schema_required": contract.NATIVE_SCHEMA,
            "legacy_native_resume_allowed": False,
            "source_ownq_provenance_schema": "NNUE-own1024-converted-data-provenance-v1",
            "source_ownq_dataset_schema": "own-kingbucket-sparse-training-data-v1",
            "target_perspective": "afterstate side-to-move",
            "seed": 1,
            "unknown_source_rows_used_as_q": True,
            "unknown_q_rows": 1024,
            "protected_or_not_played_rows_excluded": True,
            "dataset_sha256": sha(data),
            "collection_receipt_sha256": "a" * 64,
            "source_receipt_sha256": "a" * 64,
            "parent_candidate": {"sha256": "b" * 64},
            "inputs": {"schema": "NNUE-own-afterstate1024-conversion-seal-v1"},
            "trace": trace,
        }
    )
    contract_bytes = canonical(
        {
            "schema": contract.CONTRACT_SCHEMA,
            "phase": contract.PHASE,
            "native_schema": contract.NATIVE_SCHEMA,
            "dataset_schema": contract.DATA_SCHEMA,
            "target_method": contract.TARGET_METHOD,
            "initializer_kind": "named-parent-weights-only",
            "optimizer_mode": "fresh-adam",
            "initial_accepted_step": 0,
            "legacy_native_resume_allowed": False,
            "dataset_sha256": sha(data),
            "target_provenance_sha256": sha(provenance),
            "parent_candidate_sha256": "b" * 64,
            "seed": 1,
            "raw_collection_inputs": {"schema": "NNUE-own-afterstate1024-conversion-seal-v1"},
        }
    )
    return contract_bytes, data, provenance


def test_afterstate_q_negates_clipped_root_mover_value():
    child = chess.Board()
    child.push_uci("e2e4")
    assert afterstate_target({"raw_q_mover": 0.7}, child) == (
        -0.7,
        "negated-selected-root-search-q",
    )
    assert afterstate_target({"raw_q_mover": 1.4}, child) == (
        -1.0,
        "negated-selected-root-search-q",
    )
    assert afterstate_target({"raw_q_mover": -1.4}, child) == (
        1.0,
        "negated-selected-root-search-q",
    )
    with pytest.raises(ValueError):
        afterstate_target({"raw_q_mover": math.inf}, child)


def test_afterstate_terminal_outcome_overrides_search_q():
    board = chess.Board()
    for move in ("f2f3", "e7e5", "g2g4", "d8h4"):
        board.push_uci(move)
    assert board.is_checkmate()
    assert afterstate_target({"raw_q_mover": 0.95}, board) == (
        -1.0,
        "exact-terminal-afterstate-wdl",
    )
    stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert stalemate.is_stalemate()
    assert afterstate_target({"raw_q_mover": 0.9}, stalemate) == (
        0.0,
        "exact-terminal-afterstate-wdl",
    )


def test_afterstate_contract_accepts_complete_q_trace():
    contract_bytes, data, provenance = fixture()
    result = contract.admit_contract(contract_bytes, data, provenance)
    assert result["phase"] == contract.PHASE


def test_afterstate_contract_accepts_exact_terminal_child_target():
    contract_bytes, data, provenance = fixture(terminal=True)
    result = contract.admit_contract(contract_bytes, data, provenance)
    assert result["phase"] == contract.PHASE


def test_afterstate_contract_rejects_wrong_sign_even_if_trace_resealed():
    contract_bytes, data, provenance = fixture()
    parsed = json.loads(data)
    parsed["rows"][0]["target"] *= -1
    trace = json.loads(provenance)
    trace["trace"][0]["target"] = parsed["rows"][0]["target"]
    data = canonical(parsed)
    trace["dataset_sha256"] = sha(data)
    provenance = canonical(trace)
    c = json.loads(contract_bytes)
    c["dataset_sha256"] = sha(data)
    c["target_provenance_sha256"] = sha(provenance)
    with pytest.raises(ValueError, match="negated clipped root Q"):
        contract.admit_contract(canonical(c), data, provenance)


def test_afterstate_contract_rejects_action_not_played_and_duplicate_id():
    contract_bytes, data, provenance = fixture()
    parsed = json.loads(data)
    parsed["rows"][0]["selected_action_played"] = False
    data = canonical(parsed)
    c = json.loads(contract_bytes)
    c["dataset_sha256"] = sha(data)
    with pytest.raises(ValueError, match="finite mover-perspective"):
        contract.admit_contract(canonical(c), data, provenance)

    contract_bytes, data, provenance = fixture()
    parsed = json.loads(data)
    parsed["rows"][1]["source_row_id"] = parsed["rows"][0]["source_row_id"]
    data = canonical(parsed)
    c = json.loads(contract_bytes)
    c["dataset_sha256"] = sha(data)
    with pytest.raises(ValueError, match="unique original source"):
        contract.admit_contract(canonical(c), data, provenance)


def test_afterstate_contract_rejects_terminal_q_and_unknown_draw_fabrication():
    contract_bytes, data, provenance = fixture()
    parsed = json.loads(data)
    parsed["rows"][0]["afterstate_terminal"] = True
    data = canonical(parsed)
    c = json.loads(contract_bytes)
    c["dataset_sha256"] = sha(data)
    with pytest.raises(ValueError, match="cannot label a terminal"):
        contract.admit_contract(canonical(c), data, provenance)


def test_afterstate_contract_rejects_legacy_native_phase():
    contract_bytes, data, provenance = fixture()
    c = json.loads(contract_bytes)
    c["native_schema"] = "own-kingbucket-nnue16-full-native-cpu-v1"
    with pytest.raises(ValueError, match="contract differs"):
        contract.admit_contract(canonical(c), data, provenance)
