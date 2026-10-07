import hashlib
import json
import sys
from pathlib import Path

import chess
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

import contract  # noqa: E402
from convert import alias, alias_values, legal_candidate  # noqa: E402
from native import (  # noqa: E402
    OWN_MODEL_SCHEMA,
    SYNTHETIC_PHASE,
    Learner,
    bits_equal,
)
from ranking import objective  # noqa: E402


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def history_sha(moves):
    return sha(canonical({"root_fen": chess.STARTING_FEN, "prefix_uci": moves}))


def synthetic_root(row_number):
    board = chess.Board()
    candidates = []
    observed = []
    for index, move in enumerate(board.legal_moves):
        child = board.copy(stack=True)
        child.push(move)
        child_alias = alias(child)
        observed.append(child_alias)
        history = [move.uci()]
        candidates.append(
            {
                "action_uci": move.uci(),
                "after_mover": "black",
                "afterstate_history_uci": history,
                "afterstate_history_sha256": history_sha(history),
                "terminal": False,
                "indices": [index + 1],
                "prior_logit": (index - 10) / 10.0,
                "visited_alias": child_alias,
            }
        )
    row_id = f"game-{row_number}:0"
    return {
        "source_row_id": row_id,
        "source_game_id": f"game-{row_number}",
        "source_local_ply": 0,
        "source_root_fen": chess.STARTING_FEN,
        "source_history_uci": [],
        "source_history_sha256": history_sha([]),
        "source_outcome_status": "UNKNOWN",
        "raw_q_mover": 0.0,
        "q_target": 0.0,
        "target_kind": "negated-selected-root-search-q",
        "selected_action_uci": candidates[0]["action_uci"],
        "selected_action_played": True,
        "selected_index": 0,
        "candidates": candidates,
        "search_alias_count": 1000,
        "search_alias_sha256": "a" * 64,
        "observed_child_aliases": sorted(set(observed)),
    }


def synthetic_contract_data(count=1024):
    rows = [synthetic_root(i) for i in range(count)]
    data = {
        "schema": contract.DATA_SCHEMA,
        "phase": contract.PHASE,
        "target_method": contract.TARGET_METHOD,
        "candidate_policy": (
            "all-legal-child-classes; only-selected-action-onehot; no-sibling-outcomes"
        ),
        "temperature": 0.25,
        "selected_action_cross_entropy_weight": 0.1,
        "rows": rows,
    }
    data_bytes = canonical(data) + b"\n"
    inputs = {"schema": "NNUE-own-action-ranking1024-conversion-seal-v1"}
    provenance = {
        "schema": contract.PROVENANCE_SCHEMA,
        "phase": contract.PHASE,
        "target_method": contract.TARGET_METHOD,
        "output_train_rows": count,
        "teacher_labels_used": False,
        "native_phase_required": contract.PHASE,
        "native_schema_required": contract.NATIVE_SCHEMA,
        "legacy_native_resume_allowed": False,
        "source_afterstate_provenance_sha256": "b" * 64,
        "dataset_sha256": sha(data_bytes),
        "seed": 1,
        "parent_candidate": {"sha256": "c" * 64},
        "source_receipt_sha256": "d" * 64,
        "collection_receipt_sha256": "d" * 64,
        "unknown_source_rows_used_as_q": True,
        "unknown_q_rows": count,
        "inputs": inputs,
        "trace": [
            {
                "source_row_id": row["source_row_id"],
                "selected_index": row["selected_index"],
                "q_target": row["q_target"],
                "candidate_count": len(row["candidates"]),
                "observed_child_aliases_sha256": sha(canonical(row["observed_child_aliases"])),
                "search_alias_count": row["search_alias_count"],
                "search_alias_sha256": row["search_alias_sha256"],
                "source_outcome_status": row["source_outcome_status"],
            }
            for row in rows
        ],
    }
    provenance_bytes = canonical(provenance) + b"\n"
    contract_obj = {
        "schema": contract.CONTRACT_SCHEMA,
        "phase": contract.PHASE,
        "native_schema": contract.NATIVE_SCHEMA,
        "dataset_schema": contract.DATA_SCHEMA,
        "target_method": contract.TARGET_METHOD,
        "initializer_kind": "named-parent-weights-only",
        "optimizer_mode": "fresh-adam",
        "initial_accepted_step": 0,
        "legacy_native_resume_allowed": False,
        "batch_roots": 16,
        "temperature": 0.25,
        "selected_action_cross_entropy_weight": 0.1,
        "dataset_sha256": sha(data_bytes),
        "target_provenance_sha256": sha(provenance_bytes),
        "parent_candidate_sha256": "c" * 64,
        "seed": 1,
        "raw_collection_inputs": inputs,
    }
    return canonical(contract_obj), data_bytes, provenance_bytes, rows


def test_contract_accepts_complete_legal_child_classes_and_onehot_selected_action():
    c, data, provenance, _ = synthetic_contract_data()
    assert contract.admit_contract(c, data, provenance)["phase"] == contract.PHASE


def test_contract_rejects_fabricated_sibling_q_or_outcome_after_reseal():
    c, data, provenance, _ = synthetic_contract_data()
    parsed = json.loads(data)
    parsed["rows"][0]["candidates"][1]["target"] = -1.0
    data = canonical(parsed) + b"\n"
    cobj, pobj = json.loads(c), json.loads(provenance)
    cobj["dataset_sha256"] = sha(data)
    pobj["dataset_sha256"] = sha(data)
    provenance = canonical(pobj) + b"\n"
    cobj["target_provenance_sha256"] = sha(provenance)
    with pytest.raises(ValueError, match="nonterminal child"):
        contract.admit_contract(canonical(cobj), data, provenance)


def test_contract_rejects_untraced_legal_child_even_when_trace_is_resealed():
    c, data, provenance, _ = synthetic_contract_data()
    parsed = json.loads(data)
    removed = parsed["rows"][0]["candidates"][1]["visited_alias"]
    parsed["rows"][0]["observed_child_aliases"].remove(removed)
    data = canonical(parsed) + b"\n"
    cobj, pobj = json.loads(c), json.loads(provenance)
    cobj["dataset_sha256"] = sha(data)
    pobj["dataset_sha256"] = sha(data)
    pobj["trace"][0]["observed_child_aliases_sha256"] = sha(
        canonical(parsed["rows"][0]["observed_child_aliases"])
    )
    provenance = canonical(pobj) + b"\n"
    cobj["target_provenance_sha256"] = sha(provenance)
    with pytest.raises(ValueError, match="all nonterminal legal child aliases"):
        contract.admit_contract(canonical(cobj), data, provenance)

    c, data, provenance, _ = synthetic_contract_data()
    parsed = json.loads(data)
    parsed["rows"][0]["candidates"][1]["visited_alias"] = 999
    parsed["rows"][0]["observed_child_aliases"] = sorted(
        {candidate["visited_alias"] for candidate in parsed["rows"][0]["candidates"]}
    )
    data = canonical(parsed) + b"\n"
    cobj, pobj = json.loads(c), json.loads(provenance)
    cobj["dataset_sha256"] = sha(data)
    pobj["dataset_sha256"] = sha(data)
    pobj["trace"][0]["observed_child_aliases_sha256"] = sha(
        canonical(parsed["rows"][0]["observed_child_aliases"])
    )
    provenance = canonical(pobj) + b"\n"
    cobj["target_provenance_sha256"] = sha(provenance)
    with pytest.raises(ValueError, match="nonterminal child"):
        contract.admit_contract(canonical(cobj), data, provenance)


def test_terminal_child_uses_side_to_move_wdl_and_skips_features():
    board = chess.Board()
    for token in ("f2f3", "e7e5", "g2g4"):
        board.push_uci(token)
    move = chess.Move.from_uci("d8h4")

    class Features:
        @staticmethod
        def board_indices(_board):
            raise AssertionError("terminal child must not invoke feature inference")

    class Prior:
        PRIOR = (1.0,)
        SCALE = 1.0

        @staticmethod
        def features(_board):
            raise AssertionError("terminal child must not invoke prior inference")

    candidate = legal_candidate(
        board, move, chess.STARTING_FEN, ["f2f3", "e7e5", "g2g4"], Features, Prior, set()
    )
    assert candidate["terminal"] is True
    assert candidate["child_wdl"] == -1.0
    assert "indices" not in candidate and "prior_logit" not in candidate


def test_nonterminal_child_must_be_present_in_original_root_trace():
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    child = board.copy(stack=True)
    child.push(move)

    class Features:
        @staticmethod
        def board_indices(_board):
            return [1, 4]

    class Prior:
        PRIOR = (1.0,)
        SCALE = 1.0

        @staticmethod
        def features(_board):
            return (0.25,)

    child_alias = alias(child)
    with pytest.raises(ValueError, match="absent from original root search trace"):
        legal_candidate(board, move, chess.STARTING_FEN, [], Features, Prior, set())
    candidate = legal_candidate(board, move, chess.STARTING_FEN, [], Features, Prior, {child_alias})
    assert candidate["terminal"] is False
    assert candidate["visited_alias"] == child_alias
    assert candidate["indices"] == [1, 4]
    assert candidate["prior_logit"] == 0.25


def test_alias_segment_decodes_exact_little_endian_int64_and_checks_sha(tmp_path):
    import struct

    values = [-9, 0, 17]
    payload = struct.pack("<3q", *values)
    chunk = tmp_path / "aliases.bin"
    chunk.write_bytes(payload)
    digest = sha(payload)
    row = {"search_alias_ref": {"file": chunk.name, "offset_bytes": 0, "count": 3}}
    receipt = {"alias_chunks": [{"file": chunk.name, "bytes": len(payload), "sha256": digest}]}
    seal = {"alias_chunks": {chunk.name: {"path": str(chunk), "sha256": digest}}}
    decoded, segment_sha = alias_values(row, receipt, seal, {})
    assert decoded == values
    assert segment_sha == digest
    chunk.write_bytes(payload + b"x")
    with pytest.raises(ValueError, match="pinned input/source SHA"):
        alias_values(row, receipt, seal, {})


def test_ranking_loss_has_selected_q_anchor_and_action_gradient():
    class FixedPredictionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.predictions = torch.nn.Parameter(torch.tensor([0.1, -0.2], dtype=torch.float64))

        def residual(self, _ids, _offsets):
            return self.predictions

    model = FixedPredictionModel()
    root = {
        "selected_index": 0,
        "q_target": 0.8,
        "candidates": [
            {"terminal": False, "indices": [1, 2], "prior_logit": 0.1},
            {"terminal": False, "indices": [3, 4], "prior_logit": -0.2},
        ],
    }
    loss = objective(model, [root])
    prediction = torch.tanh(model.predictions + torch.tensor([0.1, -0.2], dtype=torch.float64))
    expected = (prediction[0] - root["q_target"]) ** 2 + 0.1 * torch.nn.functional.cross_entropy(
        -prediction.reshape(1, -1) / 0.25, torch.tensor([0])
    )
    assert torch.allclose(loss, expected)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.predictions.grad is not None
    assert model.predictions.grad.abs().sum() > 0


def test_child_value_is_negated_to_rank_from_root_mover_pov():
    class ZeroModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

        def residual(self, _ids, _offsets):
            return self.anchor.reshape(1)

    model = ZeroModel()
    root = {
        "selected_index": 1,
        "q_target": -1.0,
        "candidates": [
            {"terminal": False, "indices": [1], "prior_logit": 0.0},
            {"terminal": True, "child_wdl": -1.0},
        ],
    }
    # A child-side loss (-1) is a winning root action (+1), so its rank logit
    # is +4 at T=.25 and selected-action CE remains small.
    assert objective(model, [root]).item() < 0.01


def synthetic_native_contract():
    return {
        "phase": SYNTHETIC_PHASE,
        "updates": 8,
        "math": __import__("native").MATH,
        "feature_schema": "mover-oriented-king2x2-relative-piece12-square64-v1",
        "native_schema": __import__("native").SCHEMA,
        "dataset_schema": "own-kingbucket-afterstate-action-ranking-data-v1",
        "target_method": "own-selected-action-ranking-with-q-anchor-v1",
        "seed": 13,
    }


def test_native_whole_pause_fresh_resume_fullbits():
    torch.set_num_threads(1)
    _, _, _, roots = synthetic_contract_data(24)
    whole = Learner(synthetic_native_contract())
    whole.advance(roots, 8)
    split = Learner(synthetic_native_contract())
    split.advance(roots, 4)
    resumed = load_native_from_payload(split.native(), synthetic_native_contract())
    resumed.advance(roots, 8)
    assert bits_equal(whole.native(), resumed.native())
    assert whole.candidate()["schema"] == OWN_MODEL_SCHEMA
    assert OWN_MODEL_SCHEMA != "own-kingbucket-nnue16-model-v1"


def load_native_from_payload(payload, contract_obj):
    learner = Learner(contract_obj, state=payload)
    assert bits_equal(learner.native(), payload)
    return learner
