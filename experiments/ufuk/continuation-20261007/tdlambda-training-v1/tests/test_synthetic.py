from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import chess
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from arena_adapter import validate_td_contract  # noqa: E402
from contract import (  # noqa: E402
    CONTRACT_SCHEMA,
    DATA_SCHEMA,
    NATIVE_SCHEMA,
    PHASE,
    PROVENANCE_SCHEMA,
    TARGET_METHOD,
    admit_contract,
)
from native import MATH, SCHEMA, Learner, bits_equal  # noqa: E402
from prepare import (  # noqa: E402
    BUILD_SEAL_SCHEMA,
    OUTPUT_DATA_SCHEMA,
    OUTPUT_PROVENANCE_SCHEMA,
    alias,
    build_tdlambda_dataset,
    canonical,
    parse_final_events,
    sha_bytes,
)
from tdlambda_targets import transform_episode  # noqa: E402


def row(root_id: str, ply: int, mover: str, q: float, *, played: bool = True) -> dict:
    return {
        "root_id": root_id,
        "local_ply": ply,
        "mover": mover,
        "clipped_q_mover": q,
        "selected_action_played": played,
        "protected_search_aliases": [],
        "episode_status": "completed-own-terminal",
        "own_wdl_mover": -1 if mover == "white" else 1,
        "train_eligible": True,
    }


def test_lambda_known_terminal_colors_and_unknown_endpoint() -> None:
    known = transform_episode([row("g", 0, "white", 0.2), row("g", 1, "black", 0.4)])
    assert known["rows"][0]["target_white"] == pytest.approx(-0.7)
    assert known["rows"][0]["target_mover"] == pytest.approx(-0.7)
    assert known["rows"][1]["target_mover"] == 1.0
    unknown_rows = [
        {
            **row("u", 0, "white", 0.2),
            "episode_status": "unknown-row-budget-prefix",
            "own_wdl_mover": None,
        },
        {
            **row("u", 1, "black", 0.4),
            "episode_status": "unknown-row-budget-prefix",
            "own_wdl_mover": None,
        },
    ]
    unknown = transform_episode(unknown_rows)
    # Last Q is a pre-action bootstrap. No post-action board value is invented.
    assert unknown["rows"][-1]["target_mover"] == pytest.approx(0.4)
    assert unknown["rows"][0]["target_white"] == pytest.approx(-0.4)
    assert all(not x["terminal_wdl_used"] for x in unknown["rows"])


def test_lambda_excludes_protected_trajectory_whole() -> None:
    protected = {**row("p", 0, "white", 0.8, played=False), "protected_search_aliases": [123]}
    out = transform_episode([protected])
    assert out["status"] == "excluded-protected-trajectory"
    assert out["rows"] == []


def test_parse_replays_terminal_and_mover_perspective() -> None:
    root = chess.STARTING_FEN
    board = chess.Board(root)
    moves = ["f2f3", "e7e5", "g2g4", "d8h4"]
    events = [{"type": "game_start", "root_id": "mate", "root_fen": root,
               "root_prefix_uci": [], "root_alias": alias(board)}]
    labels = []
    for ply, uci in enumerate(moves):
        mover = "white" if board.turn else "black"
        move = chess.Move.from_uci(uci)
        events.append({"type": "search_row", "row": {
            "root_id": "mate", "root_fen": root, "root_prefix_uci": [],
            "local_ply": ply, "history_uci": [m.uci() for m in board.move_stack],
            "fen4": " ".join(board.fen().split()[:4]), "mover": mover,
            "root_alias": alias(board), "raw_q_mover": 0.25,
            "clipped_q_mover": 0.25, "selected_best_uci": uci,
            "selected_action_played": True, "protected_search_aliases": [],
        }})
        board.push(move)
        labels.append(-1 if mover == "white" else 1)
    events.append({"type": "game_end", "root_id": "mate", "status": "completed-own-terminal",
                   "training_eligible": True, "row_labels": labels,
                   "final_state": {"fen4": " ".join(board.fen().split()[:4]),
                                   "alias": alias(board),
                                   "history_uci": moves, "protected": False}})
    episodes, trace = parse_final_events(b"\n".join(canonical(x) for x in events) + b"\n")
    assert [x["own_wdl_mover"] for x in episodes[0]] == labels
    assert len(trace) == 4
    assert trace[-1]["own_wdl_mover"] == 1


def test_parser_rejects_row_after_protected_unplayed_action() -> None:
    root = chess.STARTING_FEN
    board = chess.Board(root)
    prefix = []
    first = {"type": "search_row", "row": {
        "root_id": "p", "root_fen": root, "root_prefix_uci": prefix, "local_ply": 0,
        "history_uci": prefix, "fen4": " ".join(board.fen().split()[:4]),
        "mover": "white", "root_alias": alias(board), "raw_q_mover": 0.0,
        "clipped_q_mover": 0.0, "selected_best_uci": "e2e4",
        "selected_action_played": False, "protected_search_aliases": [456],
    }}
    second = {"type": "search_row", "row": {**first["row"], "local_ply": 1}}
    events = [
        {"type": "game_start", "root_id": "p", "root_fen": root,
         "root_prefix_uci": [], "root_alias": alias(board)},
        first, second,
    ]
    with pytest.raises(ValueError, match="no later search row"):
        parse_final_events(b"\n".join(canonical(x) for x in events))


def test_parser_excludes_protected_search_episode_even_when_final_board_is_clear() -> None:
    root = chess.STARTING_FEN
    board = chess.Board(root)
    row_event = {"type": "search_row", "row": {
        "root_id": "protected", "root_fen": root, "root_prefix_uci": [], "local_ply": 0,
        "history_uci": [], "fen4": " ".join(board.fen().split()[:4]),
        "mover": "white", "root_alias": alias(board), "raw_q_mover": 0.0,
        "clipped_q_mover": 0.0, "selected_best_uci": "e2e4",
        "selected_action_played": False, "protected_search_aliases": [456],
    }}
    events = [
        {"type": "game_start", "root_id": "protected", "root_fen": root,
         "root_prefix_uci": [], "root_alias": alias(board)},
        row_event,
        {"type": "game_end", "root_id": "protected",
         "status": "excluded-protected-trajectory", "training_eligible": False,
         "row_labels": [None],
         "final_state": {"fen4": " ".join(board.fen().split()[:4]),
                         "alias": alias(board), "history_uci": [], "protected": False}},
    ]
    episodes, trace = parse_final_events(b"\n".join(canonical(x) for x in events))
    assert episodes[0][0]["train_eligible"] is False
    assert trace[0]["own_wdl_mover"] is None


def test_contract_rejects_legacy_ownq_lineage() -> None:
    legacy = json.dumps({"schema": "own-kingbucket-sparse-training-data-v1",
                         "phase": "own-learning", "rows": [{}]})
    provenance = json.dumps({"schema": "NNUE-own1024-converted-data-provenance-v1"})
    contract = json.dumps({"phase": "own-learning", "native_schema": "old-v1"})
    with pytest.raises(ValueError):
        admit_contract(contract.encode(), legacy.encode(), provenance.encode())


def test_new_native_schema_replays_synthetic_updates_and_keeps_parent_baseline() -> None:
    from native import FEATURE_SCHEMA, MODEL_SCHEMA

    c = {
        "phase": "synthetic-tdlambda-test",
        "updates": 8,
        "seed": 77,
        "math": MATH,
        "feature_schema": FEATURE_SCHEMA,
        "native_schema": SCHEMA,
        "dataset_schema": "own-kingbucket-tdlambda-training-data-v1",
        "target_method": "frozen-parent-own-search-trajectory-tdlambda-v1",
        "lambda": 0.5,
    }
    rows = [{"indices": [0, 1, 64], "prior_logit": 0.1, "target": -0.5}]
    whole = Learner(c)
    whole.advance(rows, 8)
    first = Learner(c)
    first.advance(rows, 4)
    resumed = Learner(c, state=first.native())
    resumed.advance(rows, 8)
    assert bits_equal(whole.native(), resumed.native())
    assert whole.native()["schema"] == "own-kingbucket-nnue16-tdlambda-full-native-cpu-v1"
    assert whole.native()["step"] == 8
    assert not bits_equal(whole.native()["baseline"], whole.native()["model"])
    assert MODEL_SCHEMA == "own-kingbucket-nnue16-model-v1"


def test_arena_adapter_rejects_legacy_q_and_mc_phases() -> None:
    base = {
        "phase": "own-learning-tdlambda-v1",
        "native_schema": "own-kingbucket-nnue16-tdlambda-full-native-cpu-v1",
        "dataset_schema": "own-kingbucket-tdlambda-training-data-v1",
        "target_method": "frozen-parent-own-search-trajectory-tdlambda-v1",
        "lambda": 0.5,
        "updates": 64,
        "seed": 20262905,
        "legacy_native_resume_allowed": False,
    }
    validate_td_contract(base, 20262905)
    for changes in (
        {"phase": "own-learning"},
        {"native_schema": "own-kingbucket-nnue16-full-native-cpu-v1"},
        {"dataset_schema": "own-kingbucket-sparse-training-data-v1"},
        {"lambda": 1.0},
    ):
        with pytest.raises(ValueError):
            validate_td_contract({**base, **changes}, 20262905)


def test_full_target_builder_and_contract_on_synthetic_1024_rows() -> None:
    root = chess.STARTING_FEN
    events: list[dict] = []
    for game in range(256):
        root_id = f"synthetic:{game}"
        board = chess.Board(root)
        moves = ["f2f3", "e7e5", "g2g4", "d8h4"]
        events.append({"type": "game_start", "root_id": root_id, "root_fen": root,
                       "root_prefix_uci": [], "root_alias": alias(board)})
        for ply, uci in enumerate(moves):
            mover = "white" if board.turn else "black"
            events.append({"type": "search_row", "row": {
                "root_id": root_id, "root_fen": root, "root_prefix_uci": [],
                "local_ply": ply, "history_uci": [m.uci() for m in board.move_stack],
                "fen4": " ".join(board.fen().split()[:4]), "mover": mover,
                "root_alias": alias(board), "raw_q_mover": 0.25,
                "clipped_q_mover": 0.25, "selected_best_uci": uci,
                "selected_action_played": True, "protected_search_aliases": [],
            }})
            board.push_uci(uci)
        labels = [-1, 1, -1, 1]
        events.append({"type": "game_end", "root_id": root_id,
                       "status": "completed-own-terminal", "training_eligible": True,
                       "row_labels": labels,
                       "final_state": {"fen4": " ".join(board.fen().split()[:4]),
                                       "alias": alias(board), "history_uci": moves,
                                       "protected": False}})
    event_bytes = b"\n".join(canonical(event) for event in events) + b"\n"
    _, trace = parse_final_events(event_bytes)
    row_ids = [row["row_id"] for row in trace]
    data = canonical({"schema": "own-kingbucket-sparse-training-data-v1",
                      "phase": "own-learning",
                      "rows": [{"indices": [0], "prior_logit": 0.0, "target": 0.25}
                               for _ in row_ids]}) + b"\n"
    receipt = canonical({"schema": "own-nnue-ownq-collection-receipt-v2",
                        "status": "PASS-exact-row-budget", "train_rows": 1024,
                        "seed": 20262905, "parent_candidate_sha256": "a" * 64,
                        "events_sha256": sha_bytes(event_bytes),
                        "events_bytes": len(event_bytes), "training_row_ids": row_ids})
    source_provenance = canonical({
        "schema": "NNUE-own1024-converted-data-provenance-v1",
        "teacher_labels_used": False,
        "seed": 20262905,
        "dataset_sha256": sha_bytes(data),
        "collection_receipt_sha256": sha_bytes(receipt),
        "trace": trace,
        "parent_candidate": {"path": "/tmp/parent.pt", "sha256": "a" * 64},
        "inputs": {"registration": {"path": "/tmp/reg.json", "sha256": "b" * 64}},
    })
    conversion = canonical({"status": "PASS-own1024-fullhistory-trace-conversion-not-strength",
                           "dataset_sha256": sha_bytes(data),
                           "provenance_sha256": sha_bytes(source_provenance)})
    raw_inputs = {
        "source_dataset": data,
        "source_provenance": source_provenance,
        "collection_receipt": receipt,
        "events": event_bytes,
        "conversion_result": conversion,
    }
    build_seal = canonical({
        "schema": BUILD_SEAL_SCHEMA, "status": "registered", "mode": "convert",
        "lambda": 0.5, "expected_train_rows": 1024, "seed": 20262905,
        "first": time.time() - 1, "deadline": time.time() + 30,
        "operator_end_epoch": 1791448916.685839,
        "inputs": {key: sha_bytes(value) for key, value in raw_inputs.items()},
        "raw_collection_inputs": {"registration": {"path": "/tmp/reg.json", "sha256": "b" * 64}},
    })
    output_data, output_provenance = build_tdlambda_dataset(
        source_dataset_bytes=data,
        source_provenance_bytes=source_provenance,
        collection_receipt_bytes=receipt,
        events_bytes=event_bytes,
        conversion_result_bytes=conversion,
        build_seal_bytes=build_seal,
    )
    parsed_data, parsed_provenance = json.loads(output_data), json.loads(output_provenance)
    assert parsed_data["schema"] == OUTPUT_DATA_SCHEMA == DATA_SCHEMA
    assert parsed_data["phase"] == PHASE and len(parsed_data["rows"]) == 1024
    assert parsed_provenance["schema"] == OUTPUT_PROVENANCE_SCHEMA == PROVENANCE_SCHEMA
    assert parsed_provenance["known_terminal_source_rows"] == 1024
    contract = canonical({
        "schema": CONTRACT_SCHEMA, "phase": PHASE, "native_schema": NATIVE_SCHEMA,
        "dataset_schema": DATA_SCHEMA, "target_method": TARGET_METHOD, "lambda": 0.5,
        "initializer_kind": "named-parent-weights-only", "optimizer_mode": "fresh-adam",
        "initial_accepted_step": 0, "legacy_native_resume_allowed": False,
        "parent_candidate_sha256": "a" * 64,
        "dataset_sha256": sha_bytes(output_data),
        "target_provenance_sha256": sha_bytes(output_provenance),
    })
    admitted = admit_contract(contract, output_data, output_provenance)
    assert admitted["native_schema"] == NATIVE_SCHEMA
