"""Strict distinct contract for selected-action afterstate Q targets."""

from __future__ import annotations

import hashlib
import json
import math

DATA_SCHEMA = "own-kingbucket-afterstate-search-q-data-v1"
PROVENANCE_SCHEMA = "NNUE-own1024-afterstate-search-q-provenance-v1"
PHASE = "own-afterstate-search-q-learning-v1"
NATIVE_SCHEMA = "own-kingbucket-nnue16-afterstate-full-native-cpu-v1"
TARGET_METHOD = "own-selected-action-afterstate-negated-search-q-v1"
CONTRACT_SCHEMA = "own-kingbucket-afterstate-search-q-contract-v1"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _history_sha(root_fen: str, moves: list[str]) -> str:
    raw = json.dumps(
        {"root_fen": root_fen, "prefix_uci": moves},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return _sha(raw)


def admit_contract(contract_bytes: bytes, dataset_bytes: bytes, provenance_bytes: bytes) -> dict:
    contract = json.loads(contract_bytes)
    data = json.loads(dataset_bytes)
    provenance = json.loads(provenance_bytes)
    if (
        contract.get("phase") != PHASE
        or contract.get("native_schema") != NATIVE_SCHEMA
        or contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("dataset_schema") != DATA_SCHEMA
        or contract.get("target_method") != TARGET_METHOD
        or "lambda" in contract
        or contract.get("initializer_kind") != "named-parent-weights-only"
        or contract.get("optimizer_mode") != "fresh-adam"
        or contract.get("initial_accepted_step") != 0
        or contract.get("legacy_native_resume_allowed") is not False
        or contract.get("dataset_sha256") != _sha(dataset_bytes)
        or contract.get("target_provenance_sha256") != _sha(provenance_bytes)
    ):
        raise ValueError("versioned afterstate contract differs")
    rows = data.get("rows", [])
    if (
        data.get("schema") != DATA_SCHEMA
        or data.get("phase") != PHASE
        or data.get("target_method") != TARGET_METHOD
        or len(rows) != 1024
    ):
        raise ValueError("exact afterstate dataset schema/phase/row count required")
    ids = [row.get("source_row_id") for row in rows]
    if any(not isinstance(x, str) for x in ids) or len(set(ids)) != 1024:
        raise ValueError("unique original source row identities required")
    for row in rows:
        target = row.get("target")
        raw_q = row.get("raw_q_mover")
        if (
            isinstance(target, bool)
            or not isinstance(target, int | float)
            or not math.isfinite(float(target))
            or not -1.0 <= float(target) <= 1.0
            or isinstance(raw_q, bool)
            or not isinstance(raw_q, int | float)
            or not math.isfinite(float(raw_q))
            or abs(float(raw_q)) > 2.0
            or row.get("target_kind")
            not in {
                "negated-selected-root-search-q",
                "exact-terminal-afterstate-wdl",
            }
            or not isinstance(row.get("afterstate_history_sha256"), str)
            or len(row["afterstate_history_sha256"]) != 64
            or not isinstance(row.get("selected_action_uci"), str)
            or row.get("selected_action_played") is not True
            or not isinstance(row.get("source_game_id"), str)
            or type(row.get("source_local_ply")) is not int
            or row["source_local_ply"] < 0
            or not isinstance(row.get("source_history_sha256"), str)
            or not isinstance(row.get("source_root_fen"), str)
            or not isinstance(row.get("source_history_uci"), list)
            or row.get("afterstate_history_uci")
            != [*row.get("source_history_uci", []), row.get("selected_action_uci")]
            or row.get("after_mover") not in {"white", "black"}
            or row.get("before_mover") not in {"white", "black"}
            or row.get("after_mover") == row.get("before_mover")
            or row.get("afterstate_fen4") is None
            or row["source_history_sha256"]
            != _history_sha(row["source_root_fen"], row["source_history_uci"])
            or row["afterstate_history_sha256"]
            != _history_sha(row["source_root_fen"], row["afterstate_history_uci"])
        ):
            raise ValueError("finite mover-perspective afterstate target required")
        if row["target_kind"] == "exact-terminal-afterstate-wdl" and target not in {-1, 0, 1}:
            raise ValueError("terminal afterstate target must be exact WDL")
        if (
            row["target_kind"] == "negated-selected-root-search-q"
            and row.get("afterstate_terminal") is not False
        ):
            raise ValueError("search-Q target cannot label a terminal afterstate")
        if (
            row["target_kind"] == "exact-terminal-afterstate-wdl"
            and row.get("afterstate_terminal") is not True
        ):
            raise ValueError("exact WDL target requires a terminal afterstate")
    if (
        provenance.get("schema") != PROVENANCE_SCHEMA
        or provenance.get("target_method") != TARGET_METHOD
        or provenance.get("output_train_rows") != 1024
        or provenance.get("teacher_labels_used") is not False
        or provenance.get("native_phase_required") != PHASE
        or provenance.get("native_schema_required") != NATIVE_SCHEMA
        or provenance.get("legacy_native_resume_allowed") is not False
        or provenance.get("source_ownq_provenance_schema")
        != "NNUE-own1024-converted-data-provenance-v1"
        or provenance.get("source_ownq_dataset_schema") != "own-kingbucket-sparse-training-data-v1"
        or provenance.get("target_perspective") != "afterstate side-to-move"
        or type(provenance.get("unknown_source_rows_used_as_q")) is not bool
        or provenance.get("protected_or_not_played_rows_excluded") is not True
        or provenance.get("source_receipt_sha256") != provenance.get("collection_receipt_sha256")
        or provenance.get("parent_candidate", {}).get("sha256")
        != contract.get("parent_candidate_sha256")
        or provenance.get("seed") != contract.get("seed")
        or contract.get("raw_collection_inputs") != provenance.get("inputs")
    ):
        raise ValueError("afterstate source/parent/native boundary differs")
    trace = provenance.get("trace")
    selected_trace = [item.get("source_row_id") for item in trace or []]
    if not isinstance(trace, list) or selected_trace != ids or len(trace) != 1024:
        raise ValueError("afterstate target trace must match exact dataset row order")
    unknown_count = sum(item.get("source_outcome_status") == "UNKNOWN" for item in trace)
    if (
        provenance.get("unknown_q_rows") != unknown_count
        or provenance.get("unknown_source_rows_used_as_q") != (unknown_count > 0)
        or any(
            item.get("source_outcome_status") not in {"UNKNOWN", "known-own-terminal"}
            for item in trace
        )
        or provenance.get("dataset_sha256") != _sha(dataset_bytes)
    ):
        raise ValueError("source censor counts/dataset digest must reconcile")
    for item, row in zip(trace, rows, strict=True):
        raw_q = row["raw_q_mover"]
        if (
            item.get("target") != row.get("target")
            or item.get("target_kind") != row.get("target_kind")
            or item.get("afterstate_history_sha256") != row.get("afterstate_history_sha256")
            or item.get("source_history_sha256") != row.get("source_history_sha256")
            or item.get("clipped_root_q_mover")
            != max(-1.0, min(1.0, float(row.get("raw_q_mover"))))
            or item.get("selected_action_played") is not True
            or item.get("afterstate_terminal") != row.get("afterstate_terminal")
        ):
            raise ValueError("afterstate targets/history differ from immutable trace")
        if row["target_kind"] == "negated-selected-root-search-q" and (
            item.get("root_search_q_mover") != row["raw_q_mover"]
            or row["target"] != -max(-1.0, min(1.0, float(raw_q)))
        ):
            raise ValueError("afterstate target must be the exact negated clipped root Q")
    return contract
