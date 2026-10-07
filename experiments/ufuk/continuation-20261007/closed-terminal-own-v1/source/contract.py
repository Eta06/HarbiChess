"""Strict admission for the separately versioned closed-terminal MC phase."""

from __future__ import annotations

import hashlib
import json
import math

DATA_SCHEMA = "own-kingbucket-closed-terminal-training-data-v1"
PROVENANCE_SCHEMA = "NNUE-own1024-closed-terminal-data-provenance-v1"
PHASE = "own-closed-terminal-learning-v1"
NATIVE_SCHEMA = "own-kingbucket-nnue16-closed-terminal-full-native-cpu-v1"
TARGET_METHOD = "frozen-parent-own-search-closed-terminal-wdl-v1"
CONTRACT_SCHEMA = "own-kingbucket-closed-terminal-training-contract-v1"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


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
        raise ValueError("versioned closed-terminal training contract differs")
    rows = data.get("rows", [])
    if (
        data.get("schema") != DATA_SCHEMA
        or data.get("phase") != PHASE
        or data.get("target_method") != TARGET_METHOD
        or len(rows) != 1024
    ):
        raise ValueError("exact closed-terminal dataset schema/phase/row count required")
    ids = [row.get("source_row_id") for row in rows]
    if any(not isinstance(x, str) for x in ids) or len(set(ids)) != 1024:
        raise ValueError("unique source row identities required")
    for row in rows:
        target = row.get("target")
        if (
            isinstance(target, bool)
            or not isinstance(target, int | float)
            or not math.isfinite(float(target))
            or target not in {-1, 0, 1}
        ):
            raise ValueError("exact mover-perspective terminal WDL required")
    if (
        provenance.get("schema") != PROVENANCE_SCHEMA
        or provenance.get("target_method") != TARGET_METHOD
        or provenance.get("output_train_rows") != 1024
        or provenance.get("teacher_labels_used") is not False
        or provenance.get("native_phase_required") != PHASE
        or provenance.get("native_schema_required") != NATIVE_SCHEMA
        or not isinstance(provenance.get("parent_candidate"), dict)
        or not isinstance(provenance["parent_candidate"].get("sha256"), str)
        or provenance.get("legacy_native_resume_allowed") is not False
        or provenance.get("closed_terminal_rows") != 1024
        or provenance.get("unknown_rows_excluded") is not True
    ):
        raise ValueError("closed-terminal source provenance/native boundary differs")
    if contract.get("parent_candidate_sha256") != provenance["parent_candidate"]["sha256"]:
        raise ValueError("weights-only parent candidate identity differs")
    trace = provenance.get("trace")
    selected_trace = [x.get("source_row_id") for x in trace or [] if x.get("selected_for_training")]
    if (
        not isinstance(trace, list)
        or selected_trace != ids
        or sum(x.get("selected_for_training") is True for x in trace) != 1024
    ):
        raise ValueError("terminal target provenance must match exact dataset order")
    return contract
