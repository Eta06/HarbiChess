"""Strict dormant admission checks for the separate TD(lambda) native lineage."""

from __future__ import annotations

import hashlib
import json
import math

DATA_SCHEMA = "own-kingbucket-tdlambda-training-data-v1"
PROVENANCE_SCHEMA = "NNUE-own1024-tdlambda-target-provenance-v1"
PHASE = "own-learning-tdlambda-v1"
NATIVE_SCHEMA = "own-kingbucket-nnue16-tdlambda-full-native-cpu-v1"
TARGET_METHOD = "frozen-parent-own-search-trajectory-tdlambda-v1"
CONTRACT_SCHEMA = "own-kingbucket-tdlambda-training-contract-v1"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def admit_contract(contract_bytes: bytes, dataset_bytes: bytes, provenance_bytes: bytes) -> dict:
    """Reject legacy Q-only/MC artifacts and mismatched TD target lineage."""
    contract = json.loads(contract_bytes)
    data = json.loads(dataset_bytes)
    provenance = json.loads(provenance_bytes)
    if (contract.get("phase") != PHASE or contract.get("native_schema") != NATIVE_SCHEMA
            or contract.get("schema") != CONTRACT_SCHEMA
            or contract.get("dataset_schema") != DATA_SCHEMA
            or contract.get("target_method") != TARGET_METHOD
            or contract.get("lambda") != 0.5
            or contract.get("initializer_kind") != "named-parent-weights-only"
            or contract.get("optimizer_mode") != "fresh-adam"
            or contract.get("initial_accepted_step") != 0
            or contract.get("legacy_native_resume_allowed") is not False
            or contract.get("dataset_sha256") != _sha(dataset_bytes)
            or contract.get("target_provenance_sha256") != _sha(provenance_bytes)):
        raise ValueError("versioned TD(lambda) training contract differs")
    if (data.get("schema") != DATA_SCHEMA or data.get("phase") != PHASE
            or data.get("target_method") != TARGET_METHOD or data.get("lambda") != 0.5
            or len(data.get("rows", [])) != 1024):
        raise ValueError("exact TD(lambda) dataset schema/phase/row count required")
    row_ids = [row.get("source_row_id") for row in data["rows"]]
    if any(not isinstance(row_id, str) for row_id in row_ids) or len(set(row_ids)) != 1024:
        raise ValueError("unique source row identities are required")
    for row in data["rows"]:
        target = row.get("target")
        if (isinstance(target, bool) or not isinstance(target, int | float)
                or not math.isfinite(float(target)) or not -1.0 <= float(target) <= 1.0):
            raise ValueError("finite mover-perspective target in [-1,1] required")
    if (provenance.get("schema") != PROVENANCE_SCHEMA
            or provenance.get("target_method") != TARGET_METHOD
            or provenance.get("lambda") != 0.5
            or provenance.get("output_train_rows") != 1024
            or provenance.get("teacher_labels_used") is not False
            or provenance.get("native_phase_required") != PHASE
            or provenance.get("native_schema_required") != NATIVE_SCHEMA
            or not isinstance(provenance.get("parent_candidate"), dict)
            or not isinstance(provenance["parent_candidate"].get("sha256"), str)
            or provenance.get("legacy_native_resume_allowed") is not False):
        raise ValueError("TD(lambda) target provenance/native boundary differs")
    if contract.get("parent_candidate_sha256") != provenance["parent_candidate"]["sha256"]:
        raise ValueError("weights-only parent candidate identity differs")
    if (provenance.get("source_train_rows") != 1024
            or provenance.get("known_terminal_source_rows", -1)
            + provenance.get("unknown_bootstrap_source_rows", -1) != 1024
            or provenance.get("output_train_rows") != 1024):
        raise ValueError("target provenance row accounting differs")
    trace = provenance.get("targets")
    if not isinstance(trace, list) or [row.get("source_row_id") for row in trace] != row_ids:
        raise ValueError("target provenance must match exact dataset row order")
    return contract
