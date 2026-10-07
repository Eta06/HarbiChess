from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from contract import (  # noqa: E402
    CONTRACT_SCHEMA,
    DATA_SCHEMA,
    NATIVE_SCHEMA,
    PHASE,
    PROVENANCE_SCHEMA,
    TARGET_METHOD,
    admit_contract,
)
from model import FEATURE_SCHEMA  # noqa: E402
from native import MATH, SCHEMA, Learner, bits_equal  # noqa: E402


def test_terminal_wdl_contract_rejects_unknowns_and_legacy_q_data() -> None:
    rows = [
        {
            "source_row_id": f"g:{i}",
            "indices": [0, 1, 64],
            "prior_logit": 0.1,
            "target": (-1, 0, 1)[i % 3],
        }
        for i in range(1024)
    ]
    dataset = {
        "schema": DATA_SCHEMA,
        "phase": PHASE,
        "target_method": TARGET_METHOD,
        "rows": rows,
    }
    provenance = {
        "schema": PROVENANCE_SCHEMA,
        "target_method": TARGET_METHOD,
        "output_train_rows": 1024,
        "teacher_labels_used": False,
        "native_phase_required": PHASE,
        "native_schema_required": NATIVE_SCHEMA,
        "parent_candidate": {"sha256": "a" * 64},
        "legacy_native_resume_allowed": False,
        "closed_terminal_rows": 1024,
        "unknown_rows_excluded": True,
        "trace": [
            {"source_row_id": row["source_row_id"], "selected_for_training": True} for row in rows
        ],
    }
    data_bytes = json.dumps(dataset, sort_keys=True, separators=(",", ":")).encode()
    provenance_bytes = json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
    contract = {
        "phase": PHASE,
        "native_schema": NATIVE_SCHEMA,
        "schema": CONTRACT_SCHEMA,
        "dataset_schema": DATA_SCHEMA,
        "target_method": TARGET_METHOD,
        "initializer_kind": "named-parent-weights-only",
        "optimizer_mode": "fresh-adam",
        "initial_accepted_step": 0,
        "legacy_native_resume_allowed": False,
        "dataset_sha256": __import__("hashlib").sha256(data_bytes).hexdigest(),
        "target_provenance_sha256": __import__("hashlib").sha256(provenance_bytes).hexdigest(),
        "parent_candidate_sha256": "a" * 64,
    }
    admitted = admit_contract(json.dumps(contract).encode(), data_bytes, provenance_bytes)
    assert admitted["target_method"] == TARGET_METHOD
    bad = dict(dataset)
    bad["rows"] = [dict(row) for row in rows]
    bad["rows"][2]["target"] = None
    bad_bytes = json.dumps(bad, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ValueError):
        admit_contract(json.dumps(contract).encode(), bad_bytes, provenance_bytes)


def test_full_native_resume_is_bit_exact_for_terminal_wdl_updates() -> None:
    contract = {
        "phase": "synthetic-closed-terminal-test",
        "updates": 8,
        "seed": 77,
        "math": MATH,
        "feature_schema": FEATURE_SCHEMA,
        "native_schema": SCHEMA,
        "dataset_schema": DATA_SCHEMA,
        "target_method": TARGET_METHOD,
    }
    rows = [
        {"indices": [0, 1, 64], "prior_logit": 0.1, "target": (-1, 0, 1)[i % 3]}
        for i in range(1024)
    ]
    whole = Learner(contract)
    whole.advance(rows, 8)
    first = Learner(contract)
    first.advance(rows, 4)
    resumed = Learner(contract, state=first.native())
    resumed.advance(rows, 8)
    assert bits_equal(whole.native(), resumed.native())
    assert whole.native()["schema"] == NATIVE_SCHEMA
    assert whole.native()["step"] == 8


def test_native_rejects_td_lambda_or_old_dataset_contract() -> None:
    contract = {
        "phase": "synthetic-closed-terminal-test",
        "updates": 8,
        "seed": 77,
        "math": MATH,
        "feature_schema": FEATURE_SCHEMA,
        "native_schema": SCHEMA,
        "dataset_schema": DATA_SCHEMA,
        "target_method": TARGET_METHOD,
        "lambda": 0.5,
    }
    with pytest.raises(ValueError):
        Learner(contract)
