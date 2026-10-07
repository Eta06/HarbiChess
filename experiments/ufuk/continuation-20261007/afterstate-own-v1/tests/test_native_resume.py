import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
import native  # noqa: E402
from model import FEATURE_SCHEMA  # noqa: E402


def synthetic_contract():
    return {
        "phase": native.SYNTHETIC_PHASE,
        "updates": 8,
        "math": native.MATH,
        "feature_schema": FEATURE_SCHEMA,
        "native_schema": native.SCHEMA,
        "dataset_schema": "own-kingbucket-afterstate-search-q-data-v1",
        "target_method": "own-selected-action-afterstate-negated-search-q-v1",
        "seed": 42,
    }


def rows():
    return [
        {"indices": [1 + index, 100 + index], "prior_logit": 0.03, "target": ((index % 3) - 1) / 2}
        for index in range(12)
    ]


def test_native_exact_whole_vs_4_4_resume(tmp_path):
    torch.set_num_threads(1)
    contract = synthetic_contract()
    whole = native.Learner(contract)
    whole.advance(rows(), 8)
    split = native.Learner(contract)
    split.advance(rows(), 4)
    path = tmp_path / "native.pt"
    torch.save(split.native(), path)
    fresh = native.load_native(path, contract)
    fresh.advance(rows(), 8)
    assert native.bits_equal(whole.native(), fresh.native())


def test_native_rejects_legacy_schema_and_wrong_target_math():
    torch.set_num_threads(1)
    contract = synthetic_contract()
    learner = native.Learner(contract)
    payload = learner.native()
    payload["schema"] = "own-kingbucket-nnue16-full-native-cpu-v1"
    with pytest.raises(ValueError, match="schema/contract/counter"):
        native.Learner(contract, state=payload)
    contract["target_method"] = "frozen-parent-own-search-closed-terminal-wdl-v1"
    with pytest.raises(ValueError, match="versioned phase/math"):
        native.Learner(contract)
