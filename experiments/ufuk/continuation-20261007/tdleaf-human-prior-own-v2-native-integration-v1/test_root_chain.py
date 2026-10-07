import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def load_runner():
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location(
        "tdleaf_root_chain_test", HERE / "run_root_chain.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_h0_contract_accepts_only_named_literal_zero_parent():
    runner = load_runner()
    collection = {
        "parent_candidate": {"path": "/dev/shm/h0/candidate.pt", "sha256": "a" * 64},
        "parent_admission_seal": {"path": "/tmp/seal", "sha256": "b" * 64},
        "parent_admission_result": {"path": "/tmp/result", "sha256": "c" * 64},
    }
    contract = {
        "phase": "human-prior-tdleaf-own-learning-v2",
        "native_schema": "human-prior-tdleaf-own-nnue16-native-cpu-v2",
        "bootstrap_candidate_sha256": "a" * 64,
        "bootstrap_candidate_path": "/dev/shm/h0/candidate.pt",
        "parent_admission_seal": collection["parent_admission_seal"],
        "parent_admission_result": collection["parent_admission_result"],
        "teacher_labels_used_in_own_phase": False,
        "updates": 64,
        "tdleaf_lambda": 0.5,
    }
    runner.verify_h0_contract(contract, collection)
    contract["teacher_labels_used_in_own_phase"] = True
    with pytest.raises(ValueError):
        runner.verify_h0_contract(contract, collection)


def test_h0_contract_rejects_teacher_candidate_substitution():
    runner = load_runner()
    collection = {
        "parent_candidate": {"path": "/dev/shm/h0/candidate.pt", "sha256": "a" * 64},
        "parent_admission_seal": {"path": "/tmp/seal", "sha256": "b" * 64},
        "parent_admission_result": {"path": "/tmp/result", "sha256": "c" * 64},
    }
    contract = {
        "phase": "human-prior-tdleaf-own-learning-v2",
        "native_schema": "human-prior-tdleaf-own-nnue16-native-cpu-v2",
        "bootstrap_candidate_sha256": "d" * 64,
        "bootstrap_candidate_path": "/dev/shm/teacher/candidate.pt",
        "parent_admission_seal": collection["parent_admission_seal"],
        "parent_admission_result": collection["parent_admission_result"],
        "teacher_labels_used_in_own_phase": False,
        "updates": 64,
        "tdleaf_lambda": 0.5,
    }
    with pytest.raises(ValueError):
        runner.verify_h0_contract(contract, collection)
