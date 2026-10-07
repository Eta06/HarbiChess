"""Admission/value bridge for a selected-action afterstate Q child on the unchanged NNUE arena."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from contract import (
    DATA_SCHEMA,
    NATIVE_SCHEMA,
    PHASE,
    PROVENANCE_SCHEMA,
    TARGET_METHOD,
    admit_contract,
)

FIT_STATUS = "PASS-afterstate-fixed-phase-and-fresh-native-loads-not-strength"
MODEL_SCHEMA = "own-kingbucket-nnue16-model-v1"


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(ref: dict) -> Path:
    path = Path(ref["path"]).resolve()
    if sha(path) != ref["sha256"]:
        raise ValueError("pinned selected-action afterstate Q arena input SHA differs")
    return path


def validate_afterstate_contract(contract: dict, seed: int) -> None:
    if (
        contract.get("phase") != PHASE
        or contract.get("native_schema") != NATIVE_SCHEMA
        or contract.get("dataset_schema") != DATA_SCHEMA
        or contract.get("target_method") != TARGET_METHOD
        or contract.get("updates") != 64
        or contract.get("seed") != seed
        or contract.get("legacy_native_resume_allowed") is not False
    ):
        raise ValueError("separate fully identified selected-action afterstate Q child required")


def _validate_model_weights(state: dict) -> None:
    import torch

    expected = {"embedding.weight": (12288, 16), "head.weight": (1, 16), "head.bias": (1,)}
    if state.keys() != expected.keys() or any(
        not torch.is_tensor(state[key])
        or state[key].device.type != "cpu"
        or state[key].dtype != torch.float64
        or tuple(state[key].shape) != shape
        or not torch.isfinite(state[key]).all()
        for key, shape in expected.items()
    ):
        raise ValueError("exact finite NNUE16 candidate storage required")


def _bitwise_equal(left: object, right: object) -> bool:
    import torch

    if torch.is_tensor(left):
        return (
            torch.is_tensor(right)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(
                left.detach().reshape(-1).view(torch.uint8),
                right.detach().reshape(-1).view(torch.uint8),
            )
        )
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_bitwise_equal(left[k], right[k]) for k in left)
    if isinstance(left, list | tuple):
        return len(left) == len(right) and all(
            _bitwise_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def admit_afterstate_child(record: dict) -> dict:
    """Verify a completed proof/fit and final candidate before arena registration.

    The caller still must run the registered original search/profile/arena
    protocol. This function only admits the candidate value endpoint.
    """
    seed = record.get("seed")
    if type(seed) is not int:
        raise ValueError("seed-bound selected-action afterstate Q candidate record required")
    contract_path = pin(record["contract"])
    data_path = pin(record["dataset"])
    provenance_path = pin(record["target_provenance"])
    candidate_path = pin(record["candidate"])
    parent_path = pin(record["parent_candidate"])
    proof_path = pin(record["proof_result"])
    fit_path = pin(record["fit_result"])
    proof_contract_path = pin(record["proof_contract"])
    proof_registration_path = pin(record["proof_registration"])
    fit_registration_path = pin(record["fit_registration"])
    contract = json.loads(contract_path.read_bytes())
    contract_bytes, data_bytes, provenance_bytes = (
        contract_path.read_bytes(),
        data_path.read_bytes(),
        provenance_path.read_bytes(),
    )
    data = json.loads(data_bytes)
    provenance = json.loads(provenance_bytes)
    proof = json.loads(proof_path.read_bytes())
    fit = json.loads(fit_path.read_bytes())
    proof_contract = json.loads(proof_contract_path.read_bytes())
    proof_registration = json.loads(proof_registration_path.read_bytes())
    fit_registration = json.loads(fit_registration_path.read_bytes())
    admit_contract(contract_bytes, data_bytes, provenance_bytes)
    validate_afterstate_contract(contract, seed)
    if (
        contract.get("dataset_sha256") != sha(data_path)
        or contract.get("target_provenance_sha256") != sha(provenance_path)
        or contract.get("parent_candidate_sha256") != sha(parent_path)
        or contract.get("bootstrap_candidate_sha256") != sha(parent_path)
        or contract.get("bootstrap_candidate_path") != str(parent_path)
        or contract.get("execution_mode") != "fresh-fit"
    ):
        raise ValueError("selected-action afterstate Q contract/data/parent bytes differ")
    if (
        data.get("schema") != DATA_SCHEMA
        or data.get("phase") != PHASE
        or data.get("target_method") != TARGET_METHOD
        or len(data.get("rows", [])) != 1024
        or provenance.get("schema") != PROVENANCE_SCHEMA
        or provenance.get("dataset_sha256") != sha(data_path)
        or provenance.get("parent_candidate", {}).get("sha256") != sha(parent_path)
        or provenance.get("teacher_labels_used") is not False
    ):
        raise ValueError("actual own-target provenance required; teacher targets rejected")
    if (
        proof.get("status") != FIT_STATUS
        or proof.get("mode") != "proof"
        or proof.get("own_updates") != 8
        or proof.get("full_payload_bits_equal") is not True
        or proof.get("seed") != seed
        or proof.get("dataset_sha256") != sha(data_path)
        or proof.get("target_provenance_sha256") != sha(provenance_path)
        or proof.get("weights_only_initializer", {}).get("sha256") != sha(parent_path)
        or proof.get("contract_sha256") != record["proof_contract"]["sha256"]
    ):
        raise ValueError("fresh selected-action afterstate Q 8/4/8 native qualification required")
    if [x.get("step") for x in proof.get("native_payloads", [])] != [0, 8, 0, 4, 4, 8] or any(
        x.get("actual_fresh_process") is not True for x in proof.get("native_payloads", [])
    ):
        raise ValueError("all six fresh-process proof payload loads required")
    for payload in proof["native_payloads"]:
        pin(payload)
    if (
        proof_contract.get("phase") != PHASE
        or proof_contract.get("execution_mode") != "proof"
        or proof_contract.get("dataset_sha256") != sha(data_path)
        or proof_contract.get("target_provenance_sha256") != sha(provenance_path)
        or proof_contract.get("parent_candidate_sha256") != sha(parent_path)
    ):
        raise ValueError("proof contract must bind same data/parent")
    if (
        fit.get("status") != FIT_STATUS
        or fit.get("mode") != "fresh-fit"
        or fit.get("own_updates") != 64
        or fit.get("seed") != seed
        or fit.get("dataset_sha256") != sha(data_path)
        or fit.get("target_provenance_sha256") != sha(provenance_path)
        or fit.get("contract_sha256") != record["contract"]["sha256"]
        or fit.get("weights_only_initializer", {}).get("sha256") != sha(parent_path)
        or [x.get("step") for x in fit.get("native_payloads", [])] != [0, 64]
    ):
        raise ValueError("fixed64 same-seed selected-action afterstate Q fit required")
    if (
        proof_registration.get("schema") != "NNUE-own-afterstate-training-orchestration-v1"
        or proof_registration.get("mode") != "proof"
        or proof_registration.get("seed") != seed
        or proof.get("registration_sha256") != record["proof_registration"]["sha256"]
        or fit_registration.get("schema") != "NNUE-own-afterstate-training-orchestration-v1"
        or fit_registration.get("mode") != "fresh-fit"
        or fit_registration.get("seed") != seed
        or fit.get("registration_sha256") != record["fit_registration"]["sha256"]
    ):
        raise ValueError("separate ROOT registrations for proof and fit required")
    if (
        proof.get("first") >= proof.get("finished")
        or proof.get("finished") > proof.get("deadline")
        or fit.get("first") >= fit.get("finished")
        or fit.get("finished") > fit.get("deadline")
    ):
        raise ValueError("phases must complete inside their original clocks")
    for payload in fit["native_payloads"]:
        pin(payload)
    packet = __import__("torch").load(candidate_path, map_location="cpu", weights_only=False)
    if (
        packet.keys() != {"schema", "contract", "model"}
        or packet.get("schema") != MODEL_SCHEMA
        or packet.get("contract") != contract
    ):
        raise ValueError(
            "candidate must be exact final selected-action afterstate Q trained model packet"
        )
    _validate_model_weights(packet["model"])
    torch = __import__("torch")
    final_native = torch.load(
        pin(fit["native_payloads"][-1]), map_location="cpu", weights_only=False
    )
    if (
        final_native.get("schema") != NATIVE_SCHEMA
        or final_native.get("step") != 64
        or final_native.get("contract") != contract
        or not _bitwise_equal(final_native.get("model"), packet["model"])
    ):
        raise ValueError("candidate weights must equal exact final 64-step native state")
    for field in ("source_sha256", "execution_helpers_sha256", "inference_source_sha256"):
        for path, expected in contract.get(field, {}).items():
            if sha(path) != expected:
                raise ValueError("candidate source/inference closure changed")
    return {
        "seed": seed,
        "candidate": packet,
        "contract": contract,
        "target_provenance_sha256": sha(provenance_path),
    }


def _load_module(ref: dict, name: str):
    path = pin(ref)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pinned module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_afterstate_value(record: dict, protocol: dict):
    """Return the same nonterminal evaluator interface used by the arena.

    `protocol` supplies the already-qualified original `admission.py`, NNUE
    helper refs, C evaluator, and prior helper. No search code is changed.
    """
    admitted = admit_afterstate_child(record)
    admission = _load_module(protocol["admission"], "afterstate_original_arena_admission")
    model_module, native_module, evaluator, compiled = admission.import_nnue(protocol)
    if model_module.MODEL_SCHEMA != MODEL_SCHEMA:
        raise ValueError("original arena model architecture differs")
    prior = _load_module(protocol["prior_helper"], "afterstate_original_authoritative_prior")
    authoritative = evaluator.AuthoritativePrior(prior, prior.ClassicalValue())
    return evaluator.Evaluator(
        admitted["candidate"]["model"], prior=authoritative, compiled=compiled
    ).nonterminal
