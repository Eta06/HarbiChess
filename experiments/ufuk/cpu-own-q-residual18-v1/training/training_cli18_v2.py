"""Registered own-search residual proof/fit runner; never selects by strength."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from feature_bundle18 import ResidualEvaluator18, labels_to_groups18
from reanalyze import replay_position
from residual_critic18 import ResidualCritic18
from train_residual18 import ResidualTrainer, audit_six_native_loads, native_digest


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    return sha_bytes(Path(path).read_bytes())


def load_pinned(path, expected, name):
    path = Path(path).resolve()
    if sha(path) != expected:
        raise ValueError(f"pinned helper hash differs: {path.name}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path:
        raise ValueError("imported module origin differs")
    return module


def _digest(text, size=64):
    if not isinstance(text, str) or len(text) != size:
        raise ValueError("registered digest missing")
    int(text, 16)
    return text


def _write_once(path, data):
    path = Path(path)
    if len(data) > 16 * 1024 * 1024:
        raise ValueError("artifact exceeds 16 MiB")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def _contract(reg, labels_sha, dataset_sha, feature_sha):
    return {
        "updates": 64,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
        "clip_norm": 5.0,
        "anchor_weight": 0.1,
        "weight_l2": 1e-4,
        "prior_sha256": reg["prior_sha256"],
        "labels_sha256": labels_sha,
        "training_dataset_sha256": dataset_sha,
        "feature_helper_sha256": feature_sha,
        "source_commit": reg["source_commit"],
        "input_count": 18,
        "hidden_count": 32,
        "training_code_sha256": reg["closure"],
        "helper_source_commit": reg["helper_source_commit"],
        "helper_hashes": reg["helper_hashes"],
        "reanalysis_registration_sha256": reg["reanalysis_registration_sha256"],
        "actor_config_sha256": reg["actor_config_sha256"],
        "actor_journal_sha256": reg["actor_journal_sha256"],
        "inference_closure": reg["inference_closure"],
    }


def _native_wire(native):
    return gzip.compress(canonical(native), mtime=0)


def _load_groups(path, expected_sha):
    raw = Path(path).read_bytes()
    if sha_bytes(raw) != expected_sha:
        raise ValueError("prepared dataset changed")
    return json.loads(gzip.decompress(raw))


def _restore_groups(payload):
    groups = {}
    for game, rows in payload.items():
        groups[game] = [
            {
                "x": row["x"],
                "prior_logit": row["prior_logit"],
                "target": row["target"],
            }
            for row in rows
        ]
    return groups


def resume_native(state, groups_payload, *, seed, contract, guard=lambda: None):
    """Run the exact fresh-process continuation on already verified payloads."""
    model = ResidualCritic18(seed=seed, contract=contract, state=state)
    ResidualTrainer(model, _restore_groups(groups_payload)).advance(8, guard=guard)
    return model.native()


def _json_groups(groups):
    return {
        game: [
            {
                "x": [float(value) for value in row["x"]],
                "prior_logit": float(row["prior_logit"]),
                "target": float(row["target"]),
            }
            for row in rows
        ]
        for game, rows in groups.items()
    }


def execute(registration_path, *, resume_child=False):
    reg_path = Path(registration_path).resolve()
    reg = json.loads(reg_path.read_text())
    if (
        reg.get("schema") != "own-search-residual18-training-registration-v1"
        or reg.get("status") != "registered"
    ):
        raise ValueError("registered training contract required")
    source_repo = Path(reg["source_repo"]).resolve()
    source_commit = _digest(reg["source_commit"], 40)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_repo, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=source_repo, text=True
    )
    if head != source_commit or dirty:
        raise ValueError("clean core source pin differs")
    core = reg["cpu_core"]
    if not isinstance(core, int) or isinstance(core, bool) or core not in os.sched_getaffinity(0):
        raise ValueError("registered CPU core is outside current affinity")
    os.sched_setaffinity(0, {core})

    phase = "proof" if resume_child else reg.get("phase")
    if phase not in {"proof", "fit"}:
        raise ValueError("phase must be proof or fit")
    clock = reg["clocks"][phase]
    first, deadline = float(clock["original_first_epoch"]), float(clock["original_deadline_epoch"])
    expected_window = 600 if phase == "proof" else 1800
    if deadline != first + expected_window or not first <= time.time() < deadline:
        raise TimeoutError("registered phase clock is invalid or expired")

    output = Path(reg["output_dir"]).resolve()
    if not str(output).startswith("/dev/shm/harbichess-own-q-residual18/"):
        raise ValueError("artifacts must stay in the registered RAM work area")
    if output.exists() and not resume_child:
        raise FileExistsError("publish-once output directory already exists")

    closure = reg["closure"]
    own_paths = {
        "critic18": Path(__file__).with_name("residual_critic18.py").resolve(),
        "features18": Path(__file__).with_name("feature_bundle18.py").resolve(),
        "reanalyze": Path(__file__).with_name("reanalyze.py").resolve(),
        "trainer18": Path(__file__).with_name("train_residual18.py").resolve(),
        "runner18": Path(__file__).resolve(),
    }
    if set(closure) != set(own_paths):
        raise ValueError("training closure keyset differs")
    for key, path in own_paths.items():
        if sha(path) != _digest(closure[key]):
            raise ValueError(f"training code SHA differs: {key}")

    helper_dir = Path(reg["helper_dir"]).resolve()
    helper_paths = {
        "value": helper_dir / "value.py",
        "runtime": helper_dir / "runtime.py",
    }
    for key, path in helper_paths.items():
        if sha(path) != _digest(reg["helper_hashes"][key]):
            raise ValueError(f"feature/runtime helper SHA differs: {key}")
    helper_root = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=helper_dir, text=True
        ).strip()
    ).resolve()
    helper_commit = _digest(reg["helper_source_commit"], 40)
    for key, path in helper_paths.items():
        relative = path.relative_to(helper_root).as_posix()
        committed = subprocess.check_output(
            ["git", "show", f"{helper_commit}:{relative}"], cwd=helper_root
        )
        if sha_bytes(committed) != reg["helper_hashes"][key]:
            raise ValueError(f"helper not present in registered helper commit: {key}")
    value = load_pinned(helper_paths["value"], reg["helper_hashes"]["value"], "residual_value")
    runtime = load_pinned(
        helper_paths["runtime"], reg["helper_hashes"]["runtime"], "residual_runtime"
    )

    labels_path = Path(reg["labels_path"]).resolve()
    labels_sha = _digest(reg["labels_sha256"])
    if sha(labels_path) != labels_sha:
        raise ValueError("label archive changed")
    labels = json.loads(gzip.decompress(labels_path.read_bytes()))
    if (
        labels.get("schema") != "own-search-deeper-value-labels-v1"
        or labels.get("registration_sha256") != reg["reanalysis_registration_sha256"]
        or labels.get("source_commit") != source_commit
        or labels.get("closure") != reg["reanalysis_closure"]
        or labels.get("config_sha256") != reg["actor_config_sha256"]
        or labels.get("journal_sha256") != reg["actor_journal_sha256"]
        or labels.get("training_dataset_sha256") != reg["actor_training_dataset_sha256"]
        or labels.get("training_receipt") != reg["actor_training_receipt"]
        or labels.get("seed") != reg["seed"]
        or labels.get("search") != reg["search"]
        or labels.get("frozen_prior_model") != value.model_dict()
    ):
        raise ValueError("labels do not match frozen source/search/prior contract")
    if reg["prior_sha256"] != _digest(reg["prior_sha256"]):
        raise ValueError("invalid prior model digest")
    prior_bytes = canonical(value.model_dict())
    if sha_bytes(prior_bytes) != reg["prior_sha256"]:
        raise ValueError("prior model identity differs")
    groups = labels_to_groups18(
        labels,
        replay_position=replay_position,
        classical_features=value.features,
        prior_weights=value.PRIOR,
        feature_scales=value.FEATURE_SCALES,
        prior_score_scale=value.SCALE,
    )
    if len(groups) < 16 or sum(map(len, groups.values())) != 1024:
        raise ValueError("full 1,024-root unique-trajectory labels are required")
    json_groups = _json_groups(groups)
    dataset_sha = sha_bytes(canonical(json_groups))
    feature_sha = _digest(reg["feature_helper_sha256"])
    expected_feature_sha = sha_bytes(
        canonical(
            {
                "bundle": closure["features18"],
                "classical": reg["helper_hashes"]["value"],
            }
        )
    )
    if feature_sha != expected_feature_sha:
        raise ValueError("combined feature source closure differs from native contract")
    contract = _contract(reg, labels_sha, dataset_sha, feature_sha)

    if phase == "fit":
        proof_path = Path(reg["proof_result_path"]).resolve()
        proof_sha = _digest(reg["proof_result_sha256"])
        if sha(proof_path) != proof_sha:
            raise ValueError("proof receipt changed before fit")
        proof = json.loads(proof_path.read_text())
        if (
            proof.get("schema") != "own-search-residual18-proof-result-v1"
            or proof.get("qualification_passed") is not True
            or proof.get("seed") != reg["seed"]
            or proof.get("source_commit") != source_commit
            or proof.get("helper_source_commit") != helper_commit
            or proof.get("training_closure") != closure
            or proof.get("training_contract_sha256") != sha_bytes(canonical(contract))
            or proof.get("labels_sha256") != labels_sha
            or proof.get("training_dataset_sha256") != dataset_sha
            or proof.get("fresh_process_resume") is not True
            or len(proof.get("strict_native_load_sha256", [])) != 6
        ):
            raise ValueError("fit proof does not qualify this exact source/data/seed")

    def guard():
        runtime.guard(deadline, source_repo, output)

    if resume_child:
        state_path = output / "pause4.native.json.gz"
        groups_path = output / "training-groups.json.gz"
        if sha(state_path) != _digest(reg["pause4_native_sha256"]):
            raise ValueError("pause native changed before child resume")
        cached = _load_groups(groups_path, reg["training_groups_sha256"])
        if sha_bytes(canonical(cached)) != dataset_sha:
            raise ValueError("prepared groups no longer match frozen labels")
        resumed = resume_native(
            json.loads(gzip.decompress(state_path.read_bytes())),
            cached,
            seed=reg["seed"],
            contract=contract,
            guard=guard,
        )
        _write_once(output / "resume8.native.json.gz", _native_wire(resumed))
        print(json.dumps({"resume8_sha256": sha(output / "resume8.native.json.gz")}))
        return

    if phase == "proof":
        output.mkdir(parents=True, exist_ok=False)
        groups_wire = gzip.compress(canonical(json_groups), mtime=0)
        _write_once(output / "training-groups.json.gz", groups_wire)
        groups_sha = sha(output / "training-groups.json.gz")
        states = []
        whole = ResidualCritic18(seed=reg["seed"], contract=contract)
        states.append(whole.native())
        _write_once(output / "whole0.native.json.gz", _native_wire(states[-1]))
        ResidualTrainer(whole, groups).advance(4, guard=guard)
        states.append(whole.native())
        _write_once(output / "whole4.native.json.gz", _native_wire(states[-1]))
        ResidualTrainer(whole, groups).advance(8, guard=guard)
        states.append(whole.native())
        _write_once(output / "whole8.native.json.gz", _native_wire(states[-1]))

        paused = ResidualCritic18(seed=reg["seed"], contract=contract)
        split_zero = paused.native()
        _write_once(output / "split0.native.json.gz", _native_wire(split_zero))
        ResidualTrainer(paused, groups).advance(4, guard=guard)
        pause_wire = _native_wire(paused.native())
        if native_digest(json.loads(gzip.decompress(_native_wire(states[1])))) != native_digest(
            json.loads(gzip.decompress(pause_wire))
        ):
            raise ValueError("whole4 and pause4 differ")
        pause_path = output / "pause4.native.json.gz"
        _write_once(pause_path, pause_wire)
        _write_once(output / "pause4.native.sha256", (sha(pause_path) + "\n").encode())
        # Child verifies the full registration again and starts a genuinely fresh interpreter.
        child_reg = dict(reg)
        child_reg["pause4_native_sha256"] = sha(pause_path)
        child_reg["training_groups_sha256"] = groups_sha
        child_reg_path = output / "resume-registration.json"
        _write_once(child_reg_path, canonical(child_reg) + b"\n")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--registration",
            str(child_reg_path),
            "--resume-child",
        ]
        subprocess.run(command, check=True, timeout=max(1, deadline - time.time()))
        resumed_path = output / "resume8.native.json.gz"
        resumed = json.loads(gzip.decompress(resumed_path.read_bytes()))
        if native_digest(resumed) != native_digest(states[-1]):
            raise ValueError("whole8 and fresh-process pause4/resume8 differ")
        strict_states = [
            states[0],
            states[1],
            states[2],
            split_zero,
            json.loads(gzip.decompress(pause_wire)),
            resumed,
        ]
        if len(audit_six_native_loads(strict_states, contract)) != 6:
            raise ValueError("six strict native loads did not round-trip")
        receipt = {
            "schema": "own-search-residual18-proof-result-v1",
            "registration_sha256": sha(reg_path),
            "source_commit": source_commit,
            "helper_source_commit": helper_commit,
            "training_closure": closure,
            "training_contract_sha256": sha_bytes(canonical(contract)),
            "labels_sha256": labels_sha,
            "training_dataset_sha256": dataset_sha,
            "training_groups_sha256": groups_sha,
            "seed": reg["seed"],
            "original_first_epoch": first,
            "original_deadline_epoch": deadline,
            "whole0_4_8_sha256": [native_digest(state) for state in states],
            "split0_pause4_resume8_sha256": [native_digest(state) for state in strict_states[3:]],
            "resume8_sha256": sha(resumed_path),
            "strict_native_load_sha256": audit_six_native_loads(strict_states, contract),
            "fresh_process_resume": True,
            "qualification_passed": True,
            "actual_labels_used_no_strength_evaluation": True,
        }
        guard()
        _write_once(output / "proof-result.json", canonical(receipt) + b"\n")
        return

    output.mkdir(parents=True, exist_ok=False)
    guard()
    model = ResidualCritic18(seed=reg["seed"], contract=contract)
    _write_once(output / "fresh0.native.json.gz", _native_wire(model.native()))
    ResidualTrainer(model, groups).advance(64, guard=guard)
    final_native = model.native()
    candidate = model.candidate()
    final_native_wire = _native_wire(final_native)
    candidate_wire = gzip.compress(canonical(candidate), mtime=0)
    guard()
    _write_once(output / "final64.native.json.gz", final_native_wire)
    _write_once(output / "candidate.json.gz", candidate_wire)
    # Fresh strict reload of the final candidate and native without further updates.
    final_loaded = ResidualCritic18(seed=reg["seed"], contract=contract, state=final_native)
    if native_digest(final_loaded.native()) != native_digest(final_native):
        raise ValueError("final native strict load changed state")
    candidate_loaded = ResidualCritic18.from_candidate(
        json.loads(gzip.decompress(candidate_wire)), seed=reg["seed"], contract=contract
    )
    evaluator = ResidualEvaluator18(
        model,
        classical_features=value.features,
        prior_weights=value.PRIOR,
        feature_scales=value.FEATURE_SCALES,
        prior_score_scale=value.SCALE,
    )
    if set(candidate_loaded.params) != set(model.params) or any(
        not __import__("numpy").array_equal(candidate_loaded.params[key], model.params[key])
        for key in model.params
    ):
        raise ValueError("candidate inference construction differs")
    receipt = {
        "schema": "own-search-residual18-fit-result-v1",
        "registration_sha256": sha(reg_path),
        "source_commit": source_commit,
        "helper_source_commit": helper_commit,
        "training_closure": closure,
        "training_contract_sha256": sha_bytes(canonical(contract)),
        "labels_sha256": labels_sha,
        "training_dataset_sha256": dataset_sha,
        "seed": reg["seed"],
        "original_first_epoch": first,
        "original_deadline_epoch": deadline,
        "fresh_initializer_step": 0,
        "fixed_final_step": 64,
        "final_native_sha256": sha(output / "final64.native.json.gz"),
        "candidate_sha256": sha(output / "candidate.json.gz"),
        "candidate_selected_by_strength": False,
        "teacher_labels": 0,
    }
    guard()
    _write_once(output / "fit-result.json", canonical(receipt) + b"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--resume-child", action="store_true")
    args = parser.parse_args()
    execute(args.registration, resume_child=args.resume_child)


if __name__ == "__main__":
    main()
