"""Versioned readonly CPU-native validation, never training-clock replacement.

The original contract (including expired training deadline) is preserved. Only
this new independent audit has a prospective resource clock. No optimizer step,
selfplay, teacher labels, network, upload or metrics/strength computation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import resource
import struct
import subprocess
import sys
import time
from pathlib import Path

SCHEMA = "cpu-native-independent-readonly-restore-audit-v2"
SOURCE = "4cae08522ac940746cf9127ca8ce4a6b63a47408"
E8 = "e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03"
END = 1791273600
NATIVES = {
    "cpu-own-outcome-linear-training-native-v2",
    "cpu-own-outcome-positional-training-native-v2",
}
FILES = ("model.safetensors", "training.pt", "checkpoint.json")
SEALED_MANIFESTS = {
    "170f4737816346e57d01f6d6fcceec48aa4e09fd8a364cccd7abcb21a4df4cf5",
    "525ddf05b06bdcc0e77d9ceefce7609ed7442a6568b89cf2c1a100062286c4ee",
    "4fecd24fee160dc7195966b012e4d739e66f6e991bafecaef54a98ca857a6591",
    "c13fb58cf4c6103b25d8a30ae0012a6d3987bfba4baa55f7ba591f2035f11fa5",
}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024**2):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class AuditClock:
    def __init__(self, started, deadline):
        if not all(
            math.isfinite(value) for value in (started, deadline)
        ) or not started <= time.time() < deadline <= min(started + 900, END):
            raise ValueError("separate-prospective-audit-clock-required")
        self.started, self.deadline = started, deadline
        self.memory_budget = None

    def __call__(self):
        if time.time() >= self.deadline:
            raise TimeoutError("original-readonly-audit-resource-clock-expired")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 3 * 1024**3:
            raise RuntimeError("own-readonly-audit-RSS-3GiB-cap")

        if self.memory_budget is not None:
            self.memory_budget.check()


def tensor_bits_equal(a, b):
    import torch

    return (
        isinstance(a, torch.Tensor)
        and isinstance(b, torch.Tensor)
        and a.device.type == b.device.type == "cpu"
        and a.dtype == b.dtype
        and a.shape == b.shape
        and torch.equal(
            a.detach().contiguous().reshape(-1).view(torch.uint8),
            b.detach().contiguous().reshape(-1).view(torch.uint8),
        )
    )


def equal_tree(a, b):
    import torch

    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return tensor_bits_equal(a, b)
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return set(a) == set(b) and all(equal_tree(a[key], b[key]) for key in a)
    if isinstance(a, float):
        return struct.pack("!d", a) == struct.pack("!d", b)
    if isinstance(a, tuple | list):
        return len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def validate_contract(manifest, original_contract):
    contract = manifest["contract"]
    if (
        manifest["schema"] not in NATIVES
        or not equal_tree(contract, original_contract)
        or contract["schema"] != manifest["schema"]
        or contract["source_commit"] != SOURCE
        or contract["initial_e8_sha256"] != E8
        or contract["device"] != "cpu"
        or contract["new_teacher_labels"] is not False
        or contract["new_selfplay_moves_generated"] != 0
        or not math.isfinite(contract["original_deadline_epoch"])
    ):
        raise ValueError("original-native-contract-schema-source-deadline")
    if set(manifest["artifacts"]) != set(FILES[:2]):
        raise ValueError("native-exact-payload-set")
    # No comparison of OLD training deadline with now, and no replacement value.
    return contract


def validate_loaded_state(
    manifest, original_contract, saved, model, portable, optimizer, prepared, guard
):
    """Strict state restoration against exact recomputed original feature output."""
    import numpy as np
    import torch

    guard()
    contract = validate_contract(manifest, original_contract)
    if str(torch.__version__) != contract["torch_version"]:
        raise ValueError("original-cpu-torch-runtime-required")
    x, y, train, validation, receipts, dataset_sha = prepared
    split_sha = hashlib.sha256(canonical({"train": train, "validation": validation})).hexdigest()
    if (
        saved["schema"] != manifest["schema"]
        or not equal_tree(saved["contract"], contract)
        or saved["dataset_sha256"] != dataset_sha
        or saved["split_sha256"] != split_sha
        or saved["dataset_receipts"] != receipts
    ):
        raise ValueError("native-original-dataset-split-receipts-contract")
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if saved["trainable_names"] != names:
        raise ValueError("original-trainable-mask-order")
    count = saved["accepted"]
    if (
        type(count) is not int
        or not 0 <= count == saved["attempted"] == manifest["accepted"] <= contract["max_steps"]
        or len(saved["history"]) != count
    ):
        raise ValueError("native-counters-history-boundary")
    for index, row in enumerate(saved["history"], 1):
        if (
            row["step"] != index
            or not math.isfinite(row["loss"])
            or not math.isfinite(row["preclip_norm"])
            or row["preclip_norm"] < 0
        ):
            raise ValueError("native-history-sequence-finite")
    baseline = model.state_dict()
    if set(saved["model"]) != set(baseline) or set(portable) != set(baseline):
        raise ValueError("complete-model-portable-state-keyset")
    frozen_count, frozen_bytes = 0, 0
    for name, tensor in saved["model"].items():
        guard()
        if (
            not tensor_bits_equal(tensor, portable[name])
            or tensor.dtype != baseline[name].dtype
            or tensor.shape != baseline[name].shape
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError("model-portable-bitshape-finite")
        if name not in names:
            if not tensor_bits_equal(tensor, baseline[name]):
                raise ValueError("original-e8-rebased-frozen-storage-bits")
            frozen_count += 1
            frozen_bytes += tensor.numel() * tensor.element_size()
    # Exact original hyperparameters, parameter ordering and Adam buffer shapes.
    initial_groups = optimizer.state_dict()["param_groups"]
    incoming_groups = saved["optimizer"]["param_groups"]
    if not equal_tree(initial_groups, incoming_groups):
        raise ValueError("original-Adam-parameter-groups-hyperparameters")
    model.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    if len(optimizer.state) != (len(names) if count else 0):
        raise ValueError("Adam-state-completeness")
    for parameter, state in optimizer.state.items():
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("Adam-buffer-keyset")
        step = state["step"]
        if (
            step.device.type != "cpu"
            or step.numel() != 1
            or not torch.isfinite(step).all()
            or step.item() != count
        ):
            raise ValueError("Adam-internal-step-counter")
        for key in ("exp_avg", "exp_avg_sq"):
            buffer = state[key]
            if (
                buffer.device.type != "cpu"
                or buffer.shape != parameter.shape
                or buffer.dtype != parameter.dtype
                or not torch.isfinite(buffer).all()
            ):
                raise ValueError("Adam-buffer-shape-dtype-finite")
        if (state["exp_avg_sq"] < 0).any():
            raise ValueError("Adam-second-moment-negative")
    if not equal_tree(optimizer.state_dict(), saved["optimizer"]):
        raise ValueError("restored-Adam-complete-storage-bits")
    # Validate restoration without changing process-global RNG or drawing samples.
    sampler, python = random.Random(), random.Random()
    sampler.setstate(saved["sampler_rng"])
    python.setstate(saved["python_rng"])
    if not equal_tree(sampler.getstate(), saved["sampler_rng"]) or not equal_tree(
        python.getstate(), saved["python_rng"]
    ):
        raise ValueError("local-Python-sampler-RNG-exact")
    generator = torch.Generator(device="cpu")
    generator.set_state(saved["torch_rng"])
    if not tensor_bits_equal(generator.get_state(), saved["torch_rng"]):
        raise ValueError("local-TorchCPU-RNG-exact")
    rng = saved["numpy_rng"]
    numpy = np.random.RandomState()
    keys = rng["keys"]
    if (
        rng["kind"] != "MT19937"
        or keys.dtype != torch.int64
        or keys.shape != (624,)
        or keys.device.type != "cpu"
        or (keys < 0).any()
        or (keys > 2**32 - 1).any()
        or rng["has_gauss"] not in (0, 1)
        or not 0 <= rng["position"] <= 624
        or not math.isfinite(rng["cached_gauss"])
    ):
        raise ValueError("NumPy-native-state-shape-kind-boundary")
    np_state = (
        rng["kind"],
        keys.numpy().astype(np.uint32),
        rng["position"],
        rng["has_gauss"],
        rng["cached_gauss"],
    )
    numpy.set_state(np_state)
    restored = numpy.get_state()
    if (
        restored[0] != np_state[0]
        or not np.array_equal(restored[1], np_state[1])
        or not equal_tree(restored[2:], np_state[2:])
    ):
        raise ValueError("local-NumPy-RNG-exact")
    if not train or not validation or x.shape[0] != y.shape[0]:
        raise ValueError("nonempty-source-disjoint-prepared-data")
    guard()
    return {
        "schema": SCHEMA,
        "status": "strict-readonly-native-state-pass",
        "original_native_schema": manifest["schema"],
        "source_commit": SOURCE,
        "original_training_contract": contract,
        "original_training_deadline_expired_at_audit": contract["original_deadline_epoch"]
        <= time.time(),
        "accepted_updates": count,
        "dataset_sha256": dataset_sha,
        "split_sha256": split_sha,
        "dataset_receipts": receipts,
        "frozen_tensor_count": frozen_count,
        "frozen_storage_bytes": frozen_bytes,
        "model_portable_all_storage_bits_equal": True,
        "Adam_all_storage_bits_restored": True,
        "all_CPU_RNG_restorable_without_draws": True,
        "optimizer_steps_or_generation_executed": 0,
        "online_actor_resume_claimed": False,
        "original_training_deadline_replaced": False,
    }


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_sealed_files(native, sealed_manifest, sealed_manifest_sha256):
    if (
        sealed_manifest_sha256 not in SEALED_MANIFESTS
        or sha(sealed_manifest) != sealed_manifest_sha256
    ):
        raise ValueError("externally-frozen-native-manifest-SHA-required")
    sealed = json.loads(sealed_manifest.read_text())
    if (
        sealed["schema"] != "cpu-exact-native-delta-chunks-v1"
        or set(sealed["files"]) != set(FILES)
        or sealed["parent_sha256"] != E8
    ):
        raise ValueError("sealed-original-native-three-file-manifest")
    for name, item in sealed["files"].items():
        path = native / name
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != item["bytes"]
            or sha(path) != item["sha256"]
        ):
            raise ValueError("externally-sealed-native-original-bytes-SHA-size")
    return sealed


def audit(args):
    sys.dont_write_bytecode = True
    clock = AuditClock(args.audit_started_epoch, args.audit_deadline_epoch)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    environment = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}

    def git(*command):
        return subprocess.check_output(
            ["git", *command], cwd=args.source_repo, env=environment, text=True
        ).strip()

    if git("rev-parse", "HEAD") != SOURCE or git("status", "--porcelain"):
        raise ValueError("original-source-clean-pin-required")
    sys.path.insert(0, str(args.source_repo / "src"))
    import torch

    torch.set_num_threads(1)
    from harbichess.backends import torch_network
    from harbichess.training import cgroup_budget

    if (
        not Path(cgroup_budget.__file__)
        .resolve()
        .is_relative_to((args.source_repo / "src").resolve())
    ):
        raise ValueError("original-cgroup-budget-source-module-origin")
    clock.memory_budget = cgroup_budget.CgroupMemoryBudget(15 * 1024**3)
    memory_initial = clock.memory_budget.check().copy()

    if (
        not Path(torch_network.__file__)
        .resolve()
        .is_relative_to((args.source_repo / "src").resolve())
    ):
        raise ValueError("original-model-source-module-origin")
    check_sealed_files(args.native, args.sealed_manifest, args.sealed_manifest_sha256)
    manifest = json.loads((args.native / "checkpoint.json").read_text())
    contract = validate_contract(manifest, json.loads(args.original_contract.read_text()))
    watched = [
        *(args.native / name for name in FILES),
        args.original_contract,
        args.weights,
        args.protocol,
        args.helper_dir / "train.py",
        args.helper_dir / "features.py",
        *args.journal,
        args.sealed_manifest,
    ]
    before = {str(path): sha(path) for path in watched}
    for name in FILES[:2]:
        if before[str(args.native / name)] != manifest["artifacts"][name]:
            raise ValueError("original-native-file-SHA")
    for path, expected in [
        (args.weights, E8),
        (args.protocol, contract["protocol_sha256"]),
        (args.helper_dir / "train.py", contract["helper_sha256"]),
        (args.helper_dir / "features.py", contract["feature_helper_sha256"]),
    ]:
        if before[str(path)] != expected:
            raise ValueError("original-input-helper-SHA")
    journal_sha = [before[str(path)] for path in args.journal]
    protocol = json.loads(args.protocol.read_text())
    expected = [
        row["sha256"]
        for row in protocol["journals"]
        if row["source_seed"] == protocol["pairing"][str(contract["seed"])]
    ]
    if (
        journal_sha != contract["journal_sha256"]
        or len(set(journal_sha)) != len(journal_sha)
        or journal_sha != expected
        or protocol["source_commit"] != SOURCE
        or protocol["initial_e8_sha256"] != E8
        or contract["max_steps"] != protocol["steps"]
    ):
        raise ValueError("original-protocol-journal-seed-pairing")
    previous_features = sys.modules.get("features")
    features = import_file(args.helper_dir / "features.py", "features")
    sys.modules["features"] = features
    try:
        helper = import_file(args.helper_dir / "train.py", "_pinned_original_train_definitions")
    finally:
        if previous_features is None:
            del sys.modules["features"]
        else:
            sys.modules["features"] = previous_features
    clock()
    prepared = features.prepare(
        args.journal, clock
    )  # Full original legal replay; only explicit audit mode.
    saved = torch.load(args.native / "training.pt", map_location="cpu", weights_only=True)
    with torch.random.fork_rng(devices=[]):
        portable = torch_network.load_weights(args.native / "model.safetensors").state_dict()
        model = torch_network.load_weights(args.weights).train()
        rebased = helper.rebase(model)
        if rebased is not None:
            model = rebased
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=contract["learning_rate"],
            weight_decay=contract["weight_decay"],
            foreach=False,
        )
        result = validate_loaded_state(
            manifest, contract, saved, model, portable, optimizer, prepared, clock
        )
    after = {str(path): sha(path) for path in watched}
    if before != after:
        raise ValueError("original-sealed-files-changed-during-audit")
    result.update(
        {
            "separate_audit_started_epoch": clock.started,
            "separate_audit_deadline_epoch": clock.deadline,
            "finished_epoch": time.time(),
            "resource_version_reason": (
                "v1 real original journal JSON exceeded 1GiB RSS; no native error"
            ),
            "process_RSS_cap_bytes": 3 * 1024**3,
            "process_peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "aggregate_active_memory_cap_bytes": 15 * 1024**3,
            "cgroup_budget_helper_sha256": sha(Path(cgroup_budget.__file__)),
            "aggregate_memory_initial": memory_initial,
            "aggregate_memory_final": clock.memory_budget.check(),
            "original_files_SHA_unchanged": before,
            "audit_helper_sha256": sha(Path(__file__)),
            "externally_sealed_manifest_sha256": args.sealed_manifest_sha256,
        }
    )
    clock()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in (
        "native",
        "source-repo",
        "original-contract",
        "weights",
        "protocol",
        "helper-dir",
        "sealed-manifest",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--journal", type=Path, action="append", required=True)
    parser.add_argument("--sealed-manifest-sha256", required=True)
    parser.add_argument("--audit-started-epoch", type=float, required=True)
    parser.add_argument("--audit-deadline-epoch", type=float, required=True)
    print(json.dumps(audit(parser.parse_args()), sort_keys=True))
