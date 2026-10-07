"""Strict, read-only forensic view of the v2 receipt path-shadow defect."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

SCHEMA = "human-randomstarts-own-forensic-receipt-view-v3"
REG_SCHEMA = "human-randomstarts-own-collection-registration-v2"
RAW_SCHEMA = "human-randomstarts-own-collection-receipt-v2"
MAX_JSON_BYTES = 8 * 2**20


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(262144), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> object:
    path = Path(path)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("forensic JSON size limit")
    return json.loads(path.read_bytes())


def ref(path: Path) -> dict:
    original = Path(path)
    if original.is_symlink():
        raise ValueError("forensic input must not be a symlink")
    path = original.resolve(strict=True)
    if not path.is_file():
        raise ValueError("forensic input must be a pinned regular file")
    return {"path": str(path), "sha256": sha(path)}


def _is_generation_helper_loop(node: ast.For) -> bool:
    target = node.target
    if not isinstance(target, ast.Tuple) or [getattr(x, "id", None) for x in target.elts] != [
        "path",
        "digest",
    ]:
        return False
    call = node.iter
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
        return False
    if call.func.attr != "items" or call.args:
        return False
    subscript = call.func.value
    return (
        isinstance(subscript, ast.Subscript)
        and isinstance(subscript.value, ast.Name)
        and subscript.value.id == "reg"
        and isinstance(subscript.slice, ast.Constant)
        and subscript.slice.value == "generation_helper_sha256"
    )


def _receipt_hash_uses_shadowed_path(node: ast.Assign) -> bool:
    if not any(isinstance(target, ast.Name) and target.id == "receipt" for target in node.targets):
        return False
    if not isinstance(node.value, ast.Dict):
        return False
    for key, value in zip(node.value.keys, node.value.values, strict=True):
        if not isinstance(key, ast.Constant) or key.value != "registration_sha256":
            continue
        return (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "sha"
            and len(value.args) == 1
            and isinstance(value.args[0], ast.Name)
            and value.args[0].id == "path"
        )
    return False


def source_has_path_shadow_bug(source_path: Path) -> bool:
    tree = ast.parse(Path(source_path).read_text())
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef) or fn.name != "execute":
            continue
        loops = [
            x for x in ast.walk(fn) if isinstance(x, ast.For) and _is_generation_helper_loop(x)
        ]
        assigns = [x for x in ast.walk(fn) if isinstance(x, ast.Assign)]
        for loop in loops:
            if any(
                loop.lineno < assignment.lineno and _receipt_hash_uses_shadowed_path(assignment)
                for assignment in assigns
            ):
                return True
    return False


def root_launch_evidence(registration_path: Path, reg: dict) -> dict:
    seed = reg["seed"]
    root_dir = Path(registration_path).parent.parent
    controller_dir = root_dir / f"human-random-collect-{seed}-controller"
    controller_path = controller_dir / "controller-registration.json"
    owner_path = controller_dir / "owner-process.json"
    seal_path = Path(registration_path).parent / "collection-build-seal.json"
    controller = read(controller_path)
    owner = read(owner_path)
    seal = read(seal_path)
    if (
        controller.get("schema") != "UFUK-DEVAM-bounded-cpu-phase-v1"
        or not any(
            marker in controller.get("phase", "")
            for marker in (f"seed{seed % 100:02d}", f"seeds{seed % 100:02d}")
        )
        or controller.get("cpu_core") != reg.get("cpu_core")
        or controller.get("checkout") != reg.get("core_repo")
        or controller.get("operator_end_epoch") != reg.get("operator_end_epoch")
        or owner.get("registration") != str(controller_path)
        or owner.get("first_epoch") != controller.get("first_epoch")
        or owner.get("deadline_epoch") != controller.get("deadline_epoch")
        or owner.get("operator_end_epoch") != controller.get("operator_end_epoch")
    ):
        raise ValueError("original ROOT command/controller/owner binding")
    expected_seal_fields = {
        "generation": reg["generation"],
        "cpu_core": reg["cpu_core"],
        "first": reg["original_first_epoch"],
        "deadline": reg["original_deadline_epoch"],
        "operator_end_epoch": reg["operator_end_epoch"],
        "parent_admission_result": reg["parent_admission_result"],
        "parent_admission_seal": reg["parent_admission_seal"],
        "parent_helpers": reg["parent_helpers"],
        "protected_aliases": reg["protected_aliases"],
        "search_helper": reg["search_helper"],
        "procedural_bank_receipt": reg["procedural_bank_receipt"],
        "root_pool_output": reg["root_pool"]["path"],
    }
    if any(seal.get(key) != value for key, value in expected_seal_fields.items()):
        raise ValueError("pre-actor collection seal differs from frozen registration")
    if (
        seal.get("schema") != "human-randomstarts-own-collection-build-seal-v2"
        or seal.get("status") != "registered"
    ):
        raise ValueError("original pre-actor collection build seal required")
    input_pins = controller.get("input_pins")
    if not isinstance(input_pins, dict) or not input_pins:
        raise ValueError("original ROOT command input pins required")
    input_refs = []
    for pin_path, digest in sorted(input_pins.items()):
        item = ref(Path(pin_path))
        if item["sha256"] != digest:
            raise ValueError("original ROOT command input changed")
        input_refs.append(item)
    return {
        "build_seal": ref(seal_path),
        "controller_registration": ref(controller_path),
        "controller_owner_process": ref(owner_path),
        "controller_code_sha256": controller["controller_sha256"],
        "controller_phase": controller["phase"],
        "controller_original_first_epoch": controller["first_epoch"],
        "controller_original_deadline_epoch": controller["deadline_epoch"],
        "controller_input_pins": input_refs,
    }


def validate_static_receipt_bindings(reg: dict, raw: dict, events_path: Path) -> None:
    expected = {
        "parent_candidate_sha256": reg["parent_candidate"]["sha256"],
        "parent_contract_sha256": reg["parent_candidate"]["contract_sha256"],
        "parent_helpers": reg["parent_helpers"],
        "parent_admission_result": reg["parent_admission_result"],
        "parent_admission_seal": reg["parent_admission_seal"],
        "generation": reg["generation"],
        "search_helper_sha256": reg["search_helper"]["sha256"],
        "root_pool_sha256": reg["root_pool"]["sha256"],
        "protected_aliases_sha256": reg["protected_aliases"]["sha256"],
        "procedural_selection_path": reg["root_pool"]["selection_path"],
        "procedural_selection_sha256": reg["root_pool"]["selection_sha256"],
        "procedural_bank_receipt": reg["procedural_bank_receipt"],
        "source_selection_sha256": reg["root_pool"]["selection_sha256"],
        "search": reg["search"],
        "original_first_epoch": reg["original_first_epoch"],
        "original_deadline_epoch": reg["original_deadline_epoch"],
        "operator_end_epoch": reg["operator_end_epoch"],
        "producer_source_sha256": reg["producer_source_sha256"],
        "generation_helper_sha256": reg["generation_helper_sha256"],
        "target_source": "own-search; own WDL only for completed episodes",
        "exposure_definition": (
            "all path boards plus actual static evaluator inputs; "
            "per-row evaluator aliases in sorted-unique sidecars"
        ),
    }
    if any(raw.get(key) != value for key, value in expected.items()):
        raise ValueError("raw receipt registered metadata fields differ")
    events_path = Path(events_path)
    if (
        raw.get("events_bytes") != events_path.stat().st_size
        or raw.get("events_sha256") != sha(events_path)
        or raw.get("train_rows") != 1024
        or raw.get("teacher_labels_used") is not False
        or not reg["original_first_epoch"]
        <= raw.get("finished_epoch", 0)
        <= reg["original_deadline_epoch"]
        <= reg["operator_end_epoch"]
    ):
        raise ValueError("raw receipt completion/event/static inputs differ")
    if (
        type(raw.get("all_actor_rows")) is not int
        or raw["all_actor_rows"] < 1024
        or type(raw.get("starts_considered")) is not int
        or raw["starts_considered"] <= 0
        or raw.get("unique_exposure_aliases") != raw.get("exposed_board_alias_count")
    ):
        raise ValueError("raw receipt row/exposure counters differ")


def derive_view(registration_path: Path, raw_receipt_path: Path, producer_directory: Path) -> dict:
    registration_path = Path(registration_path).resolve(strict=True)
    raw_receipt_path = Path(raw_receipt_path).resolve(strict=True)
    producer_directory = Path(producer_directory).resolve(strict=True)
    reg = read(registration_path)
    raw = read(raw_receipt_path)
    if reg.get("schema") != REG_SCHEMA or reg.get("status") != "registered":
        raise ValueError("original immutable v2 registration required")
    if raw.get("schema") != RAW_SCHEMA or raw.get("status") != "PASS-exact-row-budget":
        raise ValueError("original raw v2 receipt required")
    if reg.get("seed") not in (20262905, 20262906) or raw.get("seed") != reg.get("seed"):
        raise ValueError("fixed original seed binding")
    if raw.get("generation") != reg.get("generation"):
        raise ValueError("original generation binding")
    if raw.get("train_rows") != 1024 or raw.get("teacher_labels_used") is not False:
        raise ValueError("original own-search row/teacher provenance")
    validate_static_receipt_bindings(reg, raw, Path(reg["output_path"]) / "events.jsonl")

    registration_ref = ref(registration_path)
    raw_receipt_ref = ref(raw_receipt_path)
    source_ref = ref(producer_directory / "run_collection.py")
    expected_source_sha = reg.get("producer_source_sha256", {}).get("run_collection.py")
    if source_ref["sha256"] != expected_source_sha:
        raise ValueError("original registered generation source changed")
    if not source_has_path_shadow_bug(Path(source_ref["path"])):
        raise ValueError("registered source does not exhibit the exact path shadow")

    helpers = reg.get("generation_helper_sha256")
    if not isinstance(helpers, dict) or len(helpers) != 1:
        raise ValueError("expected single registered generation helper for shadow diagnosis")
    helper_path, helper_sha = next(iter(helpers.items()))
    helper_ref = ref(Path(helper_path))
    if helper_ref["sha256"] != helper_sha or raw.get("registration_sha256") != helper_sha:
        raise ValueError("raw incorrect field must equal the last verified helper SHA")
    verified_registration_sha = registration_ref["sha256"]
    if raw.get("registration_sha256") == verified_registration_sha:
        raise ValueError("no receipt defect to derive; use ordinary v2 path")

    raw_without_wrong_field = dict(raw)
    del raw_without_wrong_field["registration_sha256"]
    launch_evidence = root_launch_evidence(registration_path, reg)
    return {
        "schema": SCHEMA,
        "status": "PASS-source-shadow-only-audit-required",
        "seed": reg["seed"],
        "generation": reg["generation"],
        "original_registration": registration_ref,
        "raw_receipt": raw_receipt_ref,
        "root_launch_evidence": launch_evidence,
        "raw_receipt_registration_sha256": raw["registration_sha256"],
        "verified_original_registration_sha256": verified_registration_sha,
        "registration_binding_derivation": {
            "schema": "run-collection-loop-variable-shadow-v1",
            "source": source_ref,
            "shadowed_loop_variable": "path",
            "loop_target": ["path", "digest"],
            "loop_source": "reg['generation_helper_sha256'].items()",
            "receipt_expression": "registration_sha256 = sha(path)",
            "last_helper": helper_ref,
        },
        "raw_non_binding_fields_sha256": hashlib.sha256(
            canonical(raw_without_wrong_field)
        ).hexdigest(),
        "raw_receipt_bytes_preserved": True,
        "receipt_view_is_not_a_rewritten_v2_receipt": True,
        "all_remaining_fields_require_six_replay_and_converter_validation": True,
    }


def verify_view(
    view_path: Path,
    registration_path: Path,
    raw_receipt_path: Path,
    producer_directory: Path,
) -> tuple[dict, dict, dict]:
    view = read(view_path)
    expected = derive_view(registration_path, raw_receipt_path, producer_directory)
    if view != expected:
        raise ValueError("derived forensic view does not match immutable inputs")
    reg = read(registration_path)
    raw = read(raw_receipt_path)
    normalized = dict(raw)
    normalized["registration_sha256"] = view["verified_original_registration_sha256"]
    return view, reg, normalized


def publish_view(
    registration_path: Path,
    raw_receipt_path: Path,
    producer_directory: Path,
    output: Path,
):
    view = derive_view(registration_path, raw_receipt_path, producer_directory)
    output = Path(output)
    if not output.is_absolute() or not str(output).startswith("/dev/shm/"):
        raise ValueError("forensic view must publish under RAM")
    if output.exists() or output.is_symlink():
        raise ValueError("forensic view is publish-once")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(canonical(view) + b"\n")
        stream.flush()
    return view


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--producer-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            publish_view(args.registration, args.receipt, args.producer_directory, args.output),
            sort_keys=True,
        )
    )
