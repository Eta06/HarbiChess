"""Shared read-only verification of model, fit, and full-native arena inputs."""

import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_fit_manifest(protocol):
    path = Path(protocol["fit_provenance_path"])
    if sha(path) != protocol["fit_provenance_sha256"]:
        raise ValueError("portable fit provenance SHA differs")
    manifest = json.loads(path.read_text())
    if (
        manifest.get("schema") != "fresh-own-learning-arena-fit-provenance-v1"
        or manifest.get("status") != "PASS-fit-provenance-only-not-strength"
        or manifest.get("source_commit") != protocol["source_commit"]
        or manifest.get("seeds") != protocol["match_seeds"]
    ):
        raise ValueError("fit provenance schema/source/seed differs")
    return path, manifest


def verify_fixed_arena_protocol(protocol):
    tasks = [
        ["e8", "SF512"],
        ["mc", "e8"],
        ["mc", "SF512"],
        ["sc", "e8"],
        ["sc", "SF512"],
        ["full", "e8"],
        ["full", "SF512"],
    ]
    fixed = {
        "schema": "fresh-own-learning-224-paired-arena-protocol-v1",
        "source_commit": "6fcc8b476d25495d1c9c413e55b2c7ba4794013e",
        "match_seeds": [20262805, 20262806],
        "roles": ["mc", "sc", "full"],
        "tasks": tasks,
        "opening_pairs": 8,
        "games_per_tournament": 16,
        "total_games": 224,
        "search_nodes": 512,
        "quiescence_plies": 2,
        "max_depth": 8,
        "max_plies": 400,
        "stockfish_nodes": 512,
        "stockfish_threads": 1,
        "stockfish_hash_mib": 16,
        "threads": 1,
        "max_arena_workers": 1,
        "profile_seconds": 600,
    }
    for name, expected in fixed.items():
        if protocol.get(name) != expected:
            raise ValueError(f"arena protocol fixed field differs: {name}")
    screen = protocol.get("screen", {})
    if screen != {
        "evaluate_both_seeds_and_all_three_arms": True,
        "direct_same_search_e8_score_strictly_above": 0.6,
        "paired_sf_gain_over_e8_strictly_above": 0.1,
        "final_sf_score_at_least": 0.25,
        "caps_at_most": 0.05,
    }:
        raise ValueError("frozen arena gates changed")
    if (
        protocol.get("book_sha256")
        != "9c36b5972c9676d19c358ca7e598b9f7f5d110b1947bed838cd660f644d845da"
    ):
        raise ValueError("registered development book differs")
    if (
        protocol.get("stockfish_sha256")
        != "0f83d24cc46d2c66c60f16001af5444873bc112b7d028594513426894c12da19"
    ):
        raise ValueError("registered Stockfish binary differs")
    return True


def verify_fit_inputs(protocol, manifest):
    for item in manifest["fit_cohorts"].values():
        path = Path(item["path"])
        if sha(path) != item["sha256"]:
            raise ValueError("registered completed-fit cohort bytes changed")
        cohort = json.loads(path.read_text())
        if (
            cohort.get("status")
            != "PASS-all-registered-fits-and-native-audits-not-strength"
        ):
            raise ValueError("completed-fit cohort no longer passes its native audit")
    for seed in protocol["match_seeds"]:
        models = manifest["fits"][str(seed)]
        if set(models) != {"mc", "sc", "full"}:
            raise ValueError("exact MC/SC/FULL model roles required")
        for role, record in models.items():
            candidate = Path(record["path"])
            if (
                sha(candidate) != record["sha256"]
                or candidate.stat().st_size != record["bytes"]
            ):
                raise ValueError(f"{seed}/{role} candidate file changed")
            result_path = Path(record["fit_receipt_path"])
            if sha(result_path) != record["fit_receipt_sha256"]:
                raise ValueError(f"{seed}/{role} completed fit receipt changed")
            result = json.loads(result_path.read_text())
            if (
                result.get("contract") != record["fit_contract"]
                or result.get("status") != "completed-fit-not-strength"
                or result.get("strength_success_claimed") is True
                or result.get("teacher_labels", result.get("new_teacher_labels"))
                is not False
                or result.get("accepted_updates")
                != record["native"]["final"]["accepted_updates"]
            ):
                raise ValueError(
                    f"{seed}/{role} fit contract/status/teacher/update differs"
                )
            native_schemas = set()
            for label in ("initial", "final"):
                ref = record["native"][label]
                directory = Path(ref["checkpoint_path"])
                manifest_path = directory / "checkpoint.json"
                payload_path = directory / "training.pt"
                if (
                    sha(manifest_path) != ref["manifest_sha256"]
                    or sha(payload_path) != ref["training_pt_sha256"]
                ):
                    raise ValueError(
                        f"{seed}/{role} {label} native artifact bytes changed"
                    )
                native = json.loads(manifest_path.read_text())
                native_schemas.add(native.get("schema"))
                if (
                    native.get("accepted") != ref["accepted_updates"]
                    or native.get("contract") != record["fit_contract"]
                ):
                    raise ValueError(
                        f"{seed}/{role} {label} native contract/cursor differs"
                    )
            if len(native_schemas) != 1:
                raise ValueError(f"{seed}/{role} initial/final native schema changed")
            if record["fit_contract"].get("source_commit") != protocol["source_commit"]:
                raise ValueError(f"{seed}/{role} fit source commit differs")
            if record["fit_contract"].get("seed") != seed:
                raise ValueError(f"{seed}/{role} fit seed differs")
    e8 = manifest["e8"]
    if sha(e8["path"]) != e8["sha256"]:
        raise ValueError("frozen E8 input bytes changed")
    for seed in protocol["match_seeds"]:
        for role in ("mc", "sc", "full"):
            record = manifest["fits"][str(seed)][role]
            if (
                record["fit_contract"].get(
                    "actor_model_sha256",
                    record["fit_contract"].get(
                        "e8_sha256", record["fit_contract"].get("initial_e8_sha256")
                    ),
                )
                != e8["sha256"]
            ):
                raise ValueError(f"{seed}/{role} frozen E8 baseline binding differs")
    return True


def verify_profile_receipt(
    protocol, receipt, protocol_sha, fit_sha, helper_sha, manifest
):
    if (
        receipt.get("status") != "PASS-qualification-not-strength"
        or receipt.get("protocol_sha256") != protocol_sha
        or receipt.get("fit_provenance_sha256") != fit_sha
        or receipt.get("helper_sha256") != helper_sha
        or receipt.get("original_first_epoch") != protocol["profile_first_epoch"]
        or receipt.get("original_deadline_epoch") != protocol["profile_deadline_epoch"]
        or receipt.get("finished_epoch", float("inf"))
        > protocol["profile_deadline_epoch"]
        or len(receipt.get("rows", [])) != 24
    ):
        raise ValueError("profile receipt binding, row count, or clock differs")
    seen = set()
    for row in receipt["rows"]:
        key = (row.get("seed"), row.get("role"), row.get("opening"))
        if key in seen or row.get("role") not in protocol["roles"]:
            raise ValueError("profile search inventory has duplicate/unknown row")
        expected = manifest["fits"][str(row["seed"])][row["role"]]["sha256"]
        if row.get("model_sha256") != expected or not 0 <= row.get("opening", -1) < 4:
            raise ValueError("profile row model or opening differs")
        seen.add(key)
    expected_rows = {
        (seed, role, opening)
        for seed in protocol["match_seeds"]
        for role in protocol["roles"]
        for opening in range(4)
    }
    if seen != expected_rows:
        raise ValueError("profile rows do not cover the frozen 24-search design")
    return True


def model_record(protocol, manifest, seed, role):
    if role == "e8":
        return manifest["e8"]
    if role not in {"mc", "sc", "full"}:
        raise ValueError("unknown arena model role")
    return manifest["fits"][str(seed)][role]
