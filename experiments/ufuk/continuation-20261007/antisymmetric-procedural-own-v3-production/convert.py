"""Reexecute SHA-bound forensic converter; derive paired features at IDENTICAL roots."""

import argparse
import json
from pathlib import Path

from adapter import derive_rows
from support import guard, module, namespace, pinned, pins_tree, publish, read, sha

DATA_SCHEMA = "own-same-board-antisymmetric-forensic-data-v3"


def derive(seal, check=lambda: None):
    if seal["schema"] != "antisymmetric-forensic-conversion-seal-v3":
        raise ValueError("new explicit representation bridge")
    pins_tree(seal)
    if [x["sha256"] for x in seal["forensic_inventories"]] != [
        "bc275cf5bc69a66ca9875893c30c097c17b2c0fea4e29a7b6598525950a89fd3"
    ]:
        raise ValueError("exact ROOT-reviewed completed forensic v4 source inventory")
    directory = Path(seal["forensic_directory"]).resolve()
    common_seal = read(seal["common_conversion_seal"])
    pins_tree(common_seal)
    if common_seal["schema"] != "human-randomstarts-own1024-forensic-dataset-conversion-seal-v3":
        raise ValueError("forensic v3 exact original source view")
    with namespace(directory, seal["forensic_inventories"]) as files:
        converter_ref = seal["forensic_converter"]
        if converter_ref["path"] != str(directory / "convert_forensic_v3.py") or converter_ref[
            "sha256"
        ] != files.get("convert_forensic_v3.py"):
            raise ValueError("converter must belong to qualified complete inventory")
        converter = module(converter_ref, "antisym_common_forensic_converter")
        # ROOT already executed the full alias/episode/source replay. Bind that
        # ORIGINAL result, never redo millions of aliases for a representation arm.
        common_result = read(seal["common_result"])
        raw_data = pinned(seal["common_dataset"]).read_bytes()
        raw_provenance = pinned(seal["common_provenance"]).read_bytes()
        if (
            common_result["status"] != "PASS-own1024-fullhistory-trace-conversion-not-strength"
            or common_result["dataset_sha256"] != seal["common_dataset"]["sha256"]
            or common_result["provenance_sha256"] != seal["common_provenance"]["sha256"]
            or common_result["first"] != common_seal["first"]
            or common_result["deadline"] != common_seal["deadline"]
            or not common_result["first"] < common_result["finished"] <= common_result["deadline"]
        ):
            raise ValueError("actual completed original common conversion result/clock")
        data = json.loads(raw_data)
        provenance = json.loads(raw_provenance)
        if (
            data["schema"] != "own-kingbucket-sparse-training-data-forensic-v3"
            or len(data["rows"]) != 1024
        ):
            raise ValueError("exact1024 common own-Q rows")
        if (
            provenance["inputs"] != common_seal
            or provenance["teacher_labels_used"] is not False
            or provenance["dataset_sha256"] != seal["common_dataset"]["sha256"]
        ):
            raise ValueError("unchanged checked common-data provenance")
        # Validate metadata of both actual independent six-replay results and
        # hash all original exposure chunks without interpreting their millions of entries.
        from forensic_audit_set import validate

        validate(read(common_seal["six_audit_set"]))
        receipt = read(common_seal["receipt"])
        original_registration = read(common_seal["registration"])
        for name, digest in original_registration["producer_source_sha256"].items():
            pinned(dict(path=str(Path(common_seal["producer_directory"]) / name), sha256=digest))
        for path, digest in original_registration["generation_helper_sha256"].items():
            pinned(dict(path=path, sha256=digest))
        from parent_bridge import validate_admission_result

        _, original_parent_contract = validate_admission_result(
            original_registration["parent_admission_seal"],
            original_registration["parent_admission_result"],
        )
        if (
            original_parent_contract["phase"] != "human-prior-zero-residual-init-v1"
            or original_registration["generation"] != 1
        ):
            raise ValueError(
                "first paired-family arm uses actual original humanzero g0 common data"
            )
        events = [
            json.loads(line) for line in pinned(common_seal["events"]).read_bytes().splitlines()
        ]
        rows = {}
        for event in events:
            if event["type"] == "search_row":
                row = event["row"]
                key = row["root_id"] + ":" + str(row["local_ply"])
                if key in rows:
                    raise ValueError("duplicate raw own row")
                rows[key] = row
        ids = receipt["training_row_ids"]
        if len(ids) != 1024 or len(set(ids)) != 1024:
            raise ValueError("original exact unique row order")
        boards = []
        features = module(common_seal["features"], "antisym_original_features")
        prior = module(common_seal["prior"], "antisym_original_prior")
        for rid, original_row in zip(ids, data["rows"], strict=True):
            check()
            if converter.verify_row(rows[rid], features, prior) != original_row:
                raise ValueError("actual1024 raw mover/prior/Q/features exact common-row alignment")
            boards.append(converter.board_at(rows[rid]))
        paired = derive_rows(
            data["rows"],
            boards,
            dict(
                common_seal=seal["common_conversion_seal"],
                common_dataset=seal["common_dataset"],
                common_provenance=seal["common_provenance"],
                common_result=seal["common_result"],
            ),
        )
        for row, rid, board in zip(paired["rows"], ids, boards, strict=True):
            row["source_row_id"] = rid
            row["root_fen"] = board.root().fen()
            row["history_uci"] = [m.uci() for m in board.move_stack]
    paired.update(schema=DATA_SCHEMA, phase="antisymmetric-procedural-own-learning-v3")
    result = dict(
        schema="antisymmetric-forensic-provenance-v3",
        seal=seal,
        common_provenance=provenance,
        rows=1024,
        teacher_labels_used=False,
        target_prior_row_order_masks_unchanged=True,
        representation_family_bridge_not_NNUE_resume=True,
        data_parent_is_frozen_original_zero_NNUE=True,
        current_pairedzero_playing_function_equal_requires_actual_packet_qualification=True,
    )
    return paired, result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seal", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    seal = json.loads(a.seal.read_bytes())
    check = guard(seal, 600)
    data, provenance = derive(seal, check)
    check()
    if not a.output.resolve().is_relative_to("/dev/shm"):
        raise ValueError("RAM output")
    a.output.mkdir(parents=True, exist_ok=False)
    publish(a.output / "dataset.json", data)
    provenance["dataset_sha256"] = sha(a.output / "dataset.json")
    publish(a.output / "provenance.json", provenance)
