"""Versioned native teacher-data derivation, preserving original anchor bytes."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import time
from pathlib import Path

from harbichess.backends.torch_network import sha256
from harbichess.training.oracle_data import publish_json, read_game, validate_row

MERGE_SCHEMA = 1


def merge(directory: Path, anchor: Path, broader: Path, *, wall_seconds: float = 3600) -> dict:
    if directory.exists():
        raise FileExistsError(directory)
    if wall_seconds <= 0 or anchor.resolve() == broader.resolve():
        raise ValueError("distinct immutable sources and positive derivation budget required")
    started = time.perf_counter()
    snapshots, heldout_keys = [], set()

    def budget():
        if time.perf_counter() - started >= wall_seconds:
            raise TimeoutError("merge budget exhausted; no complete manifest published")

    # Protect ALL heldout positions before deriving any new training rows.
    for source in (anchor, broader):
        manifest = json.loads((source / "dataset.json").read_text())
        if manifest.get("schema") != 1 or manifest.get("status") != "completed":
            raise ValueError("only completed native sources can be merged")
        snapshot = {
            "path": str(source.resolve()),
            "dataset_sha256": sha256(source / "dataset.json"),
            "metadata_sha256": sha256(source / "metadata.json"),
            "files": manifest["files"],
        }
        count = 0
        for name, digest in snapshot["files"].items():
            budget()
            if Path(name).name != name or sha256(source / name) != digest:
                raise ValueError("parent source file checksum/path mismatch")
            game = read_game(source / name)
            for row in game["rows"]:
                budget()
                if row["split"] != game["job"]["split"] or row["family"] != game["job"]["family"]:
                    raise ValueError("parent split/family mismatch")
                board = validate_row(row)
                if row["position_key"] != " ".join(board.fen().split()[:4]):
                    raise ValueError("parent position-key mismatch")
                if row["split"] not in ("train", "validation"):
                    raise ValueError("unsupported parent split")
                if row["split"] == "validation":
                    heldout_keys.add(row["position_key"])
                count += 1
        if count != manifest.get("rows"):
            raise ValueError("parent row count mismatch")
        snapshots.append(snapshot)

    directory.mkdir(parents=True, exist_ok=False)
    metadata = {
        "schema": 1,
        "derivation_schema": MERGE_SCHEMA,
        "target_semantics": "engine-reference",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "derivation_code_sha256": sha256(Path(__file__)),
        "parents": snapshots,
        "family_namespace": {
            "anchor": "unchanged historical family identifiers qualified by split",
            "broader_train_offset": 100000,
            "broader_validation_offset": 200000,
        },
        "selection": (
            "Broader training excludes every original/new validation position key. "
            "Anchor bytes unchanged; later prepared loader removes any remaining "
            "validation overlap with anchor train."
        ),
        "scope": "New derived dataset, not parent in-place mutation or training/optimizer resume.",
    }
    publish_json(directory / "metadata.json", metadata)
    files, provenance, counts = (
        {},
        {},
        {"anchor": 0, "broader": 0, "dropped_broader_training_overlap": 0},
    )
    family_counts = {}
    for parent_index, source in enumerate((anchor, broader)):
        kind = "anchor" if parent_index == 0 else "broader"
        for name, digest in snapshots[parent_index]["files"].items():
            budget()
            original = source / name
            if sha256(original) != digest:
                raise ValueError("parent changed during derivation")
            game = read_game(original)
            target = directory / (kind + "-" + name)
            original_rows = len(game["rows"])
            if parent_index == 0:
                shutil.copyfile(original, target)
                if sha256(target) != digest:
                    raise ValueError("anchor byte preservation failed")
            else:
                game = copy.deepcopy(game)
                split, old_family = game["job"]["split"], game["job"]["family"]
                if type(old_family) is not int or not 0 <= old_family < 100000:
                    raise ValueError("broader family cannot enter version1namespace")
                new_family = old_family + (100000 if split == "train" else 200000)
                kept = [
                    row
                    for row in game["rows"]
                    if split != "train" or row["position_key"] not in heldout_keys
                ]
                counts["dropped_broader_training_overlap"] += original_rows - len(kept)
                game["job"].update(family=new_family, source_family=old_family)
                for row in kept:
                    row.update(family=new_family, source_family=old_family)
                game["rows"] = kept
                game["derivation"] = {
                    "schema": MERGE_SCHEMA,
                    "parent_dataset_sha256": snapshots[parent_index]["dataset_sha256"],
                    "parent_file": name,
                    "parent_file_sha256": digest,
                    "parent_rows": original_rows,
                    "kept_rows": len(kept),
                    "dropped_training_heldout_overlap": original_rows - len(kept),
                    "scope": (
                        "Targets/legal/history unchanged; family namespaced, heldout-overlap "
                        "rows excluded. Original generator queries/nodes/final outcome describe "
                        "parent complete trajectory, not a reconstructed shorter game."
                    ),
                }
                publish_json(target, game)
            counts[kind] += len(game["rows"])
            family_key = f"{kind}:{game['job']['split']}:{game['job']['family']}"
            family_counts[family_key] = family_counts.get(family_key, 0) + len(game["rows"])
            files[target.name] = sha256(target)
            provenance[target.name] = {
                "parent": parent_index,
                "parent_file": name,
                "parent_sha256": digest,
                "original_rows": original_rows,
                "kept_rows": len(game["rows"]),
            }
    for source, snapshot in zip((anchor, broader), snapshots, strict=True):
        if (
            sha256(source / "dataset.json") != snapshot["dataset_sha256"]
            or sha256(source / "metadata.json") != snapshot["metadata_sha256"]
        ):
            raise ValueError("parent metadata/manifest changed during derivation")
        for name, digest in snapshot["files"].items():
            budget()
            if sha256(source / name) != digest:
                raise ValueError("parent source changed during derivation")
    result = {
        "schema": 1,
        "status": "completed",
        "derivation_schema": MERGE_SCHEMA,
        "games": len(files),
        "rows": counts["anchor"] + counts["broader"],
        "files": files,
        "parent_files": provenance,
        "counts": counts,
        "family_row_counts": family_counts,
        "protected_validation_position_keys": len(heldout_keys),
        "wall_seconds": time.perf_counter() - started,
        "scope": (
            "Version1raw native derivation only; source bytes retained, no optimizer/training "
            "resume or strength claim. Zero-row derived games retained with provenance, "
            "not counted as sampled rows."
        ),
    }
    publish_json(directory / "dataset.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--broader", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=float, default=3600)
    result = merge(**vars(parser.parse_args()))
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("files", "parent_files", "family_row_counts")
            }
        )
    )


if __name__ == "__main__":
    main()
