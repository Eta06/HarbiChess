"""Immutable schema1 packed native oracle data; reader/decoder are Torch-free."""

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from harbichess.chess.encoding import BoardEncoder
from harbichess.chess.packed_encoding import PACKED_SCHEMA, decode_positions, pack_board

PREPARED_SCHEMA = 1


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class NumpyOracleBatch:
    inputs: np.ndarray
    policies: np.ndarray
    legal_masks: np.ndarray
    wdl: np.ndarray
    value_weights: np.ndarray


class PreparedPanel:
    def __init__(self, cache, rows):
        self.cache = cache
        self.rows = np.array(rows, dtype=np.int64)
        self.rows.flags.writeable = False
        self.size = len(self.rows)

    def select(self, indices):
        if not indices or any(type(i) is not int or i < 0 or i >= self.size for i in indices):
            raise IndexError("prepared panel indices must be nonempty and in range")
        rows = self.rows[list(indices)]
        cache, count = self.cache, len(rows)
        inputs = decode_positions(cache.pieces[rows], cache.metadata[rows], cache.ep[rows])
        masks, policies = np.zeros((count, 4672), dtype=bool), np.zeros((count, 4672), np.float32)
        for index, row in enumerate(rows):
            legal = cache.legal[int(cache.legal_offsets[row]) : int(cache.legal_offsets[row + 1])]
            begin, end = int(cache.policy_offsets[row]), int(cache.policy_offsets[row + 1])
            masks[index, legal] = True
            policies[index, cache.policy[begin:end]] = cache.probabilities[begin:end]
        return NumpyOracleBatch(
            inputs, policies, masks, cache.wdl[rows].copy(), np.ones(count, np.float32)
        )


class PreparedOracle:
    def __init__(self, directory: Path, *, source: Path):
        manifest = json.loads((directory / "prepared.json").read_text())
        if (
            manifest.get("schema") != PREPARED_SCHEMA
            or manifest.get("packed_schema") != PACKED_SCHEMA
            or manifest.get("target_semantics") != "engine-reference"
            or manifest.get("encoder_schema") != 1
            or manifest.get("action_schema") != 1
        ):
            raise ValueError("unsupported prepared oracle schema")
        snapshot = manifest["source"]
        if digest(source / "dataset.json") != snapshot["dataset_sha256"]:
            raise ValueError("prepared oracle source dataset changed")
        if digest(source / "metadata.json") != snapshot["metadata_sha256"]:
            raise ValueError("prepared oracle source metadata changed")
        for name, sha in snapshot["files"].items():
            if Path(name).name != name or digest(source / name) != sha:
                raise ValueError("prepared oracle source file changed: " + name)
        count = manifest["rows"]
        shapes = {
            "pieces": ((count, 8, 12), "<u8"),
            "metadata": ((count, 8), "<f4"),
            "ep": ((count,), "|u1"),
            "split": ((count,), "|u1"),
            "family": ((count,), "<u4"),
            "position_key": ((count,), "|S96"),
            "wdl": ((count, 3), "<f4"),
            "legal_offsets": ((count + 1,), "<u8"),
            "policy_offsets": ((count + 1,), "<u8"),
            "legal": ((manifest["legal_entries"],), "<u2"),
            "policy": ((manifest["policy_entries"],), "<u2"),
            "probabilities": ((manifest["policy_entries"],), "<f4"),
        }
        if set(manifest["arrays"]) != set(shapes) or count <= 0:
            raise ValueError("prepared oracle arrays/row count mismatch")
        for name, (shape, dtype) in shapes.items():
            receipt = manifest["arrays"][name]
            path = directory / (name + ".npy")
            if digest(path) != receipt["sha256"] or path.stat().st_size != receipt["bytes"]:
                raise ValueError("prepared oracle array checksum mismatch: " + name)
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.shape != shape or array.dtype != np.dtype(dtype):
                raise ValueError("prepared oracle array shape/dtype mismatch: " + name)
            setattr(self, name, array)
        if (
            not np.isin(self.split, (0, 1)).all()
            or np.any(self.ep > 64)
            or np.any(self.legal >= 4672)
            or np.any(self.policy >= 4672)
            or not np.isfinite(self.metadata).all()
            or not np.isfinite(self.wdl).all()
            or not np.isfinite(self.probabilities).all()
            or np.any(self.wdl < 0)
            or np.any(self.probabilities < 0)
            or not np.allclose(self.wdl.sum(axis=1), 1, atol=1e-6, rtol=0)
        ):
            raise ValueError("prepared oracle invalid arrays")
        for offsets, flat in ((self.legal_offsets, self.legal), (self.policy_offsets, self.policy)):
            if offsets[0] != 0 or offsets[-1] != len(flat) or np.any(offsets[1:] <= offsets[:-1]):
                raise ValueError("prepared oracle invalid ragged offsets")
        self.directory, self.manifest = directory, manifest

    def panels(self, *, max_train_rows=None, max_validation_rows=None, seed=20261003):
        training_keys = set(self.position_key[self.split == 0])
        rows = {
            "train": np.flatnonzero(self.split == 0).tolist(),
            "validation": [
                int(i)
                for i in np.flatnonzero(self.split == 1)
                if self.position_key[i] not in training_keys
            ],
        }
        overlap = int(np.sum(self.split == 1)) - len(rows["validation"])
        available = {split: len(indices) for split, indices in rows.items()}
        for split, cap in (("train", max_train_rows), ("validation", max_validation_rows)):
            if cap is not None:
                if type(cap) is not int or cap <= 0:
                    raise ValueError("prepared panel caps must be positive integers")
                if len(rows[split]) > cap:
                    rng = random.Random(f"{seed}:{split}")
                    selected = sorted(rng.sample(range(len(rows[split])), cap))
                    rows[split] = [rows[split][i] for i in selected]
        if any(not indices for indices in rows.values()):
            raise ValueError("empty prepared learning panel")
        info = {
            "train_rows": len(rows["train"]),
            "validation_rows": len(rows["validation"]),
            "removed_validation_position_overlap": overlap,
            "train_families": sorted({int(self.family[i]) for i in rows["train"]}),
            "validation_families": sorted({int(self.family[i]) for i in rows["validation"]}),
            "dataset_sha256": self.manifest["source"]["dataset_sha256"],
            "available_rows": available,
        }
        return PreparedPanel(self, rows["train"]), PreparedPanel(self, rows["validation"]), info


def prepare(directory: Path, *, source: Path, wall_seconds=600):
    # Raw validation/preparation uses the existing authoritative native parser.
    # Prepared reading/decoding above intentionally has no Torch dependency.
    from harbichess.training.oracle_data import read_game, validate_row

    if wall_seconds <= 0:
        raise ValueError("positive preparation wall budget required")
    started = time.perf_counter()
    source_manifest = json.loads((source / "dataset.json").read_text())
    if source_manifest.get("schema") != 1 or source_manifest.get("status") != "completed":
        raise ValueError("complete native schema1 dataset required")
    snapshot = {
        "dataset_sha256": digest(source / "dataset.json"),
        "metadata_sha256": digest(source / "metadata.json"),
        "files": source_manifest["files"],
    }
    count = legal_count = policy_count = 0
    for name, sha in snapshot["files"].items():
        if Path(name).name != name or digest(source / name) != sha:
            raise ValueError("source file checksum mismatch")
        for row in read_game(source / name)["rows"]:
            count += 1
            legal_count += len(row["legal"])
            policy_count += len(row["policy"])
    if count == 0:
        raise ValueError("empty source dataset")
    if source_manifest.get("rows") != count:
        raise ValueError("source row count mismatch")
    directory.mkdir(parents=True, exist_ok=False)
    shapes = {
        "pieces": ((count, 8, 12), "<u8"),
        "metadata": ((count, 8), "<f4"),
        "ep": ((count,), "|u1"),
        "split": ((count,), "|u1"),
        "family": ((count,), "<u4"),
        "position_key": ((count,), "|S96"),
        "wdl": ((count, 3), "<f4"),
        "legal_offsets": ((count + 1,), "<u8"),
        "policy_offsets": ((count + 1,), "<u8"),
        "legal": ((legal_count,), "<u2"),
        "policy": ((policy_count,), "<u2"),
        "probabilities": ((policy_count,), "<f4"),
    }
    arrays = {
        name: np.lib.format.open_memmap(
            directory / (name + ".npy"), mode="w+", dtype=dtype, shape=shape
        )
        for name, (shape, dtype) in shapes.items()
    }
    cursor = legal_cursor = policy_cursor = 0
    encoder = BoardEncoder()
    for name, sha in snapshot["files"].items():
        game = read_game(source / name)
        for row in game["rows"]:
            if time.perf_counter() - started >= wall_seconds:
                raise TimeoutError(
                    "preparation budget; retain incomplete cache, never publish manifest"
                )
            if row["split"] != game["job"]["split"] or row["family"] != game["job"]["family"]:
                raise ValueError("oracle split/family provenance mismatch")
            if (
                row["split"] not in ("train", "validation")
                or type(row["family"]) is not int
                or not 0 <= row["family"] < 2**32
            ):
                raise ValueError("unsupported split/family")
            board = validate_row(row)
            if row["position_key"] != " ".join(board.fen().split()[:4]):
                raise ValueError("position-key provenance mismatch")
            p, m, e = pack_board(board)
            decoded = decode_positions(p[None], m[None], np.array([e], np.uint8))
            original = np.array(encoder.encode_board(board).values, np.float32).reshape(
                1, 8, 8, 104
            )
            if not np.array_equal(decoded, original):
                raise ValueError("packed input differs from authoritative encoder")
            key = row["position_key"].encode("ascii")
            if len(key) > 96:
                raise ValueError("position key exceeds prepared schema")
            arrays["pieces"][cursor], arrays["metadata"][cursor], arrays["ep"][cursor] = p, m, e
            arrays["split"][cursor] = 0 if row["split"] == "train" else 1
            arrays["family"][cursor], arrays["position_key"][cursor] = row["family"], key
            arrays["wdl"][cursor] = row["wdl"]
            arrays["legal_offsets"][cursor], arrays["policy_offsets"][cursor] = (
                legal_cursor,
                policy_cursor,
            )
            for _, action in row["legal"]:
                arrays["legal"][legal_cursor] = action
                legal_cursor += 1
            for _, action, probability in row["policy"]:
                arrays["policy"][policy_cursor], arrays["probabilities"][policy_cursor] = (
                    action,
                    probability,
                )
                policy_cursor += 1
            cursor += 1
        if digest(source / name) != sha:
            raise ValueError("source changed during preparation")
    if digest(source / "dataset.json") != snapshot["dataset_sha256"]:
        raise ValueError("source manifest changed during preparation")
    if digest(source / "metadata.json") != snapshot["metadata_sha256"]:
        raise ValueError("source metadata changed during preparation")
    assert (cursor, legal_cursor, policy_cursor) == (count, legal_count, policy_count)
    arrays["legal_offsets"][count], arrays["policy_offsets"][count] = legal_cursor, policy_cursor
    for array in arrays.values():
        array.flush()
    arrays.clear()
    for name in shapes:
        with (directory / (name + ".npy")).open("rb") as stream:
            os.fsync(stream.fileno())
    receipts = {
        name: {
            "bytes": (directory / (name + ".npy")).stat().st_size,
            "sha256": digest(directory / (name + ".npy")),
        }
        for name in shapes
    }
    manifest = {
        "schema": PREPARED_SCHEMA,
        "packed_schema": PACKED_SCHEMA,
        "encoder_schema": 1,
        "action_schema": 1,
        "target_semantics": "engine-reference",
        "source": snapshot,
        "rows": count,
        "legal_entries": legal_count,
        "policy_entries": policy_count,
        "arrays": receipts,
        "every_row_full_input_exact": True,
        "wall_seconds_including_full_row_parity": time.perf_counter() - started,
        "scope": "Lossless prepared data only; original raw source preserved, not training resume.",
    }
    # Reuse immutable atomic publication; an interrupted directory has no accepted manifest.
    from harbichess.training.oracle_data import publish_json

    publish_json(directory / "prepared.json", manifest)
    return manifest


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=float, default=600)
    result = prepare(**vars(parser.parse_args()))
    print(
        json.dumps({key: value for key, value in result.items() if key not in ("arrays", "source")})
    )


if __name__ == "__main__":
    main()
