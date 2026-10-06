"""Optional-only TRAIN; unchanged labels."""

import hashlib
import importlib.util
import json
from pathlib import Path

import chess

PARENT = Path("/workspace/work/harbichess/cpu-selective-quiescence-proposal/model.py")
PARENT_SHA = "14f5a1790f5e9518170a7d9bd490ba0aa61c001086286798a6d1315f00e0b9d2"


def load_parent(name, expected):
    path = PARENT.with_name(name)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("frozen parent source")
    spec = importlib.util.spec_from_file_location("balanced_parent_" + name.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    if name == "learner.py":
        import sys

        sys.path.insert(0, str(PARENT.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        if name == "learner.py":
            sys.path.pop(0)
    return module


old = load_parent("model.py", PARENT_SHA)
DIM, probability = old.DIM, old.probability


def digest(x):
    return hashlib.sha256(
        json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def group_digest(groups):
    return digest(groups)


def model_dict(weights):
    packet = old.model_dict(weights)
    packet.update(
        schema="own-selective-quiescence-balanced-risk-model-v2",
        score_semantics="balanced-class-risk-logit-not-calibrated-posterior",
        training_scope="nonchecked-TRAIN-only",
    )
    return packet


def summary(groups):
    positives = sum(r["target"] for rows in groups.values() for r in rows)
    count = sum(len(rows) for rows in groups.values())
    if len(groups) < 16 or positives < 64 or positives == count:
        raise ValueError("INCOMPLETE16groups/64positives")
    prevalence = sum(sum(r["target"] for r in rows) / len(rows) for rows in groups.values()) / len(
        groups
    )
    return dict(
        groups=len(groups),
        rows=count,
        positives=positives,
        game_equal_positive_prevalence=prevalence,
        pos_weight=(1 - prevalence) / prevalence,
        gradient_normalization=2 * (1 - prevalence),
    )


def transform(data, split):
    train, validation = set(split["train"]), set(split["validation"])
    if (
        train & validation
        or len(train) != len(split["train"])
        or len(validation) != len(split["validation"])
    ):
        raise ValueError("split leakage")
    groups = {}
    for row in data["leaves"]:
        if row["trajectory_id"] not in train | validation:
            raise ValueError("unknown trajectory")
        h = row["history"]
        board = chess.Board(h["root_fen"])
        if not board.is_valid():
            raise ValueError("invalid root")
        for uci in h["prefix_uci"]:
            move = chess.Move.from_uci(uci)
            if not board.is_legal(move):
                raise ValueError("illegal leaf fullhistory")
            board.push(move)
        checked = board.is_check()
        if (
            board.outcome(claim_draw=True) is not None
            or digest(h) != row["history_sha256"]
            or row["features"] != old.features(board, row["static"])
            or ("checked" in row and row["checked"] != checked)
        ):
            raise ValueError("leaf mismatch")
        if row["status"] == "censored":
            if row["target"] is not None or row["reference_mover_value"] is not None:
                raise ValueError("UNKNOWN is not negative")
            continue
        if row["status"] != "completed" or row["target"] != int(
            abs(row["reference_mover_value"] - row["static"]) > 0.10
        ):
            raise ValueError("unchanged literal own-Q label")
        if not checked and row["trajectory_id"] in train:
            groups.setdefault(row["trajectory_id"], []).append(row)
    record = summary(groups)
    record.update(
        schema="own-selective-Q-nonchecked-game-equal-transform-v2",
        training_group_sha256=group_digest(groups),
        dataset_canonical_sha256=digest(data),
        split_canonical_sha256=digest(split),
        feature_source_sha256=PARENT_SHA,
    )
    return groups, record
