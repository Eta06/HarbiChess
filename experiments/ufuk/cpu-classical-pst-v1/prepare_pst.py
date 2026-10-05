"""Add PST features to, without changing, the frozen classical18 dataset."""

import hashlib
import importlib
import math
import sys
from pathlib import Path

import chess
from frozen_reference.value import PRIOR, SCALE, features
from pst_features import piece_square_features
from pst_value import FEATURE_COUNT

REF = Path(__file__).with_name("frozen_reference")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(path, config, protected, *, min_games=16, min_rows=1024):
    path_ref = str(REF)
    inserted = path_ref not in sys.path
    if inserted:
        sys.path.insert(0, path_ref)
    try:
        frozen_learner = importlib.import_module("learner")
        frozen_journal = importlib.import_module("journal_v3")
    finally:
        if inserted:
            sys.path.remove(path_ref)
    train18, val18, receipt, linear_dataset_sha = frozen_learner.prepare(
        path, config, protected, min_games=min_games, min_rows=min_rows
    )
    groups18 = {**train18, **val18}
    train, val = {}, {}
    assigned = set()
    state = frozen_journal.read(path)
    packets = frozen_journal.replay(state, config)
    known = {index: (positions, result) for _, index, positions, result in packets}
    for index, game in enumerate(state["games"]):
        root = config["roots"][game["root_index"]]
        moves = [row["action"] for row in game["moves"]]
        trajectory_sha = frozen_journal.digest(
            {"root_fen": root["root_fen"], "prefix": root["prefix"], "moves": moves}
        )
        if trajectory_sha not in groups18 or trajectory_sha in assigned:
            continue
        if index not in known:
            raise ValueError(
                "frozen learner admitted a trajectory without a known result"
            )
        positions, _ = known[index]
        old_rows = groups18[trajectory_sha]
        if len(positions) != len(old_rows) or len(moves) != len(old_rows):
            raise ValueError("frozen row/position alignment differs")
        rows = []
        for (board, _mover, stored_prior), old_row, action in zip(
            positions, old_rows, moves, strict=True
        ):
            if chess.Move.from_uci(action) not in board.legal_moves:
                raise ValueError(
                    "frozen action is no longer legal in PST reconstruction"
                )
            phi18, prior, z, q = old_row
            prior_logit = (
                sum(
                    weight * value
                    for weight, value in zip(PRIOR, features(board), strict=True)
                )
                / SCALE
            )
            if prior != prior_logit or math.tanh(prior_logit) != stored_prior:
                raise ValueError("frozen human-prior target alignment differs")
            rows.append((phi18, prior, z, q, piece_square_features(board), prior_logit))
        assigned.add(trajectory_sha)
        (train if trajectory_sha in train18 else val)[trajectory_sha] = rows
    if (
        assigned != set(groups18)
        or set(train) != set(train18)
        or set(val) != set(val18)
    ):
        raise ValueError("PST reconstruction changed the frozen trajectory split")
    for key in groups18:
        if len(groups18[key]) != len((train | val)[key]):
            raise ValueError("PST row count differs from frozen classical18 data")
        for old, new in zip(groups18[key], (train | val)[key], strict=True):
            if (
                old != new[:4]
                or len(new[4]) != FEATURE_COUNT
                or not isinstance(new[5], float)
            ):
                raise ValueError("targets or frozen feature columns changed")
    receipt = dict(
        **receipt,
        pst_feature_count=FEATURE_COUNT,
        linear_dataset_sha256=linear_dataset_sha,
        PST_L2=0.02,
        PST_smoothness=0.01,
        PST_scope="added zero-init spatial residual; same 18-term rows, targets and split",
    )
    data_sha = frozen_journal.digest(
        {
            "linear_dataset_sha256": linear_dataset_sha,
            "PST_schema": "rank8xfile4-mirror-files-a-h-v1",
            "receipt": receipt,
            "groups": {k: v for k, v in sorted((train | val).items())},
        }
    )
    return train, val, receipt, data_sha
