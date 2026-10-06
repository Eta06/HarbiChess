"""Strict teacher-once converter; no engine, network, forward or optimizer."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import chess


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pinned(row):
    p = Path(row["path"])
    if sha(p) != row["sha256"]:
        raise ValueError("input/source SHA mismatch")
    return p


def module(row, name):
    spec = importlib.util.spec_from_file_location(name, pinned(row))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def verify_row(row, selected, ordinal, features, prior):
    if any(row.get(k) != v for k, v in selected.items()) or row["ordinal"] != ordinal:
        raise ValueError("selection alignment")
    board = chess.Board(row["root_fen"])
    if not board.is_valid():
        raise ValueError("root validity")
    for uci in row["prefix_uci"]:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError("full legal history")
        board.push(move)
    hist = hashlib.sha256(
        (row["root_fen"] + "\n" + " ".join(row["prefix_uci"])).encode()
    ).hexdigest()
    if (hist != row["history_sha256"] or row["fen4"] != " ".join(board.fen().split()[:4])
            or row["root_mover"] != ("white" if board.turn else "black")
            or board.outcome(claim_draw=True) is not None):
        raise ValueError("fullhistory/mover/nonterminal")
    if chess.Move.from_uci(row["selected_teacher_uci"]) not in board.legal_moves:
        raise ValueError("teacher selected move illegal")
    nodes = row["teacher_nodes_actual"]
    if (type(nodes) is not int or nodes < 1 or row["nominal_nodes"] != 8192
            or row["node_overrun"] != nodes - 8192
            or type(row["teacher_depth"]) is not int or row["teacher_depth"] < 0
            or not math.isfinite(row["teacher_time_seconds"]) or row["teacher_time_seconds"] < 0):
        raise ValueError("actual teacher budget/depth")
    raw = row["raw_teacher"]
    cp, mate = raw["cp_mover"], raw["mate_mover"]
    if (cp is None) == (mate is None) or raw["target_is_calibrated_WDL"] is not False:
        raise ValueError("teacher proxy provenance")
    if cp is not None:
        if type(cp) is not int:
            raise ValueError("cp integer")
        target = math.tanh(cp / 600)
    else:
        if type(mate) is not int or mate == 0:
            raise ValueError("unambiguous mate sign")
        target = 1.0 if mate > 0 else -1.0
    expected_prior = sum(w * x for w, x in zip(prior.PRIOR, prior.features(board), strict=True))
    expected_prior /= prior.SCALE
    if (float(target).hex() != float(raw["target"]).hex()
            or float(target).hex() != float(row["target"]).hex()
            or expected_prior.hex() != float(row["prior_logit"]).hex()
            or row["indices"] != features.board_indices(board)):
        raise ValueError("proxy/features/prior exact reconstruction")
    return {k: row[k] for k in ("indices", "prior_logit", "target")}


def convert(spec):
    if spec["schema"] != "NNUE-teacher-dataset-conversion-seal-v1":
        raise ValueError("ROOT conversion seal required")
    reg = json.loads(pinned(spec["registration"]).read_bytes())
    selected = json.loads(pinned(spec["selection"]).read_bytes())
    labels = json.loads(gzip.decompress(pinned(spec["labels"]).read_bytes()))
    if (reg["phase"] != "teacher-labels" or reg["status"] != "registered"
            or reg["count"] != 4096 or len(selected["rows"]) != 4096
            or len(labels["rows"]) != 4096
            or labels["schema"] != "NNUE-teacher8192-ownTRAIN-bootstrap-labels-v1"
            or labels["phase"] != "teacher-bootstrap"
            or labels["registration_sha256"] != spec["registration"]["sha256"]
            or labels["seed"] != reg["seed"] or labels["teacher"] != reg["teacher"]
            or labels["source_closure"] != reg["source_sha256"]
            or labels["selection_receipt"] != selected["receipt"]
            or labels["teacher_bootstrap_is_selflearning"] is not False
            or labels["targets_are_calibrated_WDL"] is not False):
        raise ValueError("complete actual teacher/selection lineage")
    if not (labels["original_first_epoch"] == reg["original_first_epoch"]
            < labels["finished_epoch"] <= labels["original_deadline_epoch"]
            == reg["original_deadline_epoch"] <= 1791273600):
        raise ValueError("original teacher deadline")
    for p, h in reg["source_sha256"].items():
        pinned(dict(path=p, sha256=h))
    for key in ("known8_exclusions", "parent_registration", "nnue_feature"):
        pinned(reg[key])
    teacher = reg["teacher"]
    if (teacher["sha256"] != "0f83d24cc46d2c66c60f16001af5444873bc112b7d028594513426894c12da19"
            or teacher["asset_id"] != 545574859 or teacher["Threads"] != 1
            or teacher["compressed_sha256"] !=
            "9defc0d4e55d49c65a6d042f3e571a39fcea499ade6dbe741b53b8c65e03611f"
            or teacher["nominal_nodes"] != 8192
            or teacher["Hash"] != 16 or teacher["ClearHash_each_root"] is not True):
        raise ValueError("official fixed teacher")
    receipt = selected["receipt"]
    pool_sha = hashlib.sha256(canonical(selected["rows"])).hexdigest()
    if receipt["selected_fullhistory_sha256"] != pool_sha:
        raise ValueError("selected pool seal")
    parent = json.loads(pinned(reg["parent_registration"]).read_bytes())
    if (spec["prior"]["sha256"] != parent["closure"]["value_helper"]
            or Path(spec["prior"]["path"]) != Path(parent["source_dir"]) / "value.py"):
        raise ValueError("authoritative original prior")
    features = module(reg["nnue_feature"], "nnue_conversion_features")
    prior = module(spec["prior"], "nnue_conversion_prior")
    rows, ids, histories = [], set(), set()
    for i, (row, selection) in enumerate(zip(labels["rows"], selected["rows"], strict=True)):
        if row["row_id"] in ids or row["history_sha256"] in histories:
            raise ValueError("duplicate selected row/history")
        ids.add(row["row_id"])
        histories.add(row["history_sha256"])
        rows.append(verify_row(row, selection, i, features, prior))
    data = canonical(dict(schema="own-kingbucket-sparse-training-data-v1",
                          phase="teacher-bootstrap", rows=rows)) + b"\n"
    provenance = dict(schema="NNUE-teacher-converted-data-provenance-v1", inputs=spec,
                      seed=reg["seed"], rows=4096, dataset_sha256=hashlib.sha256(data).hexdigest(),
                      selection_receipt=receipt, teacher=teacher,
                      teacher_bootstrap_is_selflearning=False, targets_are_calibrated_WDL=False,
                      sampler="uniform rows from fixed game-balanced selected pool")
    return data, canonical(provenance) + b"\n"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seal", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    data, provenance = convert(json.loads(a.seal.read_bytes()))
    if not a.output.resolve().is_relative_to("/dev/shm"):
        raise ValueError("RAM-only outputs")
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "dataset.json").write_bytes(data)
    (a.output / "provenance.json").write_bytes(provenance)
