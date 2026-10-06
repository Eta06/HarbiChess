"""Strict own-Q/fullhistory/terminal trace converter; no forward, engine, SGD or teacher."""

import argparse
import hashlib
import importlib.util
import json
import math
import struct
import time
from pathlib import Path

import chess


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pinned(row):
    p = Path(row["path"])
    if sha(p) != row["sha256"]:
        raise ValueError("sealed input/source SHA")
    return p


def module(row, name):
    spec = importlib.util.spec_from_file_location(name, pinned(row))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def board_at(row):
    if row["root_fen"] != chess.STARTING_FEN:
        raise ValueError("standard fullhistory start")
    b = chess.Board(row["root_fen"])
    for uci in row["history_uci"]:
        move = chess.Move.from_uci(uci)
        if not b.is_legal(move):
            raise ValueError("illegal fullhistory")
        b.push(move)
    if b.outcome(claim_draw=True) is not None:
        raise ValueError("training root must be nonterminal")
    return b


def alias(b):
    return int.from_bytes(
        hashlib.sha256(min(b.board_fen(), b.mirror().board_fen()).encode()).digest()[:8],
        "little",
        signed=True,
    )


def verify_row(row, features, prior):
    b = board_at(row)
    if (
        row["fen4"] != " ".join(b.fen().split()[:4])
        or row["mover"] != ("white" if b.turn else "black")
        or row["root_alias"] != alias(b)
        or not b.is_legal(chess.Move.from_uci(row["selected_best_uci"]))
    ):
        raise ValueError("mover/root/selected-action legality")
    q = row["raw_q_mover"]
    if (
        not math.isfinite(q)
        or abs(q) > 2
        or row["clipped_q_mover"] != max(-1.0, min(1.0, q))
        or row["mate_range_score_returned"] != (abs(q) > 1)
        or row["behavior_policy_available"] is not False
        or row["label_source"] != "own-frozen-parent-search"
    ):
        raise ValueError("own Q proxy/clip/provenance; not calibrated WDL")
    if (
        type(row["root_actions"]) is not int
        or row["root_actions"] != b.legal_moves.count()
        or type(row["nodes"]) is not int
        or not row["root_actions"] + 1 <= row["nodes"] <= 8192
        or type(row["evaluations"]) is not int
        or row["evaluations"] != row["actual_eval_calls"]
        or not 0 <= row["evaluations"] <= row["nodes"]
        or type(row["completed_depth"]) is not int
        or not 0 <= row["completed_depth"] <= 8
    ):
        raise ValueError("completed root search/global actual counters")
    prior_logit = (
        sum(w * x for w, x in zip(prior.PRIOR, prior.features(b), strict=True)) / prior.SCALE
    )
    return dict(
        indices=features.board_indices(b),
        prior_logit=prior_logit,
        target=float(row["clipped_q_mover"]),
    )


def convert(spec, guard=lambda: None):
    if spec["schema"] != "NNUE-own1024-dataset-conversion-seal-v1":
        raise ValueError("own seal required")
    reg = json.loads(pinned(spec["registration"]).read_bytes())
    receipt = json.loads(pinned(spec["receipt"]).read_bytes())
    if (
        reg["schema"] != "own-nnue-ownq-collection-v1"
        or reg["status"] != "registered"
        or receipt["schema"] != "own-nnue-ownq-collection-receipt-v1"
        or receipt["status"] != "PASS-exact-row-budget"
        or receipt["train_rows"] != 1024
        or receipt["registration_sha256"] != spec["registration"]["sha256"]
        or receipt["parent_candidate_sha256"] != reg["parent_candidate"]["sha256"]
        or receipt["teacher_labels_used"] is not False
        or receipt["search"] != dict(nodes=8192, qdepth=2, max_depth=8)
        or receipt["root_pool_sha256"] != reg["root_pool"]["sha256"]
        or receipt["protected_aliases_sha256"] != reg["protected_aliases"]["sha256"]
        or not reg["original_first_epoch"]
        <= receipt["finished_epoch"]
        <= receipt["original_deadline_epoch"]
        == reg["original_deadline_epoch"]
        <= 1791273600
    ):
        raise ValueError("exact closed current-parent own-search1024 lineage")
    for entry in [
        reg["parent_candidate"],
        reg["root_pool"],
        reg["protected_aliases"],
        reg["search_helper"],
    ]:
        pinned(entry)
    for name, digest in reg["producer_source_sha256"].items():
        pinned(dict(path=str(Path(spec["producer_directory"]) / name), sha256=digest))
    protected_raw = pinned(reg["protected_aliases"]).read_bytes()
    if len(protected_raw) % 8:
        raise ValueError("int64 protected aliases")
    protected = set(struct.unpack(f"<{len(protected_raw) // 8}q", protected_raw))
    feature = module(spec["features"], "own_conversion_features")
    prior = module(spec["prior"], "own_conversion_prior")
    if spec["prior"]["sha256"] != reg["parent_helpers"]["prior_sha256"]:
        raise ValueError("original authoritative parent prior")
    events = [json.loads(line) for line in pinned(spec["events"]).read_bytes().splitlines()]
    starts = {}
    rows = {}
    ends = {}
    aliases = set()
    selected = {}
    for event in events:
        guard()
        kind = event["type"]
        key = event.get("root_id")
        if kind == "game_start":
            if key in starts:
                raise ValueError("duplicate trajectory start")
            starts[key] = event
            rows[key] = []
        elif kind == "search_row":
            row = event["row"]
            key = row["root_id"]
            if key not in starts or key in ends:
                raise ValueError("chronological active game")
            sequence = rows[key]
            start = starts[key]
            if any(x["selected_action_played"] is not True for x in sequence):
                raise ValueError("no action after discarded protected search row")
            prefix = start["root_prefix_uci"] + [x["selected_best_uci"] for x in sequence]
            if (
                row["local_ply"] != len(sequence)
                or row["history_uci"] != prefix
                or row["root_prefix_uci"] != start["root_prefix_uci"]
                or row["root_fen"] != start["root_fen"]
                or row["root_ordinal"] != start["root_ordinal"]
            ):
                raise ValueError("exact chronological complete-history row chain")
            ref = row["search_alias_ref"]
            file = ref["file"]
            if file not in spec["alias_chunks"]:
                raise ValueError("unsealed alias chunk")
            raw = pinned(spec["alias_chunks"][file]).read_bytes()
            off = ref["offset_bytes"]
            n = ref["count"]
            if (
                type(off) is not int
                or type(n) is not int
                or off < 0
                or n < 0
                or off % 8
                or off + 8 * n > len(raw)
            ):
                raise ValueError("exact alias offset/count")
            vals = list(struct.unpack(f"<{n}q", raw[off : off + 8 * n]))
            if vals != sorted(set(vals)) or n > row["actual_eval_calls"]:
                raise ValueError("sorted unique evaluator traces")
            if (
                row["protected_search_aliases"] != sorted(set(vals) & protected)
                or type(row["selected_action_played"]) is not bool
                or row["selected_action_played"] != (not row["protected_search_aliases"])
            ):
                raise ValueError("exact protected search witnesses/action played")
            aliases.update(vals)
            row["_aliases"] = vals
            sequence.append(row)
        elif kind == "game_end":
            if key not in starts or key in ends:
                raise ValueError("one closed episode each")
            ends[key] = event
        elif kind == "protected-hit":
            continue
        else:
            raise ValueError("failed/unknown producer event is not closed data")
    if set(starts) != set(ends):
        raise ValueError("partial games cannot be sealed")
    trace = []
    for key, sequence in rows.items():
        guard()
        end = ends[key]
        hit = any(r["root_alias"] in protected or set(r["_aliases"]) & protected for r in sequence)
        if sequence:
            final = board_at(sequence[-1])
            if sequence[-1]["selected_action_played"]:
                final.push_uci(sequence[-1]["selected_best_uci"])
        else:
            final = chess.Board(starts[key]["root_fen"])
            for u in starts[key]["root_prefix_uci"]:
                final.push_uci(u)
        outcome = final.outcome(claim_draw=True)
        status = end["status"]
        eligible = end["training_eligible"]
        if hit or status == "excluded-protected-trajectory":
            if eligible:
                raise ValueError("whole trajectory protected path/branch leakage")
            labels = [None] * len(sequence)
        elif status == "completed-own-terminal":
            if outcome is None or not eligible:
                raise ValueError("actual own terminal required")
            white = 0 if outcome.winner is None else (1 if outcome.winner else -1)
            labels = [white if r["mover"] == "white" else -white for r in sequence]
        elif status in ["unknown-ply-cap", "unknown-row-budget-prefix"]:
            if outcome is not None or not eligible:
                raise ValueError("UNKNOWN cap is not terminal draw")
            labels = [None] * len(sequence)
        else:
            raise ValueError("explicit censor/terminal status")
        if end["row_labels"] != labels:
            raise ValueError("terminal alternating mover WDL")
        for r, z in zip(sequence, labels, strict=True):
            parsed = verify_row(r, feature, prior)
            rid = key + ":" + str(r["local_ply"])
            if eligible:
                selected[rid] = parsed
            trace.append(
                dict(
                    row_id=rid,
                    history_sha256=hashlib.sha256(
                        canonical(dict(root_fen=r["root_fen"], prefix_uci=r["history_uci"]))
                    ).hexdigest(),
                    raw_q_mover=r["raw_q_mover"],
                    target=r["clipped_q_mover"],
                    own_wdl_mover=z,
                    outcome_status="UNKNOWN" if z is None else "known-own-terminal",
                    episode_status=status,
                    train_eligible=eligible,
                )
            )
    ids = receipt["training_row_ids"]
    if (
        len(ids) != 1024
        or len(set(ids)) != 1024
        or set(ids) != set(selected)
        or receipt["all_actor_rows"] != sum(map(len, rows.values()))
    ):
        raise ValueError("exact all-row reconciliation/1024 eligible rows")
    data = (
        canonical(
            dict(
                schema="own-kingbucket-sparse-training-data-v1",
                phase="own-learning",
                rows=[selected[i] for i in ids],
            )
        )
        + b"\n"
    )
    provenance = dict(
        schema="NNUE-own1024-converted-data-provenance-v1",
        inputs=spec,
        seed=reg["seed"],
        rows=1024,
        dataset_sha256=hashlib.sha256(data).hexdigest(),
        trace=trace,
        teacher_labels_used=False,
        target=("clipped completed current-parent root-search mover Q; "
                "terminal WDL audited separately"),
        mate_score_semantics=("producer search mate-range score, "
                              "not an independent mate certificate"),
        initializer="named teacher candidate WEIGHTS only; newAdam/globalTorch/privateSampler",
        protected_scope="entire trajectory actual-path and traced static-evaluator aliases",
        unique_search_aliases=len(aliases),
    )
    return data, canonical(provenance) + b"\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seal", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    spec = json.loads(a.seal.read_bytes())
    first, end = spec["first"], spec["deadline"]
    if not first <= time.time() < end <= min(first + 600, 1791273600):
        raise ValueError("ROOT original600")
    import shutil
    import sys

    sys.path.insert(0, spec["core_repo"] + "/src")
    from harbichess.training.cgroup_budget import CgroupMemoryBudget

    memory = CgroupMemoryBudget(15 * 2**30)

    def guard():
        memory.check()
        if time.time() >= end:
            raise TimeoutError("same original converter600")
        if shutil.disk_usage("/workspace").free < 256 * 2**20:
            raise RuntimeError("workspace floor")

    data, provenance = convert(spec, guard)
    guard()
    if len(data) + len(provenance) > 96 * 2**20 or not a.output.resolve().is_relative_to(
        "/dev/shm"
    ):
        raise ValueError("RAM96MiB phase cap")
    a.output.mkdir(parents=True, exist_ok=False)
    for name, raw in [("dataset.json", data), ("provenance.json", provenance)]:
        with (a.output / name).open("xb") as stream:
            stream.write(raw)
    guard()
    with (a.output / "result.json").open("x") as stream:
        json.dump(
            dict(
                status="PASS-own1024-fullhistory-trace-conversion-not-strength",
                first=first,
                deadline=end,
                finished=time.time(),
                dataset_sha256=sha(a.output / "dataset.json"),
                provenance_sha256=sha(a.output / "provenance.json"),
            ),
            stream,
            sort_keys=True,
        )


if __name__ == "__main__":
    main()
