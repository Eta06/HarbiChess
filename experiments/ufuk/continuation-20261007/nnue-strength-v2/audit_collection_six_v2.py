"""ROOT-only six chronological own-Q packets per seed; no new games or SGD."""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

ORDINALS = (0, 204, 409, 614, 819, 1023)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, digest, name):
    path = Path(path).resolve()
    if sha(path) != digest:
        raise ValueError("source/input SHA differs")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def check_packet(row, actual, aliases, board):
    expected = {k: row[k] for k in actual if k != "value_hex"}
    expected["value_hex"] = float(row["raw_q_mover"]).hex()
    if actual != expected or aliases != sorted(set(aliases)):
        raise ValueError("actual packet or alias canonical order differs")
    if (row["fen4"] != " ".join(board.fen().split()[:4])
            or row["mover"] != ("white" if board.turn else "black")
            or not 1 <= actual["nodes"] <= 8192
            or actual["root_actions"] != board.legal_moves.count()
            or actual["evaluations"] != actual["actual_eval_calls"]
            or row["clipped_q_mover"] != max(-1., min(1., row["raw_q_mover"]))
            or row["mate_range_score_returned"] != (abs(row["raw_q_mover"]) > 1)):
        raise ValueError("history/POV/accounting packet differs")


def alias_segment(row, receipt, directory):
    ref = row["search_alias_ref"]
    entry = next(x for x in receipt["alias_chunks"] if x["file"] == ref["file"])
    p = Path(directory) / ref["file"]
    if p.parent.resolve() != Path(directory).resolve():
        raise ValueError("alias chunk path escape")
    if p.stat().st_size != entry["bytes"] or sha(p) != entry["sha256"]:
        raise ValueError("alias chunk bytes/SHA")
    offset, count = ref["offset_bytes"], ref["count"]
    if (type(offset) is not int or type(count) is not int or offset < 0
            or count < 0 or offset % 8 or offset + count * 8 > entry["bytes"]):
        raise ValueError("alias segment bounds")
    with p.open("rb") as f:
        f.seek(offset)
        data = f.read(count * 8)
    values = list(struct.unpack(f"<{count}q", data))
    if values != sorted(set(values)):
        raise ValueError("alias segment not sorted unique")
    return values


def selected_rows(raw_rows, ids, prescribed):
    if (len(ids) != 1024 or len(set(ids)) != 1024
            or any(type(x) is not str for x in ids)
            or prescribed != [ids[x] for x in ORDINALS]):
        raise ValueError("exact fixed string row IDs/chronological six selection")
    mapping = {}
    for row in raw_rows:
        if type(row["local_ply"]) is not int or row["local_ply"] < 0:
            raise ValueError("typed nonnegative local ply")
        key = row["root_id"] + ":" + str(row["local_ply"])
        if key in mapping:
            raise ValueError("duplicate raw root:ply row ID")
        mapping[key] = row
    if any(key not in mapping for key in ids):
        raise ValueError("all declared eligible IDs must exist in raw rows")
    eligible = set(ids)
    if [key for key in mapping if key in eligible] != ids:
        raise ValueError("eligible row ID order must equal actual raw chronology")
    return [(key, mapping[key]) for key in prescribed]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("registration", "receipt", "events", "clock", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    clock = json.loads(a.clock.read_bytes())
    reg = json.loads(a.registration.read_bytes())
    receipt = json.loads(a.receipt.read_bytes())
    if (clock["schema"] != "NNUE-own-collection-six-root-audit-clock-v2"
            or not clock["first"] <= time.time() < clock["deadline"]
            <= min(clock["first"] + 600, reg["operator_end_epoch"])
            or clock["helper_sha256"] != sha(__file__)
            or any(clock[k + "_sha256"] != sha(getattr(a, k))
                   for k in ("registration", "receipt", "events"))):
        raise ValueError("ROOT observed immutable original600 audit clock/bindings")
    if (reg["schema"] != "own-nnue-ownq-collection-registration-v2"
            or reg["status"] != "registered" or reg["seed"] not in (20262905, 20262906)
            or receipt["schema"] != "own-nnue-ownq-collection-receipt-v2"
            or receipt["status"] != "PASS-exact-row-budget"
            or receipt["seed"] != reg["seed"] or receipt["train_rows"] != 1024
            or receipt["registration_sha256"] != sha(a.registration)
            or receipt["events_sha256"] != sha(a.events)
            or receipt["events_bytes"] != a.events.stat().st_size
            or receipt["parent_candidate_sha256"] != reg["parent_candidate"]["sha256"]
            or receipt["producer_source_sha256"] != reg["producer_source_sha256"]
            or receipt["search"] != reg["search"]
            or reg["search"] != dict(nodes=8192, qdepth=2, max_depth=8)
            or receipt["original_first_epoch"] != reg["original_first_epoch"]
            or receipt["original_deadline_epoch"] != reg["original_deadline_epoch"]
            or not reg["original_first_epoch"] <= receipt["finished_epoch"]
            <= reg["original_deadline_epoch"] <= reg["operator_end_epoch"]):
        raise ValueError("actual completed same-parent collection-v2 required")
    core = Path(reg["core_repo"])
    if (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=core, text=True).strip()
            != reg["core_commit"]
            or subprocess.check_output(["git", "status", "--porcelain"], cwd=core, text=True)):
        raise ValueError("clean source pin")
    sys.path.insert(0, str(core / "src"))
    import chess
    import torch

    from harbichess.training.cgroup_budget import CgroupMemoryBudget

    torch.set_num_threads(1)
    os.sched_setaffinity(0, {clock["cpu_core"]})
    memory = CgroupMemoryBudget(15 * 2**30)

    def guard():
        memory.check()
        if time.time() >= clock["deadline"]:
            raise TimeoutError("same original600 includes all packet replays")
        if shutil.disk_usage("/workspace").free < 256 * 2**20:
            raise RuntimeError("workspace256MiB floor")

    guard()
    directory = Path(clock["producer_directory"])
    snapshots = {str(x): sha(x) for x in (a.registration, a.receipt, a.events, a.clock)}
    for name, digest in reg["producer_source_sha256"].items():
        path = directory / name
        if sha(path) != digest:
            raise ValueError("actual frozen producer helper differs")
        snapshots[str(path)] = digest
    collector = load(directory / "collector.py", reg["producer_source_sha256"]["collector.py"],
                     "collector")
    producer = load(directory / "run_collection.py",
                    reg["producer_source_sha256"]["run_collection.py"], "own_collection_replay")
    search = load(reg["search_helper"]["path"], reg["search_helper"]["sha256"],
                  "own_collection_original_search")
    ids = receipt["training_row_ids"]
    chosen = [ids[x] for x in ORDINALS]
    if (len(ids) != 1024 or len(set(ids)) != 1024
            or chosen != receipt["periodic_independent_search_rows"]):
        raise ValueError("fixed chronological six-row selection")
    for entry in receipt["alias_chunks"]:
        chunk = a.events.parent / entry["file"]
        if chunk.parent.resolve() != a.events.parent.resolve() or sha(chunk) != entry["sha256"]:
            raise ValueError("all immutable alias chunk pins")
        snapshots[str(chunk)] = entry["sha256"]
    rows = []
    with a.events.open("rb") as f:
        for line in f:
            event = json.loads(line)
            if event["type"] == "search_row":
                rows.append(event["row"])
    packets = []
    selected = selected_rows(rows, ids, chosen)
    if len(rows) != receipt["all_actor_rows"]:
        raise ValueError("exact all raw actor row count")
    evaluator = producer.parent(reg)
    for row_id, row in selected:
        guard()
        if (row["root_fen"] != chess.STARTING_FEN
                or row["history_uci"][:len(row["root_prefix_uci"])] != row["root_prefix_uci"]):
            raise ValueError("full standard-start root prefix binding")
        board = chess.Board(row["root_fen"])
        for uci in row["history_uci"]:
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                raise ValueError("illegal complete history")
            board.push(move)
        if board.outcome(claim_draw=True) or not board.is_valid():
            raise ValueError("terminal/invalid preaction row")
        before = board.fen(), tuple(board.move_stack)
        traced = collector.Traced(evaluator)
        result = search.BudgetSearch(traced, nodes=8192, quiescence_plies=2,
                                     max_depth=8, guard=guard).search(board)
        actual = dict(selected_best_uci=result.move.uci(), value_hex=float(result.value).hex(),
                      nodes=result.nodes, evaluations=result.evaluations,
                      completed_depth=result.completed_depth, root_actions=result.root_actions,
                      actual_eval_calls=traced.calls)
        aliases = alias_segment(row, receipt, a.events.parent)
        check_packet(row, actual, aliases, board)
        if (sorted(traced.seen) != aliases or before != (board.fen(), tuple(board.move_stack))
                or collector.alias(board) != row["root_alias"]):
            raise ValueError("actual complete evaluator trace/history differs")
        packets.append(dict(row_id=row_id, actual_packet=actual,
                            unique_aliases=len(aliases), search_alias_ref=row["search_alias_ref"]))
        guard()
    for path, digest in snapshots.items():
        if sha(path) != digest:
            raise ValueError("immutable source/input changed during replay")
    guard()
    result = dict(schema="NNUE-own-collection-six-root-audit-result-v2",
                  row_identifier_repair="root-id:local-ply-mapping-v2",
                  status="PASS-six-actual-chronological-parent-search-packets-and-full-alias-traces",
                  seed=reg["seed"], registration_sha256=sha(a.registration),
                  receipt_sha256=sha(a.receipt), events_sha256=sha(a.events),
                  helper_sha256=sha(__file__), clock_sha256=sha(a.clock), packets=packets,
                  duplicate_search_nodes=sum(x["actual_packet"]["nodes"] for x in packets),
                  duplicate_static_evaluations=sum(
                      x["actual_packet"]["evaluations"] for x in packets),
                  new_training_rows=0, new_games=0, optimizer_updates=0,
                  first=clock["first"], deadline=clock["deadline"], finished=time.time(),
                  scope="six packets per seed only; all rows require independent converter replay")
    with a.output.open("x") as f:
        json.dump(result, f, sort_keys=True, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())


if __name__ == "__main__":
    main()
