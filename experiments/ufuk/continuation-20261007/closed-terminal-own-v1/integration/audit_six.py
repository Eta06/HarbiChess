"""Independent six-row re-search audit for the closed-terminal MC producer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
import time
from pathlib import Path

import chess

ORDINALS = (0, 204, 409, 614, 819, 1023)
AUDIT_SCHEMA = "NNUE-own-closed-terminal-six-search-audit-v1"
AUDIT_STATUS = (
    "PASS-six-actual-chronological-terminal-eligible-search-packets-and-complete-alias-traces"
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(ref, name):
    path = Path(ref["path"]).resolve()
    if sha(path) != ref["sha256"]:
        raise ValueError("pinned source/input changed")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path:
        raise ValueError("loaded module origin changed")
    return module


def read_aliases(row, spec, receipt):
    ref = row["search_alias_ref"]
    entry = next((x for x in receipt["alias_chunks"] if x["file"] == ref["file"]), None)
    if entry is None or ref["file"] not in spec["alias_chunks"]:
        raise ValueError("unregistered alias chunk")
    chunk_ref = spec["alias_chunks"][ref["file"]]
    path = Path(chunk_ref["path"]).resolve()
    if (
        path.name != ref["file"]
        or path.stat().st_size != entry["bytes"]
        or sha(path) != entry["sha256"]
    ):
        raise ValueError("alias chunk bytes/SHA mismatch")
    if chunk_ref["sha256"] != entry["sha256"]:
        raise ValueError("spec/receipt alias SHA mismatch")
    offset, count = ref["offset_bytes"], ref["count"]
    if (
        type(offset) is not int
        or type(count) is not int
        or offset < 0
        or count < 0
        or offset % 8
        or offset + count * 8 > entry["bytes"]
    ):
        raise ValueError("alias segment bounds")
    with path.open("rb") as stream:
        stream.seek(offset)
        raw = stream.read(count * 8)
    values = list(struct.unpack(f"<{count}q", raw))
    if values != sorted(set(values)):
        raise ValueError("alias trace is not sorted and unique")
    return values


def selected_six(events_path, receipt):
    ids = receipt["training_row_ids"]
    prescribed = [ids[i] for i in ORDINALS]
    if (
        len(ids) != 1024
        or len(set(ids)) != 1024
        or receipt["periodic_independent_search_rows"] != prescribed
    ):
        raise ValueError("fixed six chronological eligible MC row IDs")
    seen, mapping, ordered = set(), {}, []
    for line in Path(events_path).read_bytes().splitlines():
        event = json.loads(line)
        if event.get("type") != "search_row":
            continue
        row = event["row"]
        row_id = f"{row['root_id']}:{row['local_ply']}"
        if row_id in seen:
            raise ValueError("duplicate root:ply event")
        seen.add(row_id)
        mapping[row_id] = row
        ordered.append(row_id)
    if [x for x in ordered if x in set(ids)] != ids:
        raise ValueError("eligible IDs must occur in receipt chronology")
    if any(x not in mapping for x in prescribed):
        raise ValueError("selected chronological MC row absent")
    return [(row_id, mapping[row_id]) for row_id in prescribed]


def replay_row(row):
    if (
        row["root_fen"] != chess.STARTING_FEN
        or not row["history_uci"][: len(row["root_prefix_uci"])] == row["root_prefix_uci"]
    ):
        raise ValueError("full standard-start root history")
    board = chess.Board(row["root_fen"])
    for token in row["history_uci"]:
        move = chess.Move.from_uci(token)
        if move not in board.legal_moves:
            raise ValueError("illegal full history")
        board.push(move)
    if not board.is_valid() or board.outcome(claim_draw=True) is not None:
        raise ValueError("selected row must be nonterminal pre-action state")
    return board


def packet_for(row, result, calls):
    return {
        "selected_best_uci": result.move.uci(),
        "value_hex": float(result.value).hex(),
        "nodes": result.nodes,
        "evaluations": result.evaluations,
        "completed_depth": result.completed_depth,
        "root_actions": result.root_actions,
        "actual_eval_calls": calls,
    }


def audit_six(
    spec, receipt, *, converter, collector, producer, search, evaluator, clock, guard=lambda: None
):
    """Full converter validation plus six independent same-parent searches."""
    sealed_receipt = json.loads(Path(spec["receipt"]["path"]).read_bytes())
    if sealed_receipt != receipt:
        raise ValueError("audit receipt object differs from the pinned receipt file")
    converted, provenance = converter.convert(spec, guard)
    parsed = json.loads(provenance)
    if (
        parsed.get("schema") != "NNUE-own1024-closed-terminal-data-provenance-v1"
        or parsed.get("collection_receipt_sha256") != spec["receipt"]["sha256"]
    ):
        raise ValueError("closed-terminal conversion provenance mismatch")
    selected = selected_six(spec["events"]["path"], receipt)
    results = []
    for row_id, row in selected:
        guard()
        board = replay_row(row)
        before = board.fen(), tuple(board.move_stack)
        aliases = read_aliases(row, spec, receipt)
        traced = collector.Traced(evaluator)
        replay = search.BudgetSearch(
            traced, nodes=8192, quiescence_plies=2, max_depth=8, guard=guard
        ).search(board)
        actual = packet_for(row, replay, traced.calls)
        expected = {key: row[key] for key in actual if key != "value_hex"}
        expected["value_hex"] = float(row["raw_q_mover"]).hex()
        if actual != expected:
            raise ValueError("independent selected packet differs from recorded actual search")
        if chess.Move.from_uci(row["selected_best_uci"]) not in board.legal_moves:
            raise ValueError("selected best action is illegal on audited history")
        if (
            sorted(traced.seen) != aliases
            or collector.alias(board) != row["root_alias"]
            or before != (board.fen(), tuple(board.move_stack))
        ):
            raise ValueError("fullhistory/evaluator alias replay differs")
        if (
            row["fen4"] != " ".join(board.fen().split()[:4])
            or row["mover"] != ("white" if board.turn else "black")
            or row["root_actions"] != board.legal_moves.count()
        ):
            raise ValueError("mover/FEN/legality packet mismatch")
        results.append(
            {
                "row_id": row_id,
                "actual_packet": actual,
                "search_alias_ref": row["search_alias_ref"],
                "unique_aliases": len(aliases),
            }
        )
    return {
        "schema": AUDIT_SCHEMA,
        "status": AUDIT_STATUS,
        "seed": receipt["seed"],
        "registration_sha256": spec["registration"]["sha256"],
        "receipt_sha256": spec["receipt"]["sha256"],
        "events_sha256": spec["events"]["sha256"],
        "clock_sha256": clock["sha256"],
        "helper_sha256": sha(__file__),
        "first": clock["first"],
        "deadline": clock["deadline"],
        "finished": time.time(),
        "converted_dataset_sha256": hashlib.sha256(converted).hexdigest(),
        "converted_provenance_sha256": hashlib.sha256(provenance).hexdigest(),
        "packets": results,
        "new_training_rows": 0,
        "new_games": 0,
        "optimizer_updates": 0,
        "scope": "six chronological rows; complete converter replay",
    }


def _load_actual(spec):
    reg = json.loads(Path(spec["registration"]["path"]).read_bytes())
    directory = Path(spec["producer_directory"]).resolve()
    source = reg["producer_source_sha256"]
    sys.path.insert(0, str(directory))
    collector_path = directory / "collector.py"
    collector = load(
        {"path": str(collector_path), "sha256": source["collector.py"]},
        "closed_terminal_audit_collector",
    )
    sys.modules.pop("collector", None)
    runner_path = directory / "run_collection.py"
    runner = load(
        {"path": str(runner_path), "sha256": source["run_collection.py"]},
        "closed_terminal_audit_runner",
    )
    search = load(reg["search_helper"], "closed_terminal_audit_search")
    converter = load(
        {"path": str(directory / "convert.py"), "sha256": sha(directory / "convert.py")},
        "closed_terminal_audit_converter",
    )
    evaluator = runner.parent(reg).nonterminal
    return reg, collector, runner, search, converter, evaluator


def main():
    import argparse
    import os
    import shutil
    import time

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--clock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_bytes())
    clock = json.loads(args.clock.read_bytes())
    reg, collector, runner, search, converter, evaluator = _load_actual(spec)
    if (
        clock.get("schema") != "NNUE-own-closed-terminal-six-search-audit-clock-v1"
        or clock.get("status") != "registered"
        or clock.get("helper_sha256") != sha(__file__)
        or clock.get("spec_sha256") != sha(args.spec)
        or clock.get("registration_sha256") != sha(spec["registration"]["path"])
        or clock.get("receipt_sha256") != sha(spec["receipt"]["path"])
        or clock.get("events_sha256") != sha(spec["events"]["path"])
        or clock.get("converter_sha256") != sha(Path(spec["producer_directory"]) / "convert.py")
        or not clock["first"] <= time.time() < clock["deadline"]
        or clock["deadline"] > min(clock["first"] + 600, reg["operator_end_epoch"])
    ):
        raise ValueError("ROOT registered immutable six-packet audit clock and pins required")
    if args.output.exists() or not args.output.resolve().is_relative_to("/dev/shm"):
        raise ValueError("publish-once RAM audit result required")
    os.sched_setaffinity(0, {clock["cpu_core"]})
    import torch

    torch.set_num_threads(1)
    core = Path(reg["core_repo"]).resolve()
    sys.path.insert(0, str(core / "src"))
    from harbichess.training.cgroup_budget import CgroupMemoryBudget

    memory = CgroupMemoryBudget(15 * 2**30)

    def guard():
        memory.check()
        if time.time() >= clock["deadline"]:
            raise TimeoutError("same original audit clock")
        if shutil.disk_usage("/workspace").free < 256 * 2**20:
            raise RuntimeError("workspace disk floor")

    guard()
    report = audit_six(
        spec,
        json.loads(Path(spec["receipt"]["path"]).read_bytes()),
        converter=converter,
        collector=collector,
        producer=runner,
        search=search,
        evaluator=evaluator,
        clock={"sha256": sha(args.clock), "first": clock["first"], "deadline": clock["deadline"]},
        guard=guard,
    )
    guard()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(
            json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            + b"\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == "__main__":
    main()
