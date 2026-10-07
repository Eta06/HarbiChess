"""Bounded ROOT-run comparison of the baseline and PV-capture search.

The script itself is source-only until ROOT supplies a registration. It never
trains, changes weights, or calls an external engine.
"""

from __future__ import annotations

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

import chess

HERE = Path(__file__).resolve().parent
SCHEMA = "tdleaf-search-equivalence-registration-v1"
RESULT_SCHEMA = "tdleaf-search-equivalence-result-v1"
SEARCH = {"nodes": 8192, "qdepth": 2, "max_depth": 8}
MAX_SECONDS = 600
MAX_POOL_BYTES = 8 * 2**20
MAX_PROTECTED_BYTES = 256 * 2**20
MAX_REPORT_BYTES = 8 * 2**20


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(262144), b""):
            h.update(block)
    return h.hexdigest()


def pinned(ref, cap=MAX_POOL_BYTES) -> Path:
    if set(ref) != {"path", "sha256"}:
        raise ValueError("exact path/SHA ref required")
    path = Path(ref["path"]).resolve(strict=True)
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > cap
        or sha(path) != ref["sha256"]
    ):
        raise ValueError("pinned regular bounded input differs")
    return path


def load(path: Path, digest: str, name: str):
    path = path.resolve(strict=True)
    if sha(path) != digest:
        raise ValueError("source SHA changed: " + path.name)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path:
        raise ImportError("module origin changed")
    return module


def alias(board: chess.Board) -> int:
    raw = min(board.board_fen(), board.mirror().board_fen()).encode("ascii")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little", signed=True)


def replay(row):
    if row.get("role") != "TRAIN" or not isinstance(row.get("root_fen"), str):
        raise ValueError("TRAIN root with full history required")
    history = row.get("prefix_uci")
    if not isinstance(history, list) or not all(isinstance(x, str) for x in history):
        raise ValueError("full UCI history required")
    board = chess.Board(row["root_fen"])
    for token in history:
        move = chess.Move.from_uci(token)
        if move not in board.legal_moves:
            raise ValueError("illegal full root history")
        board.push(move)
    if not board.is_valid() or board.outcome(claim_draw=True) is not None:
        raise ValueError("root must be valid and nonterminal")
    return board


def choose_roots(pool, protected, count=24):
    rows = pool.get("rows")
    if not isinstance(rows, list) or len(rows) != 4096:
        raise ValueError("exact frozen 4096-root source required")
    selected, source_ids, histories = [], set(), set()
    considered = 0
    for row in rows:
        considered += 1
        board = replay(row)
        rid = row.get("root_id")
        history_sha = hashlib.sha256(
            (row["root_fen"] + "\n" + " ".join(row["prefix_uci"])).encode("utf-8")
        ).hexdigest()
        if not isinstance(rid, str) or rid in source_ids or history_sha in histories:
            raise ValueError("duplicate/missing immutable root identity")
        source_ids.add(rid)
        histories.add(history_sha)
        if alias(board) in protected:
            continue
        selected.append(row)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError("fixed source exhausted before 24 legal TRAIN roots")
    return selected, considered


class Trace:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.aliases = []
        self.observed_aliases = []
        self.observed_inputs = []
        self.evaluated_inputs = []
        self.calls = 0
        self.protected_hits = []
        self.observed_protected_inputs = []
        self.protected = set()

    def reset(self):
        self.aliases.clear()
        self.observed_aliases.clear()
        self.observed_inputs.clear()
        self.evaluated_inputs.clear()
        self.calls = 0
        self.protected_hits.clear()
        self.observed_protected_inputs.clear()

    def _check(self, a):
        if a in self.protected:
            self.protected_hits.append(a)

    @staticmethod
    def input_packet(board):
        return {
            "fen": board.fen(),
            "history_uci": [move.uci() for move in board.move_stack],
        }

    def observe(self, board):
        a = alias(board)
        self.observed_aliases.append(a)
        self.observed_inputs.append(self.input_packet(board))
        if a in self.protected:
            self.observed_protected_inputs.append(a)

    def __call__(self, board):
        a = alias(board)
        self.aliases.append(a)
        self._check(a)
        self.calls += 1
        value = float(self.evaluator(board))
        self.evaluated_inputs.append({**self.input_packet(board), "value_hex": value.hex()})
        return value


def collector_aliases(row):
    board = chess.Board(row["root_fen"])
    values = [alias(board)]
    for token in row["prefix_uci"]:
        board.push_uci(token)
        values.append(alias(board))
    return values


def result_payload(result):
    return {
        "move": None if result.move is None else result.move.uci(),
        "value": float(result.value),
        "nodes": int(result.nodes),
        "evaluations": int(result.evaluations),
        "completed_depth": int(result.completed_depth),
        "root_actions": int(result.root_actions),
    }


def verify_pv(root_board, result, trace_aliases):
    if not result.pv or result.move != result.pv[0] or result.leaf.ply != len(result.pv):
        raise ValueError("captured principal variation is incomplete")
    board = root_board.copy(stack=True)
    for move in result.pv:
        if move not in board.legal_moves:
            raise ValueError("captured PV is not legal from full history")
        board.push(move)
    terminal = board.outcome(claim_draw=True)
    if result.leaf.kind == "terminal":
        if terminal is None:
            raise ValueError("terminal PV leaf is not rule-terminal")
    elif result.leaf.kind == "static":
        if terminal is not None or alias(board) not in trace_aliases:
            raise ValueError("static PV leaf missing exact evaluator input")
    else:
        raise ValueError("unknown PV leaf kind")
    return board


def run(registration_path: Path):
    reg_path = registration_path.resolve(strict=True)
    reg = json.loads(reg_path.read_bytes())
    if reg.get("schema") != SCHEMA or reg.get("status") != "registered":
        raise ValueError("ROOT-sealed qualification registration required")
    first, deadline = reg.get("first_epoch"), reg.get("deadline_epoch")
    if (
        type(first) not in (int, float)
        or type(deadline) not in (int, float)
        or deadline <= first
        or deadline - first > MAX_SECONDS
        or not first <= time.time() < deadline
        or reg.get("search") != SEARCH
        or reg.get("root_count") != 24
        or reg.get("teacher_labels_used") is not False
        or reg.get("updates") != 0
    ):
        raise ValueError("fixed scope, budget, and original observed clock required")
    source = reg["source"]
    original_path = pinned(source["baseline_search"])
    pv_path = pinned(source["pv_search"])
    parent_runner_path = pinned(source["parent_runner"])
    if sha(HERE / "search_original.py") != source["baseline_search"]["sha256"]:
        raise ValueError("local baseline search source changed")
    if sha(HERE / "search_pv.py") != source["pv_search"]["sha256"]:
        raise ValueError("local PV search source changed")
    if source["baseline_search"]["sha256"] != reg["search_helper"]["sha256"]:
        raise ValueError("baseline search must match admitted parent search")
    baseline = load(original_path, source["baseline_search"]["sha256"], "tdleaf_baseline_search")
    pvsearch = load(pv_path, source["pv_search"]["sha256"], "tdleaf_pv_search")
    sys.path.insert(0, str(parent_runner_path.parent))
    parent_module = load(
        parent_runner_path,
        source["parent_runner"]["sha256"],
        "tdleaf_parent_runner",
    )

    pool_path = pinned(reg["root_pool"])
    if pool_path.stat().st_size > MAX_POOL_BYTES:
        raise ValueError("root pool bound")
    pool = json.loads(pool_path.read_bytes())
    if pool.get("selection_status") != "pass" or pool.get("train_only") is not True:
        raise ValueError("sealed TRAIN-only pool required")
    protected_path = pinned(reg["protected_aliases"], MAX_PROTECTED_BYTES)
    packed = protected_path.read_bytes()
    if len(packed) % 8:
        raise ValueError("protected aliases must be int64")
    protected_values = [x[0] for x in struct.iter_unpack("<q", packed)]
    if protected_values != sorted(set(protected_values)):
        raise ValueError("protected aliases must be sorted unique")
    protected = set(protected_values)
    roots, considered = choose_roots(pool, protected)

    core = Path(reg["core_repo"]).resolve(strict=True)
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=core, text=True).strip() != reg[
        "core_commit"
    ] or subprocess.check_output(["git", "status", "--porcelain"], cwd=core, text=True):
        raise ValueError("registered clean core source required")
    os.sched_setaffinity(0, {reg["cpu_core"]})
    sys.path.insert(0, str(core / "src"))
    import torch
    from harbichess.training.cgroup_budget import CgroupMemoryBudget

    torch.set_num_threads(1)
    budget = CgroupMemoryBudget(15 * 2**30)

    def guard():
        budget.check()
        if time.time() >= deadline:
            raise TimeoutError("original 600-second parity clock")
        if shutil.disk_usage("/workspace").free < 256 * 2**20:
            raise RuntimeError("workspace disk floor")

    guard()
    evaluator = parent_module.parent(reg)
    baseline_trace, pv_trace = Trace(evaluator.nonterminal), Trace(evaluator.nonterminal)
    baseline_trace.protected = protected
    pv_trace.protected = protected

    rows = []
    baseline_seconds = pv_seconds = 0.0
    for index, root in enumerate(roots):
        guard()
        board = replay(root)
        root_alias = alias(board)
        baseline_trace.reset()
        pv_trace.reset()

        def baseline_call(position):
            return baseline.BudgetSearch(
                baseline_trace,
                nodes=8192,
                quiescence_plies=2,
                max_depth=8,
                guard=guard,
            ).search(position)

        def pv_call(position):
            return pvsearch.BudgetSearch(
                pv_trace,
                input_observer=pv_trace.observe,
                nodes=8192,
                quiescence_plies=2,
                max_depth=8,
                guard=guard,
            ).search(position)

        calls = (baseline_call, pv_call) if index % 2 == 0 else (pv_call, baseline_call)
        outputs = []
        for call in calls:
            start = time.perf_counter()
            outputs.append(call(board))
            elapsed = time.perf_counter() - start
            if calls[0] is baseline_call:
                if len(outputs) == 1:
                    baseline_seconds += elapsed
                else:
                    pv_seconds += elapsed
            else:
                if len(outputs) == 1:
                    pv_seconds += elapsed
                else:
                    baseline_seconds += elapsed
        if index % 2 == 0:
            base_result, pv_result = outputs
        else:
            pv_result, base_result = outputs
        base_trace, pv_aliases = baseline_trace.aliases[:], pv_trace.observed_aliases[:]
        baseline_inputs = baseline_trace.evaluated_inputs[:]
        pv_observed_inputs = pv_trace.observed_inputs[:]
        pv_evaluated_inputs = pv_trace.evaluated_inputs[:]
        exact_base = result_payload(base_result)
        exact_pv = result_payload(pv_result)
        pv_path_aliases = [root_alias]
        pv_path_board = board.copy(stack=True)
        for move in pv_result.pv:
            if move not in pv_path_board.legal_moves:
                raise ValueError("PV full-history legality changed")
            pv_path_board.push(move)
            pv_path_aliases.append(alias(pv_path_board))
        protected_path_hits = sorted(set(pv_path_aliases) & protected)
        equal = all(
            exact_base[k] == exact_pv[k]
            for k in ("move", "value", "nodes", "evaluations", "completed_depth", "root_actions")
        )
        leaf_board = verify_pv(board, pv_result, pv_aliases)
        rows.append(
            {
                "index": index,
                "root_id": root["root_id"],
                "root_fen": root["root_fen"],
                "prefix_uci": root["prefix_uci"],
                "root_alias": root_alias,
                "ancestral_prefix_aliases": collector_aliases(root),
                "ancestral_protected_intersections": sorted(
                    set(collector_aliases(root)) & protected
                ),
                "baseline": exact_base,
                "pv": exact_pv,
                "same_search_result": equal,
                "baseline_ordered_aliases": base_trace,
                "pv_ordered_aliases": pv_aliases,
                "ordered_alias_trace_equal": base_trace == pv_aliases,
                "ordered_full_history_trace_equal": (
                    baseline_inputs == pv_evaluated_inputs
                    and pv_observed_inputs
                    == [
                        {"fen": row["fen"], "history_uci": row["history_uci"]}
                        for row in pv_evaluated_inputs
                    ]
                ),
                "baseline_ordered_input_packets": baseline_inputs,
                "pv_observed_input_histories": pv_observed_inputs,
                "pv_ordered_input_packets": pv_evaluated_inputs,
                "protected_current_root_to_pv_path": protected_path_hits,
                "baseline_protected_inputs": baseline_trace.protected_hits[:],
                "pv_protected_inputs": pv_trace.protected_hits[:],
                "pv_observer_protected_inputs": pv_trace.observed_protected_inputs[:],
                "pv_evaluator_call_aliases": pv_trace.aliases[:],
                "pv_uci": [m.uci() for m in pv_result.pv],
                "leaf": {
                    "kind": pv_result.leaf.kind,
                    "value": float(pv_result.leaf.value),
                    "ply": int(pv_result.leaf.ply),
                    "fen": leaf_board.fen(),
                    "history_uci": [m.uci() for m in leaf_board.move_stack],
                },
            }
        )
        guard()

    exact = all(
        r["same_search_result"]
        and r["ordered_alias_trace_equal"]
        and r["ordered_full_history_trace_equal"]
        for r in rows
    )
    no_protected = all(
        not r["baseline_protected_inputs"] and not r["pv_protected_inputs"] for r in rows
    )
    ratio = pv_seconds / baseline_seconds if baseline_seconds else float("inf")
    report = {
        "schema": RESULT_SCHEMA,
        "status": "PASS" if exact and no_protected and ratio <= 1.10 else "FAIL",
        "registration_sha256": sha(reg_path),
        "source": source,
        "root_pool": reg["root_pool"],
        "protected_aliases": reg["protected_aliases"],
        "first_epoch": first,
        "deadline_epoch": deadline,
        "finished_epoch": time.time(),
        "search": SEARCH,
        "root_count": len(rows),
        "pool_rows_considered": considered,
        "teacher_labels_used": False,
        "updates": 0,
        "baseline_seconds": baseline_seconds,
        "pv_seconds": pv_seconds,
        "pv_to_baseline_wall_ratio": ratio,
        "all_search_results_and_ordered_aliases_equal": exact,
        "no_protected_evaluator_inputs": no_protected,
        "rows": rows,
    }
    raw = (
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    )
    if len(raw) > MAX_REPORT_BYTES or time.time() > deadline:
        raise TimeoutError("report bound/original deadline")
    output = Path(reg["output_path"])
    if output.exists() or not output.is_absolute() or not str(output).startswith("/dev/shm/"):
        raise ValueError("new publish-once RAM output path required")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.registration)
    print(
        json.dumps(
            {k: result[k] for k in ("status", "root_count", "pv_to_baseline_wall_ratio")},
            sort_keys=True,
        )
    )
