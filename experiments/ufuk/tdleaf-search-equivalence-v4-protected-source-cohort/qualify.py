"""Bounded ROOT-run comparison of the baseline and PV-capture search.

The script itself is source-only until ROOT supplies a registration. It never
trains, changes weights, or calls an external engine.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path

import chess

HERE = Path(__file__).resolve().parent
SCHEMA = "tdleaf-search-equivalence-registration-v4"
RESULT_SCHEMA = "tdleaf-search-equivalence-result-v4"
PACKET_SCHEMA = "tdleaf-search-query-packet-v4"
SEARCH = {"nodes": 8192, "qdepth": 2, "max_depth": 8}
MAX_SECONDS = 600
MAX_POOL_BYTES = 8 * 2**20
MAX_PROTECTED_BYTES = 256 * 2**20
MAX_REPORT_BYTES = 8 * 2**20
MAX_PACKET_RAW_BYTES = 32 * 2**20
MAX_PACKET_BYTES = 8 * 2**20
MAX_PACKET_TOTAL_BYTES = 128 * 2**20


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(262144), b""):
            h.update(block)
    return h.hexdigest()


def packet_bytes(packet: dict) -> bytes:
    raw = (
        json.dumps(
            packet, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        + b"\n"
    )
    if len(raw) > MAX_PACKET_RAW_BYTES:
        raise ValueError("uncompressed query packet exceeds per-root bound")
    packed = gzip.compress(raw, compresslevel=6, mtime=0)
    if len(packed) > MAX_PACKET_BYTES:
        raise ValueError("compressed query packet exceeds per-root bound")
    return packed


def decode_packet_bytes(packed: bytes) -> dict:
    if not packed or len(packed) > MAX_PACKET_BYTES:
        raise ValueError("compressed query packet size invalid")
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    raw = decoder.decompress(packed, MAX_PACKET_RAW_BYTES + 1)
    if len(raw) > MAX_PACKET_RAW_BYTES or decoder.unconsumed_tail:
        raise ValueError("query packet decompression exceeds bound")
    raw += decoder.flush(MAX_PACKET_RAW_BYTES + 1 - len(raw))
    if (
        len(raw) > MAX_PACKET_RAW_BYTES
        or not decoder.eof
        or decoder.unused_data
        or decoder.unconsumed_tail
    ):
        raise ValueError("truncated, concatenated, or trailing query packet data")
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("schema") != PACKET_SCHEMA:
        raise ValueError("query packet schema mismatch")
    return value


def write_packet(
    packet_dir: Path, index: int, packet: dict, total: int
) -> tuple[dict, int]:
    packed = packet_bytes(packet)
    total += len(packed)
    if total > MAX_PACKET_TOTAL_BYTES:
        raise ValueError("aggregate query packet bound exceeded")
    path = packet_dir / f"root-{index:02d}.json.gz"
    if path.exists() or path.is_symlink():
        raise ValueError("publish-once query packet path already exists")
    with path.open("xb") as stream:
        stream.write(packed)
        stream.flush()
        os.fsync(stream.fileno())
    ref = {
        "path": str(path),
        "bytes": len(packed),
        "sha256": hashlib.sha256(packed).hexdigest(),
    }
    return ref, total


def read_packet_ref(packet_dir: Path, ref: dict, index: int) -> dict:
    if set(ref) != {"path", "bytes", "sha256"}:
        raise ValueError("query packet reference shape mismatch")
    expected = packet_dir / f"root-{index:02d}.json.gz"
    path = Path(ref["path"])
    if (
        path != expected
        or path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > MAX_PACKET_BYTES
    ):
        raise ValueError("query packet path differs from fixed publish-once name")
    packed = path.read_bytes()
    if (
        len(packed) != ref["bytes"]
        or hashlib.sha256(packed).hexdigest() != ref["sha256"]
    ):
        raise ValueError("query packet hash/length mismatch")
    return decode_packet_bytes(packed)


def packet_matches_summary(packet: dict, summary: dict) -> bool:
    return (
        packet.get("index") == summary.get("index")
        and packet.get("root_id") == summary.get("root_id")
        and packet.get("baseline_result") == summary.get("baseline")
        and packet.get("pv_result") == summary.get("pv")
        and packet.get("baseline_ordered_query_packets")
        == packet.get("pv_ordered_query_packets")
        and packet.get("baseline_observed_input_histories")
        == [
            {"fen": item["fen"], "history_uci": item["history_uci"]}
            for item in packet.get("baseline_ordered_query_packets", [])
        ]
        and packet.get("baseline_ordered_aliases")
        == packet.get("baseline_observed_aliases")
        and packet.get("baseline_ordered_aliases") == packet.get("pv_observed_aliases")
        and packet.get("pv_observed_aliases") == packet.get("pv_evaluator_call_aliases")
        and packet.get("pv_observed_input_histories")
        == [
            {"fen": item["fen"], "history_uci": item["history_uci"]}
            for item in packet.get("pv_ordered_query_packets", [])
        ]
        and packet.get("current_root_to_pv_protected_hits")
        == summary.get("protected_current_root_to_pv_path")
    )


def packet_queries_clear(packet: dict, protected: set[int]) -> bool:
    return (
        not packet.get("baseline_protected_alias_hits")
        and not packet.get("pv_protected_alias_hits")
        and not packet.get("pv_observer_protected_alias_hits")
        and not (set(packet.get("baseline_ordered_aliases", [])) & protected)
        and not (set(packet.get("pv_observed_aliases", [])) & protected)
        and not (set(packet.get("pv_evaluator_call_aliases", [])) & protected)
    )


def packet_path_clear(packet: dict, protected: set[int]) -> bool:
    path_aliases = packet.get("current_root_to_pv_aliases")
    if not isinstance(path_aliases, list):
        return False
    return set(path_aliases) & protected == set(
        packet.get("current_root_to_pv_protected_hits", [])
    ) and not packet.get("current_root_to_pv_protected_hits")


def packet_protection_clear(packet: dict, protected: set[int]) -> bool:
    return packet_queries_clear(packet, protected) and packet_path_clear(
        packet, protected
    )


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
        self.evaluated_inputs.append(
            {**self.input_packet(board), "value_hex": value.hex()}
        )
        return value


def collector_aliases(row):
    board = chess.Board(row["root_fen"])
    values = [alias(board)]
    for token in row["prefix_uci"]:
        board.push_uci(token)
        values.append(alias(board))
    return values


def result_payload(result):
    value = float(result.value)
    if not math.isfinite(value):
        raise ValueError("search result value must be finite")
    return {
        "move": None if result.move is None else result.move.uci(),
        "value": value,
        "value_hex": value.hex(),
        "nodes": int(result.nodes),
        "evaluations": int(result.evaluations),
        "completed_depth": int(result.completed_depth),
        "root_actions": int(result.root_actions),
    }


def same_search_result(left, right) -> bool:
    return all(
        left[key] == right[key]
        for key in (
            "move",
            "value_hex",
            "nodes",
            "evaluations",
            "completed_depth",
            "root_actions",
        )
    )


def verify_pv(root_board, result, trace_aliases):
    if (
        not result.pv
        or result.move != result.pv[0]
        or result.leaf.ply != len(result.pv)
    ):
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
        or reg.get("max_attempted_roots") != 128
        or reg.get("selection_rule")
        != "first24-source-order-current-query-PV-protected-clear-v1"
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
    baseline = load(
        original_path, source["baseline_search"]["sha256"], "tdleaf_baseline_search"
    )
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
    roots, considered = choose_roots(pool, protected, count=128)

    core = Path(reg["core_repo"]).resolve(strict=True)
    if subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=core, text=True
    ).strip() != reg["core_commit"] or subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=core, text=True
    ):
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
    baseline_trace, pv_trace = (
        Trace(evaluator.nonterminal),
        Trace(evaluator.nonterminal),
    )
    baseline_trace.protected = protected
    pv_trace.protected = protected

    rows = []
    packet_refs = []
    packet_dir = Path(reg["packet_dir"])
    output_path = Path(reg["output_path"])
    if (
        not packet_dir.is_absolute()
        or not str(packet_dir).startswith("/dev/shm/")
        or packet_dir.exists()
        or packet_dir.is_symlink()
        or output_path.resolve(strict=False).is_relative_to(
            packet_dir.resolve(strict=False)
        )
    ):
        raise ValueError("new publish-once RAM packet directory required")
    packet_dir.mkdir(parents=True, exist_ok=False)
    traced_baseline_seconds = traced_pv_seconds = 0.0
    untraced_baseline_seconds = untraced_pv_seconds = 0.0
    packet_total_bytes = 0
    for index, root in enumerate(roots):
        guard()
        board = replay(root)
        root_alias = alias(board)
        baseline_trace.reset()
        pv_trace.reset()

        def traced_baseline_evaluator(position):
            baseline_trace.observe(position)
            return baseline_trace(position)

        def baseline_call(position):
            return baseline.BudgetSearch(
                traced_baseline_evaluator,
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
                    traced_baseline_seconds += elapsed
                else:
                    traced_pv_seconds += elapsed
            else:
                if len(outputs) == 1:
                    traced_pv_seconds += elapsed
                else:
                    traced_baseline_seconds += elapsed
        if index % 2 == 0:
            base_result, pv_result = outputs
        else:
            pv_result, base_result = outputs
        base_trace, pv_aliases = baseline_trace.aliases[:], pv_trace.observed_aliases[:]
        baseline_inputs = baseline_trace.evaluated_inputs[:]
        baseline_observed_inputs = baseline_trace.observed_inputs[:]
        baseline_observed_aliases = baseline_trace.observed_aliases[:]
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
        equal = same_search_result(exact_base, exact_pv)
        leaf_board = verify_pv(board, pv_result, pv_aliases)

        def untraced_baseline_call(position):
            return baseline.BudgetSearch(
                evaluator,
                nodes=8192,
                quiescence_plies=2,
                max_depth=8,
                guard=guard,
            ).search(position)

        def untraced_pv_call(position):
            return pvsearch.BudgetSearch(
                evaluator,
                nodes=8192,
                quiescence_plies=2,
                max_depth=8,
                guard=guard,
            ).search(position)

        untraced_calls = (
            (untraced_pv_call, untraced_baseline_call)
            if index % 2 == 0
            else (untraced_baseline_call, untraced_pv_call)
        )
        untraced_outputs = []
        for call in untraced_calls:
            start = time.perf_counter()
            untraced_outputs.append(call(board))
            elapsed = time.perf_counter() - start
            if call is untraced_baseline_call:
                untraced_baseline_seconds += elapsed
            else:
                untraced_pv_seconds += elapsed
        if untraced_calls[0] is untraced_baseline_call:
            untraced_base_result, untraced_pv_result = untraced_outputs
        else:
            untraced_pv_result, untraced_base_result = untraced_outputs
        untraced_base_payload = result_payload(untraced_base_result)
        untraced_pv_payload = result_payload(untraced_pv_result)
        untraced_equal = (
            same_search_result(untraced_base_payload, untraced_pv_payload)
            and untraced_base_payload == exact_base
            and untraced_pv_payload == exact_pv
        )
        if not untraced_equal:
            raise ValueError("untraced paired search outputs differ")

        packet = {
            "schema": PACKET_SCHEMA,
            "index": index,
            "root_id": root["root_id"],
            "baseline_result": exact_base,
            "pv_result": exact_pv,
            "baseline_ordered_query_packets": baseline_inputs,
            "baseline_observed_input_histories": baseline_observed_inputs,
            "baseline_observed_aliases": baseline_observed_aliases,
            "pv_observed_input_histories": pv_observed_inputs,
            "pv_ordered_query_packets": pv_evaluated_inputs,
            "baseline_ordered_aliases": base_trace,
            "pv_observed_aliases": pv_aliases,
            "pv_evaluator_call_aliases": pv_trace.aliases[:],
            "baseline_protected_alias_hits": baseline_trace.protected_hits[:],
            "pv_protected_alias_hits": pv_trace.protected_hits[:],
            "pv_observer_protected_alias_hits": pv_trace.observed_protected_inputs[:],
            "current_root_to_pv_aliases": pv_path_aliases,
            "current_root_to_pv_protected_hits": protected_path_hits,
        }
        packet_ref, packet_total_bytes = write_packet(
            packet_dir, index, packet, packet_total_bytes
        )
        packet_refs.append(packet_ref)
        rows.append(
            {
                "accepted_for_protected_cohort": packet_protection_clear(
                    packet, protected
                ),
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
                "untraced_baseline": untraced_base_payload,
                "untraced_pv": untraced_pv_payload,
                "untraced_same_search_result": untraced_equal,
                "baseline_evaluator_calls": len(base_trace),
                "pv_observer_calls": len(pv_aliases),
                "ordered_alias_trace_equal": base_trace == pv_aliases,
                "ordered_full_history_trace_equal": (
                    baseline_inputs == pv_evaluated_inputs
                    and baseline_observed_inputs
                    == [
                        {"fen": row["fen"], "history_uci": row["history_uci"]}
                        for row in baseline_inputs
                    ]
                    and baseline_observed_aliases == base_trace
                    and pv_observed_inputs
                    == [
                        {"fen": row["fen"], "history_uci": row["history_uci"]}
                        for row in pv_evaluated_inputs
                    ]
                ),
                "protected_current_root_to_pv_path": protected_path_hits,
                "baseline_protected_input_count": len(baseline_trace.protected_hits),
                "pv_protected_input_count": len(pv_trace.protected_hits),
                "pv_observer_protected_input_count": len(
                    pv_trace.observed_protected_inputs
                ),
                "query_packet_index": index,
                "pv_uci": [m.uci() for m in pv_result.pv],
                "leaf": {
                    "kind": pv_result.leaf.kind,
                    "value": float(pv_result.leaf.value),
                    "value_hex": float(pv_result.leaf.value).hex(),
                    "ply": int(pv_result.leaf.ply),
                    "fen": leaf_board.fen(),
                    "history_uci": [m.uci() for m in leaf_board.move_stack],
                },
            }
        )
        guard()
        if sum(r["accepted_for_protected_cohort"] for r in rows) == 24:
            break

    accepted = [r for r in rows if r["accepted_for_protected_cohort"]]
    exact = all(
        r["same_search_result"]
        and r["ordered_alias_trace_equal"]
        and r["ordered_full_history_trace_equal"]
        for r in rows
    )
    no_protected_queries = all(
        r["baseline_protected_input_count"] == 0
        and r["pv_protected_input_count"] == 0
        and r["pv_observer_protected_input_count"] == 0
        for r in accepted
    )
    no_protected_paths = all(
        not r["protected_current_root_to_pv_path"] for r in accepted
    )
    packet_integrity = len(packet_refs) == len(rows) and len(accepted) == 24
    for index, ref in enumerate(packet_refs):
        packet = read_packet_ref(packet_dir, ref, index)
        summary = rows[index]
        if not packet_matches_summary(packet, summary):
            packet_integrity = False
        actual_clear = packet_protection_clear(packet, protected)
        if actual_clear != summary["accepted_for_protected_cohort"]:
            packet_integrity = False
        if summary["accepted_for_protected_cohort"]:
            no_protected_queries = no_protected_queries and packet_queries_clear(
                packet, protected
            )
            no_protected_paths = no_protected_paths and packet_path_clear(
                packet, protected
            )
    traced_ratio = (
        traced_pv_seconds / traced_baseline_seconds
        if traced_baseline_seconds
        else float("inf")
    )
    untraced_ratio = (
        untraced_pv_seconds / untraced_baseline_seconds
        if untraced_baseline_seconds
        else float("inf")
    )
    report = {
        "schema": RESULT_SCHEMA,
        "status": "PASS"
        if (
            exact
            and no_protected_queries
            and no_protected_paths
            and packet_integrity
            and all(r["untraced_same_search_result"] for r in rows)
            and traced_ratio <= 1.10
            and untraced_ratio <= 1.10
        )
        else "FAIL",
        "registration_sha256": sha(reg_path),
        "source": source,
        "root_pool": reg["root_pool"],
        "protected_aliases": reg["protected_aliases"],
        "first_epoch": first,
        "deadline_epoch": deadline,
        "finished_epoch": time.time(),
        "search": SEARCH,
        "root_count": len(accepted),
        "attempted_root_count": len(rows),
        "rejected_root_count": len(rows) - len(accepted),
        "selection_rule": reg["selection_rule"],
        "all_attempts_in_search_equality_and_wall": True,
        "pool_rows_considered": considered,
        "teacher_labels_used": False,
        "updates": 0,
        "traced_baseline_seconds": traced_baseline_seconds,
        "traced_pv_seconds": traced_pv_seconds,
        "traced_pv_to_baseline_wall_ratio": traced_ratio,
        "untraced_baseline_seconds": untraced_baseline_seconds,
        "untraced_pv_seconds": untraced_pv_seconds,
        "untraced_pv_to_baseline_wall_ratio": untraced_ratio,
        "all_search_results_and_ordered_aliases_equal": exact,
        "no_protected_evaluator_inputs": no_protected_queries,
        "no_protected_root_to_pv_path": no_protected_paths,
        "query_packet_directory": str(packet_dir),
        "query_packet_count": len(packet_refs),
        "query_packet_total_bytes": packet_total_bytes,
        "query_packet_integrity": packet_integrity,
        "query_packets": packet_refs,
        "rows": rows,
    }
    raw = (
        json.dumps(
            report, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        + b"\n"
    )
    if len(raw) > MAX_REPORT_BYTES or time.time() > deadline:
        raise TimeoutError("report bound/original deadline")
    output = Path(reg["output_path"])
    if (
        output.exists()
        or not output.is_absolute()
        or not str(output).startswith("/dev/shm/")
    ):
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
            {
                k: result[k]
                for k in (
                    "status",
                    "root_count",
                    "traced_pv_to_baseline_wall_ratio",
                    "untraced_pv_to_baseline_wall_ratio",
                )
            },
            sort_keys=True,
        )
    )
