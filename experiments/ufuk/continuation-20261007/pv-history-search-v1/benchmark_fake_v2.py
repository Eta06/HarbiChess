"""Bounded synthetic fake-evaluator timing, not strength or real-model latency."""

import dataclasses
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import chess
import search
import search_v2
from test_search import static

old_path = Path("/workspace/HarbiChess/experiments/ufuk/cpu-budget-search-v1/search.py")
spec = importlib.util.spec_from_file_location("fake_bench_original", old_path)
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)
b = chess.Board()
# Outcomeblind deterministic legal synthetic history; stop before any terminal.
for _ in range(48):
    for move in sorted(b.legal_moves, key=lambda m: m.uci()):
        b.push(move)
        if b.outcome(claim_draw=True) is None:
            break
        b.pop()
    else:
        break
boards = [chess.Board(), chess.Board("8/8/8/8/8/2k5/7R/6K1 w - - 0 1"), b]
rows = []
start = time.time()
for repeat in range(3):
    for board in boards:
        row = {
            "repeat": repeat,
            "history_plies": len(board.move_stack),
            "packets": {},
            "seconds": {},
        }
        order = [("v1", search), ("v2", search_v2), ("old", old)]
        if repeat % 2:
            order.reverse()
        for name, module in order:
            first = time.perf_counter()
            result = module.BudgetSearch(static).search(board)
            row["seconds"][name] = time.perf_counter() - first
            packet = dataclasses.asdict(result)
            packet["move"] = result.move.uci()
            row["packets"][name] = packet
        assert row["packets"]["v1"] == row["packets"]["v2"]
        rows.append(row)
result = dict(
    schema="pv-history-v2-fake-whole-search-timing-v1",
    scope="synthetic fake statics only, no model/teacher/strength claim",
    first=start,
    finished=time.time(),
    rows=rows,
    median_v2_over_v1=statistics.median(r["seconds"]["v2"] / r["seconds"]["v1"] for r in rows),
    median_v2_over_old=statistics.median(r["seconds"]["v2"] / r["seconds"]["old"] for r in rows),
)
print(json.dumps(result, sort_keys=True))
