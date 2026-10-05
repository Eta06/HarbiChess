"""One-CPU static evaluation only; no search, selfplay, fitting or matches."""

import hashlib
import json
import os
import time
from pathlib import Path

import chess
from value import ClassicalValue, features

os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
boards = []
b = chess.Board()
for u in (
    "e2e4",
    "e7e5",
    "g1f3",
    "b8c6",
    "f1b5",
    "a7a6",
    "b5a4",
    "g8f6",
    "e1g1",
    "f8e7",
    "f1e1",
    "b7b5",
    "a4b3",
    "d7d6",
    "c2c3",
    "e8g8",
):
    b.push_uci(u)
    boards.append(b.copy(stack=True))
boards += [
    chess.Board("4k3/8/8/2p5/2P5/3P4/8/4K3 w - - 0 1"),
    chess.Board("4k3/8/2n5/8/3N4/8/8/4K3 b - - 0 1"),
]
v = ClassicalValue()
n = 1800
report = {}
for name, fn in [
    ("features", features),
    ("nonterminal_scalar", v.nonterminal),
    ("exact_terminal_checked_scalar", v),
]:
    start = time.perf_counter()
    for i in range(n):
        fn(boards[i % len(boards)])
    elapsed = time.perf_counter() - start
    report[name] = {
        "evaluations": n,
        "seconds": elapsed,
        "microseconds_per_eval": elapsed / n * 1e6,
    }
report.update(
    scope=(
        "Static eval only18fixed full-history boards. "
        "No NN comparative benchmark, no search performance/strength claim."
    ),
    cpu_affinity=sorted(os.sched_getaffinity(0)),
    helper_sha256=hashlib.sha256(Path(__file__).with_name("value.py").read_bytes()).hexdigest(),
    max_static_evals_8192_actions=4194304,
    max_static_evals_32768_actions=16777216,
)
Path(__file__).with_name("microbench-result.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
