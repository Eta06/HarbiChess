"""ROOT-only fixed24 architectural packet/latency gate, before fit. No games."""

import argparse
import gzip
import json
import statistics
import time
from pathlib import Path

from labels import restore
from search import BASE_SHA, search_type
from support import load, publish, registration


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--registration", type=Path, required=True)
    a = p.parse_args()
    r, out, guard = registration(a.registration)
    if r["deadline_epoch"] > r["first_epoch"] + 600:
        raise ValueError("original architecture600")
    with gzip.open(r["inputs"]["ownQ_labels"]["path"], "rb") as stream:
        raw = stream.read(16 * 2**20 + 1)
    if len(raw) > 16 * 2**20:
        raise ValueError("bounded input")
    rows = sorted(json.loads(raw)["roots"], key=lambda x: x["row_id"])[:12]
    base = load(r["base_search"]["path"], BASE_SHA, "profile_pinned_base")
    value = load(r["prior_value"]["path"], r["prior_value"]["sha256"], "profile_pinned_prior")
    prior = value.load_classical(Path(r["inputs"]["prior"]["path"]))
    if any(prior.theta):
        raise ValueError("prior only")
    selective = search_type(base.BudgetSearch, base.BudgetExhausted)
    receipts = []
    for row in rows:
        for role, kind in [("old-q2", base.BudgetSearch), ("new-zero", selective)]:
            guard()
            b = restore(row["history"])
            started = time.time()
            search = kind(
                prior.nonterminal, nodes=512, quiescence_plies=2, max_depth=8, guard=guard
            )
            packet = search.search(b)
            receipts.append(
                dict(
                    row_id=row["row_id"],
                    role=role,
                    wall=time.time() - started,
                    nodes=packet.nodes,
                    evaluations=packet.evaluations,
                    depth=packet.completed_depth,
                    move=packet.move.uci(),
                    extensions=getattr(search, "extension_receipts", []),
                )
            )
    ratio = statistics.median(
        x["wall"] for x in receipts if x["role"] == "new-zero"
    ) / statistics.median(x["wall"] for x in receipts if x["role"] == "old-q2")
    guard()
    publish(
        out / "profile.json",
        dict(
            status="PASS-architecture-only-not-strength"
            if ratio <= 1.10
            else "INCOMPLETE-architecture-latency",
            median_ratio=ratio,
            packets=receipts,
            finished=time.time(),
            deadline=r["deadline_epoch"],
        ),
    )


if __name__ == "__main__":
    main()
