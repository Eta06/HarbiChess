"""ROOT-only fixed24 architectural packet/latency gate, before fit. No games."""

import argparse
import gzip
import json
import statistics
import sys
import time
from pathlib import Path

from support import load, publish, registration

OLD = Path("/workspace/work/harbichess/cpu-selective-quiescence-proposal")
sys.path.insert(0, str(OLD))
try:
    labels = load(
        OLD / "labels.py",
        "1d21a8885f3dc19f6338bc9b7b6470cbb29fc83e76ba9b2d7f30fd32351b6fd4",
        "balanced_labels",
    )
    search_module = load(
        OLD / "search.py",
        "993e9c2c6d949b3d8753935405b5c289b35249a108b0ef3f8f984ac55de94293",
        "balanced_arch_search",
    )
finally:
    sys.path.pop(0)
restore, BASE_SHA, search_type = labels.restore, search_module.BASE_SHA, search_module.search_type


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
