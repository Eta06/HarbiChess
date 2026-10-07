"""ROOT-stamped procedural-bank registration only; no generation or model calls."""

import argparse
import time
from pathlib import Path

from root_bank import RECIPE, canonical, ref, sha


def build(a):
    if (
        not a.first
        <= time.time()
        < a.deadline
        <= min(a.first + 600, a.operator_end_epoch, 1791448916.685839)
    ):
        raise ValueError("new prospective bank600 clock")
    return dict(
        schema="human-procedural-root-bank-registration-v2",
        status="registered",
        seed=a.seed,
        first=a.first,
        deadline=a.deadline,
        operator_end_epoch=a.operator_end_epoch,
        recipe=RECIPE,
        generator_sha256=sha(Path(__file__).with_name("root_bank.py")),
        protected_aliases=ref(a.protected_aliases),
        teacher_labels_used=False,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, required=True)
    for name in ("first", "deadline", "operator-end-epoch"):
        p.add_argument("--" + name, type=float, required=True)
    p.add_argument("--protected-aliases", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    with a.output.open("xb") as f:
        f.write(canonical(build(a)))
