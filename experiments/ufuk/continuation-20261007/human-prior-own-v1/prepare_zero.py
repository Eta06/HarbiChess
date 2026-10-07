"""Metadata-only exact initializer seal; ROOT supplies every phase timestamp."""

import argparse
from pathlib import Path

from parent_bridge import END, canonical, pinned, sha


def ref(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=sha(path))


def build(a):
    if not a.first < a.deadline <= min(a.first + 600, a.operator_end_epoch, END):
        raise ValueError("ROOT new600 initialization clock")
    inference = {str(Path(p).resolve()): sha(p) for p in a.inference_file}
    for path, digest in inference.items():
        pinned(dict(path=path, sha256=digest))
    seal = dict(
        schema="human-prior-zero-initialization-seal-v1",
        status="registered",
        seed=a.seed,
        first=a.first,
        deadline=a.deadline,
        operator_end_epoch=a.operator_end_epoch,
        core_repo=str(a.core_repo.resolve()),
        core_commit=a.core_commit,
        prior_helper=ref(a.prior_helper),
        search_helper=ref(a.search_helper),
        inference_source_sha256=inference,
        teacher_labels_used=False,
    )
    if a.search_admission:
        seal["search_admission"] = ref(a.search_admission)
    return seal


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, required=True)
    for name in ("first", "deadline", "operator-end-epoch"):
        p.add_argument("--" + name, type=float, required=True)
    for name in ("core-repo", "prior-helper", "search-helper", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--core-commit", required=True)
    p.add_argument("--inference-file", type=Path, action="append", required=True)
    p.add_argument("--search-admission", type=Path)
    a = p.parse_args()
    seal = build(a)
    with a.output.open("xb") as f:
        f.write(canonical(seal) + b"\n")
