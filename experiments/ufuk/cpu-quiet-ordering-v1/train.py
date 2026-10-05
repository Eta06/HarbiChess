"""ROOT-only fixed-final quiet ranker. No actor or automatic selection."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from data import prepare_ordering
from model import model_dict
from ordering_learner import MATH, Learner


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["protocol", "config", "journal", "output", "source-repo"]:
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--protocol-sha256", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--cpu-core", type=int, required=True)
    p.add_argument("--deadline", type=float, required=True)
    p.add_argument("--stop-at", type=int)
    p.add_argument("--resume", type=Path)
    p.add_argument("--resume-sha256")
    p.add_argument("--audit-only", action="store_true")
    a = p.parse_args()
    if sha(a.protocol) != a.protocol_sha256:
        raise ValueError("immutable protocol SHA differs")
    q = json.loads(a.protocol.read_text())
    if q["schema"] != "own-quiet-ordering-training-protocol-v1" or q["math"] != MATH:
        raise ValueError("fixed learning schema/math differs")
    if a.deadline != q["deadline_by_seed"][str(a.seed)]:
        raise ValueError("original clock differs")
    if set(q["ordering_helper_sha256"]) != {
        "train.py",
        "data.py",
        "model.py",
        "ordering_learner.py",
        "ordered_search.py",
        "qualify.py",
    }:
        raise ValueError("exact entire ordering helper closure required")
    for name, expected in q["ordering_helper_sha256"].items():
        if sha(Path(__file__).with_name(name)) != expected:
            raise ValueError("ordering helper differs")
    old = Path(q["classical_helper_directory"])
    for name, expected in q["classical_helper_sha256"].items():
        if sha(old / name) != expected:
            raise ValueError("immutable classical dependency differs")
    sys.path.append(str(old))
    import subprocess

    from runtime import affinity, guard
    from train_v3 import publish

    if subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=a.source_repo, text=True
    ).strip() != q["source_commit"] or subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=a.source_repo, text=True
    ):
        raise ValueError("clean source differs")
    affinity(a.cpu_core)
    c = json.loads(a.config.read_text())
    if c["seed"] != a.seed or c["source_commit"] != q["source_commit"]:
        raise ValueError("source/same seed differs")
    pair = q["inputs"][str(a.seed)]
    if pair != dict(config_sha256=sha(a.config), journal_sha256=sha(a.journal)):
        raise ValueError("actual immutable dataset inputs differ")

    def budget():
        guard(a.deadline, a.source_repo, a.output)

    budget()
    train, _, receipt, dataset_sha = prepare_ordering(
        a.journal, c, c["excluded_training_position_keys"]
    )
    updates = min(1024, 4 * receipt["training_rows"] // 256)
    if updates < 8:
        raise ValueError("fixed derived training budget must support 8-update proof")
    contract = dict(
        seed=a.seed,
        source_commit=c["source_commit"],
        updates=updates,
        protocol_sha256=a.protocol_sha256,
        dataset_sha256=dataset_sha,
        config_sha256=sha(a.config),
        journal_sha256=sha(a.journal),
        receipt=receipt,
        deadline=a.deadline,
    )
    state = None
    if a.resume:
        if sha(a.resume) != a.resume_sha256:
            raise ValueError("native SHA differs")
        state = json.loads(a.resume.read_text())
    learner = Learner(a.seed, contract, state)
    if a.audit_only:
        budget()
        print(json.dumps(dict(status="strict-quiet-native-load-PASS", step=learner.step)))
        return
    if state is None:
        a.output.mkdir(parents=True, exist_ok=False)
        publish(a.output / "step-00000000.json", learner.native())
    if (
        state is not None
        and json.loads((a.output / "step-00000000.json").read_text())["contract"] != contract
    ):
        raise ValueError("original output contract differs")
    stop = updates if a.stop_at is None else a.stop_at
    while learner.step < stop:
        learner.advance(train, min(stop, learner.step + 4), budget)
        publish(a.output / f"step-{learner.step:08d}.json", learner.native())
    if stop == updates:
        publish(a.output / "candidate.json", model_dict(learner.weights))
    budget()
    print(json.dumps(dict(status="fixed-final-quiet-fit-not-strength", step=learner.step)))


if __name__ == "__main__":
    main()
