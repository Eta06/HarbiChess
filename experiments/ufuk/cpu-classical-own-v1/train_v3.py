"""Fixed-final-journal offline learner; no automatic data selection or online cycling."""

import argparse
import json
from pathlib import Path

from journal_v3 import canonical, sha
from learner import Learner, prepare
from value import model_dict


def publish(path, obj):
    import os

    path = Path(path)
    data = canonical(obj)
    if len(data) > 512 * 1024:
        raise ValueError("compact candidate/native512KiB ceiling")
    temp = path.with_name(path.name + ".tmp")
    with temp.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(temp, path)
    finally:
        temp.unlink()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("protocol", "config", "journal", "output", "source-repo"):
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
    from runtime import affinity

    affinity(a.cpu_core)
    if sha(a.protocol) != a.protocol_sha256:
        raise ValueError("protocol bytes differ")
    q = json.loads(a.protocol.read_text())
    c = json.loads(a.config.read_text())
    if (
        q["schema"] != "classical-own-offline-protocol-v3"
        or a.seed not in q["seeds"]
        or a.deadline != q["deadline_by_seed"][str(a.seed)]
        or c["seed"] != a.seed
    ):
        raise ValueError("protocol/seed/clock differs")
    import subprocess

    if subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=a.source_repo, text=True
    ).strip() != c["source_commit"] or subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=a.source_repo, text=True
    ):
        raise ValueError("clean source differs")
    pair = q["inputs"][str(a.seed)]
    if pair["journal_sha256"] != sha(a.journal) or pair["config_sha256"] != sha(a.config):
        raise ValueError("paired data differs")
    if set(q["helper_sha256"]) != {
        "train_v3.py",
        "learner.py",
        "journal_v3.py",
        "value.py",
        "runtime.py",
    }:
        raise ValueError("entire offline helper closure required")
    for filename, expected in q["helper_sha256"].items():
        if sha(Path(__file__).with_name(filename)) != expected:
            raise ValueError("learning helper differs")
    if q["objective"] != {"terminal_weight": 0.75, "search_weight": 0.25, "prior_l2": 0.01} or q[
        "optimizer"
    ] != {
        "lr": 0.01,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "batch": 256,
        "slots": 4,
        "max_updates": 1024,
    }:
        raise ValueError("learning math differs")
    if c["excluded_training_position_keys"] != q["protected_position_keys"]:
        raise ValueError("protected set differs")

    def guard():
        from runtime import guard as resource_guard

        resource_guard(a.deadline, a.source_repo, a.output.parent)

    guard()
    train, val, receipt, data_sha = prepare(a.journal, c, q["protected_position_keys"])
    if receipt["actions"] != q["final_actions"]:
        raise ValueError("not fixed final data")
    updates = min(1024, 4 * receipt["training_rows"] // 256)
    contract = dict(
        protocol_sha256=a.protocol_sha256,
        journal_sha256=sha(a.journal),
        config_sha256=sha(a.config),
        dataset_sha256=data_sha,
        updates=updates,
        receipt=receipt,
        source_commit=c["source_commit"],
        seed=a.seed,
        original_deadline=a.deadline,
    )
    state = None
    if a.resume:
        if sha(a.resume) != a.resume_sha256:
            raise ValueError("native SHA differs")
        state = json.loads(a.resume.read_text())
    learner = Learner(a.seed, contract, state)
    if a.audit_only:
        print(json.dumps(dict(status="strict-offline-native-load-PASS", step=learner.step)))
        return
    if state is None:
        a.output.mkdir(parents=True, exist_ok=False)
        publish(a.output / "step-00000000.json", learner.native())
    elif json.loads((a.output / "step-00000000.json").read_text())["contract"] != contract:
        raise ValueError("original output contract differs")
    stop = updates if a.stop_at is None else a.stop_at
    while learner.step < stop:
        nextstep = min(stop, learner.step + (4 if stop <= 8 else 128))
        learner.advance(train, nextstep, guard)
        publish(a.output / f"step-{nextstep:08d}.json", learner.native())
    if stop == updates:
        publish(a.output / "candidate.json", model_dict(learner.theta))
    print(
        json.dumps(
            dict(
                status="completed-offline-fit-not-strength"
                if stop == updates
                else "offline-native-proof-not-candidate",
                step=learner.step,
                receipt=receipt,
            )
        )
    )


if __name__ == "__main__":
    main()
