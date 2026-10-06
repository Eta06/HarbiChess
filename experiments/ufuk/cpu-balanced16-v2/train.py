"""Fresh balancedv2 fixed48 fit or strict audit-only native load; ROOT clocks only."""

import argparse
import json
import time
from pathlib import Path

from learner_v2 import MATH, Learner
from support import canonical, publish, registration, sha
from transform import transform


def contract_for(r):
    data = json.loads(Path(r["inputs"]["dataset"]["path"]).read_bytes())
    split = json.loads(Path(r["inputs"]["split"]["path"]).read_bytes())
    if (
        data["schema"] != "own-selective-q-leaf-dataset-v1"
        or data["seed"] != r["seed"]
        or data["source_commit"] != r["source_commit"]
        or len(data["roots"]) != 256
    ):
        raise ValueError("fixed original dataset/source/256 roots")
    groups, record = transform(data, split)
    contract = dict(
        schema="own-balanced-risk-learning-contract-v2",
        seed=r["seed"],
        updates=48,
        math=MATH,
        transform=record,
        inputs=r["inputs"],
        source_commit=r["source_commit"],
        helper_sha256=r["helper_sha256"],
        first_epoch=r["first_epoch"],
        deadline_epoch=r["deadline_epoch"],
    )
    return groups, contract


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registration", type=Path, required=True)
    p.add_argument("--stop", type=int, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--audit-only", action="store_true")
    a = p.parse_args()
    r, out, guard = registration(a.registration)
    try:
        groups, contract = contract_for(r)
        state = None if a.resume is None else json.loads(a.resume.read_bytes())
        learner = Learner(r["seed"], contract, state)
        if a.audit_only:
            if state is None or a.stop != state["step"]:
                raise ValueError("audit-only actual exact native step")
            if canonical(learner.native()) != canonical(state):
                raise ValueError("all model/Adam/global/sampler RNG canonical storage")
        publish(out / "initial.json", learner.native())
        if not a.audit_only:
            learner.advance(groups, a.stop, guard)
        guard()
        publish(out / "native.json", learner.native())
        publish(out / "candidate.json", learner.native()["model"])
        guard()
        publish(
            out / "result.json",
            dict(
                status="PASS-fixed-stop-not-strength",
                step=learner.step,
                audit_only=a.audit_only,
                finished=time.time(),
                deadline=r["deadline_epoch"],
                native_sha256=sha(out / "native.json"),
            ),
        )
    except BaseException as e:
        publish(out / "failure.json", dict(status="INCOMPLETE", error=repr(e)))
        raise


if __name__ == "__main__":
    main()
