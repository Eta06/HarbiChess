"""Fixed48-slot JSON native trainer; fresh8/4/resume8 uses identical48 contract."""

import argparse
import json
import time
from pathlib import Path

from labels import admission
from learner import Learner
from support import publish, registration, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--registration", required=True, type=Path)
    p.add_argument("--stop", required=True, type=int)
    p.add_argument("--resume", type=Path)
    a = p.parse_args()
    r, out, guard = registration(a.registration)
    try:
        data = json.loads(Path(r["inputs"]["dataset"]["path"]).read_text())
        if (
            data["schema"] != "own-selective-q-leaf-dataset-v1"
            or data["seed"] != r["seed"]
            or data["source_commit"] != r["source_commit"]
            or len(data["roots"]) != 256
        ):
            raise ValueError("actual bothseed fixed256 label source")
        split = json.loads(Path(r["inputs"]["split"]["path"]).read_text())
        if set(split["train"]) & set(split["validation"]):
            raise ValueError("trajectory leakage")
        groups = admission(data, set(split["train"]))
        if (
            {x["trajectory_id"] for x in data["leaves"]}
            - set(split["train"])
            - set(split["validation"])
        ):
            raise ValueError("unknown trajectory split")
        contract = dict(
            schema="own-selective-q-learning-contract-v1",
            seed=r["seed"],
            updates=48,
            inputs=r["inputs"],
            source_commit=r["source_commit"],
            helper_sha256=r["helper_sha256"],
            first_epoch=r["first_epoch"],
            deadline_epoch=r["deadline_epoch"],
        )
        state = None if a.resume is None else json.loads(a.resume.read_text())
        learner = Learner(r["seed"], contract, state)
        publish(out / "initial.json", learner.native())
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
                finished=time.time(),
                deadline=r["deadline_epoch"],
                native_sha256=sha(out / "native.json"),
            ),
        )
    except Exception as e:
        publish(out / "failure.json", dict(status="INCOMPLETE", error=repr(e)))
        raise


if __name__ == "__main__":
    main()
