"""Tiny synthetic-only native worker; never reads actual action labels or fits."""

import argparse
import json
import sys
from pathlib import Path

SOURCE = Path("/workspace/work/harbichess/cpu-own-deeper-action-proposal")
sys.path.insert(0, str(SOURCE))


def publish(path, obj):
    with path.open("x") as f:
        json.dump(obj, f, sort_keys=True, separators=(",", ":"), allow_nan=False)


def main():
    from action_learner import MATH, Learner

    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stop", type=int, default=8)
    p.add_argument("--resume", type=Path)
    p.add_argument("--audit-only", action="store_true")
    a = p.parse_args()
    contract = {
        "updates": 64,
        "math": MATH,
        "source": "synthetic-not-real-labels",
        "seed": 20262905,
    }
    state = None if a.resume is None else json.loads(a.resume.read_bytes())
    learner = Learner(20262905, contract, state)
    if a.audit_only:
        assert learner.native()["model"] == state["model"]
        assert json.loads(json.dumps(learner.native())) == state
        print(json.dumps({"status": "PASS-synthetic-strict-native-load", "step": learner.step}))
        return
    if state is None:
        a.output.mkdir(exist_ok=False)
        publish(a.output / "step-00000000.json", learner.native())
    groups = {
        f"synthetic{i}": [{"features": [[0, 7, 262], [1, 7, 263]], "target": i % 2}]
        for i in range(16)
    }
    while learner.step < a.stop:
        learner.advance(groups, min(a.stop, learner.step + 4))
        publish(a.output / f"step-{learner.step:08d}.json", learner.native())
    print(json.dumps({"status": "PASS-synthetic-no-real-data", "step": learner.step}))


if __name__ == "__main__":
    main()
