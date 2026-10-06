"""ROOT-only metadata freeze, no SGD/search: explicit observed clocks and fresh receipts."""

import argparse
import json
import time
from pathlib import Path

from support import END, publish, sha

HERE = Path(__file__).parent
OLD = Path("/workspace/work/harbichess/cpu-selective-quiescence-proposal")
SOURCE = Path("/workspace/work/harbichess/cpu-additive-source-6fcc8b4")
COMMIT = "6fcc8b476d25495d1c9c413e55b2c7ba4794013e"


def binding(path):
    return dict(path=str(path.resolve()), sha256=sha(path))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["proof", "architecture", "fit"], required=True)
    p.add_argument("--seed", type=int, choices=[20262905, 20262906], required=True)
    p.add_argument("--first", type=float, required=True)
    p.add_argument("--cpu-core", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--registration", type=Path, required=True)
    for name in [
        "proof-registration",
        "proof-receipt",
        "architecture-registration",
        "architecture-receipt",
    ]:
        p.add_argument("--" + name, type=Path)
    a = p.parse_args()
    seconds = 600 if a.phase == "architecture" else 900
    if not a.first <= time.time() < a.first + seconds <= END:
        raise ValueError("actual original clock, no future first or reset")
    if a.output.exists():
        raise ValueError("new output only")
    closure = {
        str(x.resolve()): sha(x) for x in HERE.glob("*.py") if not x.name.startswith("test_")
    }
    for x in [
        *(OLD / n for n in ["learner.py", "model.py", "labels.py", "search.py"]),
        SOURCE / "src/harbichess/training/cgroup_budget.py",
        Path("/workspace/HarbiChess/experiments/ufuk/cpu-classical-own-v1/arena/search.py"),
        Path("/workspace/HarbiChess/experiments/ufuk/cpu-classical-own-v1/value.py"),
    ]:
        closure[str(x.resolve())] = sha(x)
    data = Path(f"/dev/shm/harbichess-selective-q-labels-{a.seed}-v1")
    inputs = {n: binding(data / (n + ".json")) for n in ["dataset", "split"]}
    r = dict(
        schema="own-balanced-risk-ROOT-registration-v2",
        status="registered",
        seed=a.seed,
        source_repo=str(SOURCE),
        source_commit=COMMIT,
        first_epoch=a.first,
        deadline_epoch=a.first + seconds,
        cpu_core=a.cpu_core,
        output=str(a.output),
        helper_sha256=closure,
        inputs=inputs,
    )
    if a.phase == "architecture":
        old = json.loads(
            Path(
                f"/workspace/work/harbichess/continuation-20261005/selective-q-profile-v1-registration/{a.seed}-registration.json"
            ).read_bytes()
        )
        r.update(
            inputs=old["inputs"], base_search=old["base_search"], prior_value=old["prior_value"]
        )
    if a.phase == "fit":
        if any(
            x is None
            for x in [
                a.proof_registration,
                a.proof_receipt,
                a.architecture_registration,
                a.architecture_receipt,
            ]
        ):
            raise ValueError("ALL four qualification files mandatory")
        proof = json.loads(a.proof_receipt.read_bytes())
        pr = json.loads(a.proof_registration.read_bytes())
        arch = json.loads(a.architecture_receipt.read_bytes())
        ar = json.loads(a.architecture_registration.read_bytes())
        if (
            proof["status"]
            != "PASS-balanced-v2-own-data8-4-fresh8-six-fullnative-loads-not-strength"
            or proof["registration_sha256"] != sha(a.proof_registration)
            or len(proof["native_payloads"]) != 6
            or pr["inputs"] != inputs
            or any(
                x["seed"] != a.seed or x["helper_sha256"] != closure or x["source_commit"] != COMMIT
                for x in [pr, ar]
            )
            or arch["status"] != "PASS-architecture-only-not-strength"
            or len(arch["packets"]) != 24
            or arch["median_ratio"] > 1.10
        ):
            raise ValueError("new exact proof and architectural profile before fresh fit")
        for receipt, reg in [(proof, pr), (arch, ar)]:
            if (
                not reg["first_epoch"]
                <= receipt["finished"]
                <= receipt["deadline"]
                == reg["deadline_epoch"]
            ):
                raise ValueError("qualification original clock")
        for row in proof["native_payloads"]:
            if sha(row["path"]) != row["sha256"]:
                raise ValueError("actual proof payload changed")
        inputs.update(
            native_qualification=binding(a.proof_receipt),
            profile_qualification=binding(a.architecture_receipt),
        )
    publish(a.registration, r)
    print(
        json.dumps(
            dict(registration=str(a.registration), sha256=sha(a.registration), phase=a.phase)
        )
    )


if __name__ == "__main__":
    main()
