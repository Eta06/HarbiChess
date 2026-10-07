"""ROOT-stamped contract/orchestration preparation; never launches or samples a clock."""

import argparse
import json
from pathlib import Path

import contracts
from parent_bridge import canonical, pinned, sha

HERE = Path(__file__).resolve().parent


def ref(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=sha(path))


def prepare(spec, contract_path, registration_path):
    c = contracts.build(spec)
    cp = Path(contract_path)
    rp = Path(registration_path)
    if cp.exists() or rp.exists() or not cp.resolve().is_relative_to("/dev/shm"):
        raise ValueError("new RAM publish-once outputs required")
    cp.parent.mkdir(parents=True, exist_ok=True)
    with cp.open("xb") as f:
        f.write(canonical(c) + b"\n")
    reg = dict(
        schema="procedural-generational-tdleaf-training-orchestration-v2",
        status="registered",
        first=spec["first"],
        deadline=spec["deadline"],
        operator_end_epoch=spec["operator_end_epoch"],
        mode=spec["mode"],
        seed=c["seed"],
        generation=c["generation"],
        cpu_core=spec["cpu_core"],
        output=spec["output"],
        train=ref(HERE / "train.py"),
        native=ref(HERE / "native.py"),
        contract=ref(cp),
        dataset=spec["dataset"],
        parent_candidate=c["parent_candidate"],
        inputs={
            k: spec[k]
            for k in [
                "collection_registration",
                "collection_receipt",
                "collection_audit_result",
                "collection_audit_clock",
                "target_provenance",
            ]
        },
        source_sha256={
            **c["source_sha256"],
            **c["execution_helpers_sha256"],
            str(HERE / "prepare_training.py"): sha(__file__),
        },
    )
    for r in reg["inputs"].values():
        pinned(r)
    with rp.open("xb") as f:
        f.write(canonical(reg) + b"\n")
    return reg


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seal", type=Path, required=True)
    p.add_argument("--contract", type=Path, required=True)
    p.add_argument("--registration", type=Path, required=True)
    a = p.parse_args()
    prepare(json.loads(a.seal.read_bytes()), a.contract, a.registration)
