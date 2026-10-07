"""ROOT metadata preparation only; never launches a child or evaluates/trains a model."""

import argparse
import json
from pathlib import Path

from support import clock, pins_tree, publish, read, ref

HERE = Path(__file__).resolve().parent


def prepare(inputs, mode, first, deadline, operator, cpu_core, output):
    common = dict(
        seed=inputs["seed"],
        core_repo=inputs["core_repo"],
        core_commit=inputs["core_commit"],
        first=first,
        deadline=deadline,
        operator_end_epoch=operator,
        cpu_core=cpu_core,
        status="registered",
    )
    registry_fit = mode == "registry" and read(inputs["contract"])["mode"] == "fresh-fit"
    clock(common, 1800 if mode == "fresh-fit" or registry_fit else 600)
    pins_tree(inputs)
    if mode == "initialize":
        return dict(
            common,
            schema="antisymmetric-zero-initialization-seal-v3",
            prior_helper=inputs["prior_helper"],
            search_helper=inputs["search_helper"],
            compiled_refs=inputs["compiled_refs"],
            teacher_labels_used=False,
        )
    if mode == "convert":
        return dict(
            common, **inputs["conversion"], schema="antisymmetric-forensic-conversion-seal-v3"
        )
    if mode in ["proof", "fresh-fit"]:
        value = dict(
            common,
            schema="antisymmetric-forensic-training-build-seal-v3",
            mode=mode,
            conversion_seal=inputs["conversion_seal"],
            dataset=inputs["paired_dataset"],
            provenance=inputs["paired_provenance"],
            zero_parent=inputs["zero_parent"],
            zero_profile=inputs["zero_profile"],
            zero_profile_protocol=inputs["zero_profile_protocol"],
        )
        if mode == "fresh-fit":
            value.update(
                own_proof_result=inputs["own_proof_result"],
                own_proof_contract=inputs["own_proof_contract"],
            )
        return value
    if mode == "registry":
        if Path(output).exists():
            raise FileExistsError("fresh phase output already exists")
        if Path(inputs["contract"]["path"]).resolve().is_relative_to(Path(output).resolve()):
            raise ValueError("contract/seals outside actual proof/fresh output directory")
        c = read(inputs["contract"])
        if (c["first"], c["deadline"], c["operator_end_epoch"], c["cpu_core"]) != (
            first,
            deadline,
            operator,
            cpu_core,
        ):
            raise ValueError("registry SAME already frozen phase clock")
        return dict(
            common,
            schema="antisymmetric-forensic-training-orchestration-v3",
            mode=c["mode"],
            train=ref(HERE / "train.py"),
            native=ref(HERE / "native.py"),
            contract=inputs["contract"],
            dataset=c["dataset"],
            parent_candidate=c["zero_parent"]["candidate"],
            inputs=dict(conversion_seal=c["conversion_seal"], provenance=c["provenance"]),
            source_sha256={v["path"]: v["sha256"] for v in c["source_refs"].values()},
            output=str(output),
        )
    raise ValueError("explicit phase only")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument(
        "--mode", choices=["initialize", "convert", "proof", "fresh-fit", "registry"], required=True
    )
    for key in ["first", "deadline", "operator-end-epoch"]:
        p.add_argument("--" + key, type=float, required=True)
    p.add_argument("--cpu-core", type=int, required=True)
    p.add_argument("--phase-output", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    publish(
        a.output,
        prepare(
            json.loads(a.inputs.read_bytes()),
            a.mode,
            a.first,
            a.deadline,
            a.operator_end_epoch,
            a.cpu_core,
            a.phase_output,
        ),
    )
