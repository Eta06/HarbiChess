"""Actual replay-bound same-data paired family proof/fresh64 ROOT contracts."""

import argparse
import json
from pathlib import Path

from convert import derive
from model import FEATURE_SCHEMA
from native import MATH
from support import canonical, clock, pins_tree, publish, read, ref
from zero_parent import admit

HERE = Path(__file__).resolve().parent


def build(spec, check=lambda: None):
    if spec["schema"] != "antisymmetric-forensic-training-build-seal-v3" or spec["mode"] not in [
        "proof",
        "fresh-fit",
    ]:
        raise ValueError("typed actual own proof/fit contract")
    clock(spec, 600 if spec["mode"] == "proof" else 1800)
    pins_tree(spec)
    paired, provenance = derive(read(spec["conversion_seal"]), check)
    stored = read(spec["dataset"])
    stored_provenance = read(spec["provenance"])
    provenance["dataset_sha256"] = spec["dataset"]["sha256"]
    if canonical(stored) != canonical(paired) or canonical(stored_provenance) != canonical(
        provenance
    ):
        raise ValueError("exact common target/fullhistory/paired representation replay")
    _, zero_contract = admit(spec["zero_parent"], spec["seed"])
    zero_profile = read(spec["zero_profile"])
    zero_protocol = read(spec["zero_profile_protocol"])
    if (
        zero_profile["schema"] != "antisymmetric-actual-profile-result-v3"
        or zero_profile["status"] != "PASS-zero24-exact-packets-traces-not-strength"
        or zero_profile["mode"] != "zero"
        or len(zero_profile["packets"]) != 24
        or zero_profile["protocol_sha256"] != spec["zero_profile_protocol"]["sha256"]
        or zero_profile["clock"]["protocol"] != spec["zero_profile_protocol"]
        or zero_profile["clock"]["helper"] != ref(HERE / "qualify_profile.py")
        or zero_profile["helper_sha256"] != ref(HERE / "qualify_profile.py")["sha256"]
        or zero_protocol["zeros"][str(spec["seed"])] != spec["zero_parent"]
        or zero_protocol["paired_datasets"][str(spec["seed"])] != spec["dataset"]
        or zero_protocol["original_search"] != zero_contract["search_helper"]
        or not zero_profile["clock"]["first"]
        <= zero_profile["finished"]
        <= zero_profile["clock"]["deadline"]
    ):
        raise ValueError(
            "actual fixed zero24 full packet/alias-source qualification before optimizer proof/fit"
        )
    raw = provenance["common_provenance"]["inputs"]
    reg = read(raw["registration"])
    if reg["seed"] != spec["seed"] or reg["search_helper"] != zero_contract["search_helper"]:
        raise ValueError("same named seed/prior original search and human-zero source")
    if raw["prior"]["sha256"] != zero_contract["prior_helper"]["sha256"]:
        raise ValueError("exact unchanged authoritative humanprior")
    if spec["mode"] == "fresh-fit":
        from admission import phase

        c, states, _ = phase(spec["own_proof_result"], spec["own_proof_contract"], "proof")
        if (
            c["dataset_sha256"] != spec["dataset"]["sha256"]
            or c["zero_parent"] != spec["zero_parent"]
            or c["seed"] != spec["seed"]
            or not states[1]["step"] == 8
        ):
            raise ValueError("same-data same-namedzero actual six-open proof before fresh64")
    source = {
        n: ref(HERE / n)
        for n in [
            "model.py",
            "native.py",
            "train.py",
            "support.py",
            "zero_parent.py",
            "initialize.py",
            "convert.py",
            "adapter.py",
            "contracts.py",
            "prove.py",
            "admission.py",
            "compiled_evaluator.py",
        ]
    }
    return dict(
        spec,
        phase="antisymmetric-procedural-own-learning-v3",
        updates=64,
        feature_schema=FEATURE_SCHEMA,
        math=MATH,
        initializer="literalzero-antisymmetric-new-Adam-RNG-v1",
        dataset_sha256=spec["dataset"]["sha256"],
        source_refs=source,
        search_helper=zero_contract["search_helper"],
        prior_helper=zero_contract["prior_helper"],
        compiled_refs=zero_contract["compiled_refs"],
        teacher_labels_used=False,
        data_family_bridge=(
            "frozen-humanzero procedural1024 own-Q, "
            "same common-data representation ablation;not-NNUE-resume"
        ),
        generation=1,
        original_first_epoch=spec["first"],
        original_deadline_epoch=spec["deadline"],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seal", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    spec = json.loads(a.seal.read_bytes())
    from support import guard

    check = guard(spec, 600 if spec["mode"] == "proof" else 1800)
    publish(a.output, build(spec, check))
