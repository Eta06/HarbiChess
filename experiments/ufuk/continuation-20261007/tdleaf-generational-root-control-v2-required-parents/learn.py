"""ROOT-only collection->conversion->six audit->proof->fresh128, immutable phases."""

import argparse
import os
import sys
from pathlib import Path

from control import END, Owner, pinned, publish, read, ref, sha


def inputs(collection):
    reg = read(collection["registration"])
    receipt = read(collection["receipt"])
    if (
        receipt["status"] != "PASS-exact-row-budget"
        or receipt["train_rows"] != 2048
        or receipt["registration_sha256"] != collection["registration"]["sha256"]
    ):
        raise ValueError("complete fixed2048 collection first")
    out = Path(reg["output_path"])
    if pinned(collection["receipt"]) != out / "receipt.json":
        raise ValueError("actual registered receipt path")
    h = reg["parent_helpers"]
    return (
        reg,
        receipt,
        dict(
            schema="procedural-generational-tdleaf-dataset-conversion-seal-v2",
            registration=collection["registration"],
            receipt=collection["receipt"],
            events=ref(out / "events.jsonl"),
            protected_aliases=reg["protected_aliases"],
            features=dict(path=str(Path(h["directory"]) / "model.py"), sha256=h["model_sha256"]),
            prior=dict(path=h["prior_path"], sha256=h["prior_sha256"]),
            alias_chunks=[ref(out / x["file"]) for x in receipt["alias_chunks"]],
        ),
    )


def run_chain(owner, collection):
    reg, receipt, spec = inputs(collection)
    seed, generation = reg["seed"], reg["generation"]
    if seed not in owner.config["seeds"] or generation not in (1, 2, 3):
        raise ValueError("fixed seed/generation")
    os.sched_setaffinity(0, {owner.config["cpu_cores"][str(seed)]})
    spec["producer_directory"] = str(owner.h)

    def convert_cmd(p, first, end):
        publish(p / "seal.json", spec)
        clock = dict(
            schema="procedural-generational-tdleaf-conversion-clock-v1",
            first=first,
            deadline=end,
            operator_end_epoch=END,
            helper_sha256=sha(owner.h / "convert_cli.py"),
            conversion_seal_sha256=sha(p / "seal.json"),
            cpu_core=owner.config["cpu_cores"][str(seed)],
        )
        publish(p / "clock.json", clock)
        return [
            sys.executable,
            str(owner.h / "convert_cli.py"),
            "--seal",
            str(p / "seal.json"),
            "--clock",
            str(p / "clock.json"),
            "--output",
            str(p / "converted"),
        ]

    converted = owner.phase("convert", 600, convert_cmd)
    conversion = read(ref(converted / "converted/result.json"))
    if conversion["status"] != "PASS-generational-TDLeaf-full-ledger-conversion-not-strength":
        raise ValueError("actual conversion required")

    def audit_cmd(p, first, end):
        clock = dict(
            schema="procedural-generational-tdleaf-six-root-audit-clock-v2",
            first=first,
            deadline=end,
            cpu_core=owner.config["cpu_cores"][str(seed)],
            producer_directory=str(owner.h),
            helper_sha256=sha(owner.h / "audit_collection_six.py"),
            registration_sha256=collection["registration"]["sha256"],
            receipt_sha256=collection["receipt"]["sha256"],
            events_sha256=spec["events"]["sha256"],
        )
        publish(p / "clock.json", clock)
        return [
            sys.executable,
            str(owner.h / "audit_collection_six.py"),
            "--registration",
            str(pinned(collection["registration"])),
            "--receipt",
            str(pinned(collection["receipt"])),
            "--events",
            str(pinned(spec["events"])),
            "--clock",
            str(p / "clock.json"),
            "--output",
            str(p / "audit.json"),
        ]

    audited = owner.phase("six-actual-searches", 600, audit_cmd)
    audit = read(ref(audited / "audit.json"))
    if audit["status"] != "PASS-six-actual-chronological-TDLeaf-PV-packets-and-full-alias-traces":
        raise ValueError("six actual chronological packets required")
    base = dict(
        schema="procedural-generational-tdleaf-contract-build-seal-v2",
        status="registered",
        operator_end_epoch=END,
        parent_admission_seal=collection["parent_admission_seal"],
        parent_admission_result=collection["parent_admission_result"],
        collection_registration=collection["registration"],
        collection_receipt=collection["receipt"],
        collection_audit_result=ref(audited / "audit.json"),
        collection_audit_clock=ref(audited / "clock.json"),
        collection_audit_helper=ref(owner.h / "audit_collection_six.py"),
        dataset=ref(converted / "converted/dataset.json"),
        target_provenance=ref(converted / "converted/provenance.json"),
        cpu_core=owner.config["cpu_cores"][str(seed)],
    )

    def training_cmd(mode, proof_contract=None, proof_result=None):
        def build(p, first, end):
            seal = dict(base, mode=mode, first=first, deadline=end, output=str(p / "native-output"))
            if mode == "fresh-fit":
                seal.update(proof_contract=proof_contract, proof_result=proof_result)
            publish(p / "seal.json", seal)
            return [
                sys.executable,
                str(Path(__file__).with_name("train_entry.py")),
                "--helper-directory",
                str(owner.h),
                "--seal",
                str(p / "seal.json"),
                "--contract",
                str(p / "contract.json"),
                "--registration",
                str(p / "orchestration.json"),
            ]

        return build

    proof = owner.phase("proof8-4-fresh8", 600, training_cmd("proof"))
    fit = owner.phase(
        "fresh128",
        1800,
        training_cmd(
            "fresh-fit", ref(proof / "contract.json"), ref(proof / "native-output/result.json")
        ),
    )
    result = read(ref(fit / "native-output/result.json"))
    if (
        result["status"]
        != "PASS-generational-TDLeaf-fixed-phase-and-fresh-native-loads-not-strength"
    ):
        raise ValueError("actual fresh128 required")
    if result["own_updates"] != 128 or [r["step"] for r in result["native_payloads"]] != [0, 128]:
        raise ValueError("actual final128 and two fresh native loads")
    record = dict(
        schema="ROOT-generational-TDLeaf-generation-complete-not-strength-v1",
        seed=seed,
        generation=generation,
        collection=collection,
        conversion=ref(converted / "converted/result.json"),
        audit=ref(audited / "audit.json"),
        audit_clock=ref(audited / "clock.json"),
        dataset=base["dataset"],
        provenance=base["target_provenance"],
        proof_contract=ref(proof / "contract.json"),
        proof_result=ref(proof / "native-output/result.json"),
        fit_contract=ref(fit / "contract.json"),
        fit_result=ref(fit / "native-output/result.json"),
        candidate=ref(fit / "native-output/whole/candidate.pt"),
        native=ref(fit / "native-output/whole/native.pt"),
        endpoint_selectable=generation == 3,
        scope=owner.config["strength_scope"],
    )
    publish(owner.root / "generation-complete.json", record)
    return record


def next_parent(record, helper_directory):
    """Pure typed next-parent seal; actual readonly admit is a separate600 phase."""
    c = read(record["fit_contract"])
    if c["generation"] != record["generation"] or c["updates"] != 128:
        raise ValueError("completed fixed previous generation")
    return dict(
        schema="human-prior-own-parent-admission-seal-v1",
        status="registered",
        weights_only=True,
        seed=record["seed"],
        generation=record["generation"] + 1,
        parent_contract=record["fit_contract"],
        parent_proof_contract=record["proof_contract"],
        parent_proof_result=record["proof_result"],
        parent_fit_result=record["fit_result"],
        parent_dataset=record["dataset"],
        parent_candidate=record["candidate"],
        parent_native=record["native"],
        parent_native_helper=ref(Path(helper_directory) / "native.py"),
        parent_model_helper=ref(Path(helper_directory) / "model.py"),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--config-sha256", required=True)
    p.add_argument("--collection", type=Path, required=True)
    p.add_argument("--collection-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    if not a.execute:
        p.error("ROOT --execute required")
    owner = Owner(read(dict(path=str(a.config), sha256=a.config_sha256)), a.output)
    run_chain(owner, read(dict(path=str(a.collection), sha256=a.collection_sha256)))
