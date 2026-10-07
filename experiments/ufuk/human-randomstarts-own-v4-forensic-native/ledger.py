"""Publish CLOSED generation ledger only; no model, phase launch or promotion."""

import argparse
import json
from pathlib import Path

from parent_bridge import canonical, read, validate_metadata


def build(spec):
    if (
        spec["schema"] != "human-randomstarts-own-closed-ledger-seal-v2"
        or spec["status"] != "registered"
    ):
        raise ValueError("ROOT closed-generation ledger seal required")
    outgoing = read(spec["next_parent_admission_seal"])
    contract = validate_metadata(outgoing)
    generation = outgoing["generation"] - 1
    if (
        generation != spec["generation"]
        or outgoing["seed"] != spec["seed"]
        or contract["generation"] != generation
        or contract["execution_scope_schema"] != "human-prior-own-execution-contract-v1"
    ):
        raise ValueError("this exact finished generation required")
    registration = read(spec["collection_registration"])
    receipt = read(spec["collection_receipt"])
    provenance = read(spec["converted_provenance"])
    if (
        registration["generation"] != generation
        or receipt["generation"] != generation
        or receipt["status"] != "PASS-exact-row-budget"
        or receipt["registration_sha256"] != spec["collection_registration"]["sha256"]
        or receipt["train_rows"] != 1024
        or contract["collection_receipt_sha256"] != spec["collection_receipt"]["sha256"]
        or provenance["collection_receipt_sha256"] != spec["collection_receipt"]["sha256"]
        or provenance["generation"] != generation
        or contract["target_provenance_sha256"] != spec["converted_provenance"]["sha256"]
        or registration["parent_admission_seal"] != contract["parent_admission_seal"]
        or registration["parent_admission_result"] != contract["parent_admission_result"]
    ):
        raise ValueError("complete current-parent collection/conversion lineage")
    return dict(
        schema="human-randomstarts-own-closed-ledger-v2",
        status="CLOSED-fitted-not-strength",
        generation=generation,
        seed=spec["seed"],
        optimizer_updates=64,
        own_rows=1024,
        next_parent_admission_seal=spec["next_parent_admission_seal"],
        collection_registration=spec["collection_registration"],
        collection_receipt=spec["collection_receipt"],
        converted_provenance=spec["converted_provenance"],
        state_resume="offline native only, exact same contract/phase clock",
        collection_half_cursor_resume_supported=False,
        teacher_labels_used=False,
        source_sha256=contract["source_sha256"],
        phase_clocks_unchanged=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seal", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    body = build(json.loads(a.seal.read_bytes()))
    with a.output.open("xb") as f:
        f.write(canonical(body) + b"\n")
