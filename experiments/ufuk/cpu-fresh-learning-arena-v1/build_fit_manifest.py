"""Derive a portable arena input manifest from six already-finished fit receipts."""

import argparse
import hashlib
import json
from pathlib import Path

from safetensors import safe_open

SOURCE = "6fcc8b476d25495d1c9c413e55b2c7ba4794013e"
SEEDS = (20262805, 20262806)
ROOT = Path("/workspace")
MC_SC_COHORT = ROOT / "work/harbichess/cpu-fresh-production-fits-v1/cohort-result.json"
FULL_COHORT = (
    ROOT / "work/harbichess/cpu-fresh-fullcritic-production-fits-v1/cohort-result.json"
)
MC_SC_ROOT = Path("/dev/shm/harbichess-fresh-production-fits-v1-20261005")
FULL_ROOT = Path("/dev/shm/harbichess-fresh-fullcritic-production-fits-v1-20261005")
E8 = ROOT / (
    "work/harbichess/a100/restoration/local-rehearsal-content/"
    "harbichess-inputs/initial-e8.safetensors"
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def derive(cohort_path, cohort_root, family, tags):
    cohort = json.loads(cohort_path.read_text())
    if (
        cohort.get("status")
        != "PASS-all-registered-fits-and-native-audits-not-strength"
    ):
        raise ValueError(
            f"{family} fit cohort did not pass its registered native audit"
        )
    records = {}
    for row in cohort["rows"]:
        tag = row["tag"]
        if tag not in tags:
            continue
        if row["status"] != "PASS-final-fit-and-two-strict-native-loads-not-strength":
            raise ValueError(f"{tag} final-fit/native audit status differs")
        seed, role = int(tag.rsplit("-", 1)[1]), family
        if sha(row["result_path"]) != row["result_sha256"]:
            raise ValueError(f"{tag} result receipt bytes differ")
        result = json.loads(Path(row["result_path"]).read_text())
        if (
            result.get("status") != "completed-fit-not-strength"
            or result.get("strength_success_claimed") is True
        ):
            raise ValueError(f"{tag} result is not a completed non-strength fit")
        contract = result["contract"]
        if contract.get("seed") != seed or contract.get("source_commit") != SOURCE:
            raise ValueError(f"{tag} seed/source contract differs")
        if result.get("accepted_updates") != row["result"]["accepted_updates"]:
            raise ValueError(f"{tag} update cursor differs")
        teacher_flag = row["result"].get(
            "new_teacher_labels", row["result"].get("teacher_labels")
        )
        if teacher_flag is not False:
            raise ValueError(f"{tag} unexpectedly carries teacher labels")
        candidate = (
            Path("/dev/shm")
            / Path(row["result_path"]).parent.relative_to("/dev/shm")
            / "candidate.safetensors"
        )
        if (
            sha(candidate) != row["candidate_sha256"]
            or candidate.stat().st_size != row["candidate_bytes"]
        ):
            raise ValueError(f"{tag} candidate bytes differ from cohort")
        accepted = result["accepted_updates"]
        run_dir = candidate.parent
        initial_dir = run_dir / "checkpoints/step-00000000"
        final_dir = run_dir / "checkpoints" / f"step-{accepted:08d}"
        native = {}
        for label, directory, expected_step in (
            ("initial", initial_dir, 0),
            ("final", final_dir, accepted),
        ):
            manifest_path = directory / "checkpoint.json"
            payload_path = directory / "training.pt"
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("accepted") != expected_step:
                raise ValueError(f"{tag} {label} native cursor differs")
            payload_sha = manifest.get(
                "training_pt_sha256", manifest.get("artifacts", {}).get("training.pt")
            )
            if payload_sha != sha(payload_path):
                raise ValueError(f"{tag} {label} native payload SHA differs")
            if manifest.get("contract") != result["contract"]:
                raise ValueError(
                    f"{tag} {label} native contract differs from completed fit receipt"
                )
            native[label] = {
                "checkpoint_path": str(directory),
                "manifest_sha256": sha(manifest_path),
                "training_pt_sha256": sha(payload_path),
                "accepted_updates": expected_step,
                "schema": manifest["schema"],
                "contract": manifest["contract"],
            }
        with safe_open(candidate, framework="pt", device="cpu") as f:
            raw = (f.metadata() or {}).get("harbichess")
        metadata = json.loads(raw) if raw else None
        if not metadata or metadata.get("transfer") != "weights-only":
            raise ValueError(f"{tag} candidate metadata missing/version differs")
        provenance = metadata.get("provenance", {})
        if (
            provenance.get("updates") != accepted
            or provenance.get("strength_claimed") is not False
        ):
            raise ValueError(f"{tag} candidate provenance cursor/claim differs")
        dataset_sha = contract.get(
            "dataset_sha256", contract.get("search_extended_dataset_sha256")
        )
        if not dataset_sha:
            dataset_sha = contract.get("search_extended_dataset_sha256")
        records[seed, role] = {
            "path": str(candidate),
            "sha256": row["candidate_sha256"],
            "bytes": row["candidate_bytes"],
            "fit_family": family,
            "fit_cohort_path": str(cohort_path),
            "fit_cohort_sha256": sha(cohort_path),
            "fit_tag": tag,
            "fit_receipt_path": row["result_path"],
            "fit_receipt_sha256": row["result_sha256"],
            "fit_contract": contract,
            "dataset_sha256": dataset_sha,
            "common_mc_dataset_sha256": contract.get(
                "common_dataset_sha256",
                contract.get(
                    "common_mc_dataset_sha256", contract.get("dataset_sha256")
                ),
            ),
            "search_dataset_sha256": contract.get(
                "search_extended_dataset_sha256", contract.get("dataset_sha256")
            ),
            "native": native,
            "candidate_metadata_sha256": hashlib.sha256(
                json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
    if len(records) != len(tags):
        raise ValueError(
            f"{family} cohort does not contain exactly the expected six rows"
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("fit-provenance.json")
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    mc_tags = {f"mc-{s}" for s in SEEDS}
    sc_tags = {f"sc-{s}" for s in SEEDS}
    full_tags = {f"fullcritic-{s}" for s in SEEDS}
    records = {}
    records.update(derive(MC_SC_COHORT, MC_SC_ROOT, "mc", mc_tags))
    records.update(derive(MC_SC_COHORT, MC_SC_ROOT, "sc", sc_tags))
    records.update(derive(FULL_COHORT, FULL_ROOT, "full", full_tags))
    result = {
        "schema": "fresh-own-learning-arena-fit-provenance-v1",
        "status": "PASS-fit-provenance-only-not-strength",
        "source_commit": SOURCE,
        "seeds": list(SEEDS),
        "e8": {"path": str(E8), "sha256": sha(E8)},
        "fits": {
            str(seed): {role: records[seed, role] for role in ("mc", "sc", "full")}
            for seed in SEEDS
        },
        "fit_cohorts": {
            "mc_sc": {"path": str(MC_SC_COHORT), "sha256": sha(MC_SC_COHORT)},
            "full": {"path": str(FULL_COHORT), "sha256": sha(FULL_COHORT)},
        },
        "scope": (
            "External immutable fit/native provenance manifest; candidate files do not "
            "inherit a checkpoint manifest from their own parent directory."
        ),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"status": result["status"], "models": 6, "sha256": sha(args.output)}
        )
    )


if __name__ == "__main__":
    main()
