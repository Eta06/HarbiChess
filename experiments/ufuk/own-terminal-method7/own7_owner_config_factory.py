"""Build exact sidecar owner configs from an already frozen slot7 registration."""

import argparse
import json
import time
from pathlib import Path

from own7_audit_support import publish, sha
from own7_strength_config import SEEDS, SOURCE, validate_config

AUDIT_HELPERS = (
    "own7_adapter_controls.py",
    "own7_audit_support.py",
    "own7_audit_core.py",
    "own7_full_audit.py",
    "own7_fresh_cli_replay.py",
    "own7_qualify_e1.py",
)


def audit_manifest(registration, inputs, seed, first, paths, helpers):
    assert registration["status"] == "frozen-before-formal-execution"
    assert registration["qualification_ledger_slot"] == 7
    assert registration["source_commit"] == SOURCE and seed in SEEDS
    assert inputs["protocol"]["sha256"] == sha(paths["registration"])
    return {
        "schema": "ufuk-search-acting-formal7-audit-manifest-v1",
        "status": registration["status"],
        "qualification_ledger_slot": 7,
        "source_commit": SOURCE,
        "fixed_epochs": registration["fixed_epochs"],
        "neural_audit_epochs": registration["neural_audit_epochs"],
        "neural_witness_K": 8,
        "original_training_started_epoch": first,
        "original_training_deadline_epoch": first + registration["whole_training_seconds_per_seed"],
        "whole_training_seconds": registration["whole_training_seconds_per_seed"],
        "absolute_audit_cutoff_epoch": registration["absolute_audit_cutoff_epoch"],
        "scheduling_helper_sha256": {
            "own7_schedule_v3.py": sha(Path(helpers) / "own7_schedule_v3.py")
        },
        "whole_audit_seconds": registration["whole_audit_seconds_from_originalfirstclock"],
        "producer_checkout": paths["repo"],
        "run": paths["run"],
        "inputs": inputs,
        "frozen_config": registration["configs"][str(seed)],
        "helper_sha256": {name: sha(Path(helpers) / name) for name in AUDIT_HELPERS},
    }


def main():
    p = argparse.ArgumentParser()
    for name in (
        "registration",
        "three-input-manifest",
        "qualification-config",
        "paths-config",
        "helpers",
        "output",
    ):
        p.add_argument("--" + name, type=Path, required=True)
        if name in (
            "registration",
            "three-input-manifest",
            "qualification-config",
            "paths-config",
        ):
            p.add_argument("--" + name + "-sha256", required=True)
    a = p.parse_args()
    for name in (
        "registration",
        "three_input_manifest",
        "qualification_config",
        "paths_config",
    ):
        assert sha(getattr(a, name)) == getattr(a, name + "_sha256")
    reg = json.loads(a.registration.read_text())
    original = json.loads(a.three_input_manifest.read_text())
    q = validate_config(json.loads(a.qualification_config.read_text()))
    paths = json.loads(a.paths_config.read_text())
    clock_path = Path(paths["common_original_firstclock_receipt"])
    assert sha(clock_path) == paths["common_original_firstclock_receipt_sha256"]
    clock = json.loads(clock_path.read_text())
    assert clock["schema"] == "own7-common-original-firstclock-v1"
    assert clock["slots"] == [7]
    common_first = clock["original_training_started_epoch"]
    assert type(common_first) in (int, float) and common_first <= time.time()
    assert reg["source_commit"] == q["source_commit"] == SOURCE
    assert reg["fixed_epochs"] == q["fixed_epochs"]
    assert reg["three_input_manifest_sha256"] == a.three_input_manifest_sha256
    for name, digest in q["helper_sha256"].items():
        assert sha(a.helpers / name) == digest
    assert list(map(int, paths["seeds"])) == list(SEEDS)
    a.output.mkdir(exist_ok=False, parents=True)
    inputs4 = {"seeds": {}}
    rows = []
    for seed in SEEDS:
        row = paths["seeds"][str(seed)]
        inputs = {
            **original["seeds"][str(seed)],
            "protocol": {
                "path": str(a.registration.resolve()),
                "sha256": a.registration_sha256,
            },
        }
        for info in inputs.values():
            assert sha(info["path"]) == info["sha256"]
        inputs4["seeds"][str(seed)] = inputs
        first = row["original_training_started_epoch"]
        assert first == common_first
        assert reg["earliest_training_epoch"] <= first <= time.time()
        from own7_schedule_v3 import validate_clock

        validate_clock(reg, first, first + reg["whole_training_seconds_per_seed"], time.time())
        full = audit_manifest(
            reg,
            inputs,
            seed,
            first,
            {
                "registration": str(a.registration),
                "repo": paths["repo"],
                "run": row["run"],
            },
            a.helpers,
        )
        manifest_path = a.output / f"audit-manifest-{seed}.json"
        publish(manifest_path, full)
        rows.append({**row, "seed": seed, "audit_manifest": str(manifest_path.resolve())})
    publish(a.output / "four-input-manifest.json", inputs4)
    post = {
        **paths["posttraining"],
        "source_commit": SOURCE,
        "repo": paths["repo"],
        "python": paths["python"],
        "helpers": str(a.helpers.resolve()),
        "qualification_config": str(a.qualification_config.resolve()),
        "qualification_config_sha256": a.qualification_config_sha256,
        "whole_training_seconds": reg["whole_training_seconds_per_seed"],
        "whole_audit_seconds": reg["whole_audit_seconds_from_originalfirstclock"],
        "absolute_audit_cutoff_epoch": reg["absolute_audit_cutoff_epoch"],
        "previous45_completion_barrier": reg["terminal45_barrier_binding"],
        "cohort67": reg["cohort67"],
        "seeds": rows,
    }
    publish(a.output / "posttraining-config.json", post)
    assert sha(a.helpers / "own67_cohort.py") == q["helper_sha256"]["own67_cohort.py"]
    import own67_cohort

    descriptor = json.loads(Path(reg["cohort67"]["manifest"]).read_text())
    assert sha(reg["cohort67"]["manifest"]) == reg["cohort67"]["manifest_sha256"]
    own67_cohort.register_member(descriptor, 7, a.registration, a.qualification_config)


if __name__ == "__main__":
    main()
