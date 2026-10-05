"""Fail-closed method8 audit template; source and producer remain deliberately unfrozen."""

from pathlib import Path

# Replace only in a newly reviewed/frozen method8 manifest. The port is not runnable
# as a formal qualifier until the new producer commit and producer schemas are known.
SOURCE = "3be5b87db27a0fbde83464e7ea7157f0d9a76ae4"
RUN_SCHEMA = "search-acting-supervised-run-v2"
JOURNAL_SCHEMA = "torch-fresh-sparse-search-acting-v3"
NATIVE_SCHEMA = "torch-search-acting-native-cuda-v3"
TRAINING_SCHEMA = "search-acting-supervised-train-v2"
SEEDS = (20261825, 20261826)
END = 1791180000
HELPERS = {
    "own8_adapter_controls.py",
    "own8_audit_support.py",
    "own8_audit_core.py",
    "own8_full_audit.py",
    "own8_fresh_cli_replay.py",
    "own8_qualify_e1.py",
}


def validate_spec(spec):
    if len(SOURCE) != 40 or any(c not in "0123456789abcdef" for c in SOURCE):
        raise RuntimeError("method8 producer source/schema must be frozen before qualification")
    if (
        spec.get("schema") != "ufuk-search-acting-formal8-audit-manifest-v1"
        or spec.get("status") != "frozen-before-formal-execution"
        or spec.get("qualification_ledger_slot") != 8
        or spec.get("source_commit") != SOURCE
    ):
        raise ValueError("requires separately frozen formal8/new-source manifest")
    epochs = spec["fixed_epochs"]
    if type(epochs) is not int or epochs != 8:
        raise ValueError("requires prospective fixed E>=8")
    selected = spec["neural_audit_epochs"]
    if (
        len(selected) != 6
        or len(set(selected)) != 6
        or selected != sorted(selected)
        or selected[:2] != [1, 2]
        or any(type(i) is not int or not 1 <= i <= epochs for i in selected)
    ):
        raise ValueError("requires six prospective neural epochs including1,2")
    if spec["neural_witness_K"] != 8:
        raise ValueError("chronological first8/last8 mask witness is fixed")
    first = spec["original_training_started_epoch"]
    if (
        spec["whole_training_seconds"] <= 0
        or spec["whole_audit_seconds"] <= 0
        or spec["original_training_deadline_epoch"] != first + spec["whole_training_seconds"]
        or not first < spec["original_training_deadline_epoch"] <= END
    ):
        raise ValueError("original clocks/budgets must be frozen within hard deadline")
    cfg = spec["frozen_config"]
    if (
        cfg["seed"] not in SEEDS
        or cfg["device"] != "cuda:0"
        or cfg["epoch_steps"] != 256
        or cfg["actors"]
        != {
            "games": 128,
            "max_additional_plies": 256,
            "claim_draw": True,
            "temperature": 1.0,
        }
        or cfg["search"]
        != {
            "simulations": 16,
            "max_considered_actions": 4,
            "gumbel_scale": 0.0,
            "value_scale": 0.1,
            "maxvisit_init": 50.0,
            "block_plies": 8,
        }
        or cfg["objective"]
        != {
            "policy_weight": 1.0,
            "value_weight": 0.2,
            "policy_anchor_weight": 0.03,
            "value_anchor_weight": 0.02,
            "behavior_kl_stop": 0.02,
        }
        or cfg["schedule"] != {"minibatch_size": 1024, "passes": 4, "max_gradient_norm": 5.0}
        or cfg["learning_rate"] != 0.000025
        or cfg["weight_decay"] != 0.0001
    ):
        raise ValueError("requires the controlled 16-simulation certificate-search shape")
    if set(spec["inputs"]) != {
        "initial_weights",
        "book",
        "experiment_config",
        "protocol",
    }:
        raise ValueError("requires four original native inputs")
    if set(spec["helper_sha256"]) != HELPERS:
        raise ValueError("complete frozen helper inventory required")
    for name, digest in spec["helper_sha256"].items():
        if (
            Path(name).name != name
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("invalid frozen helper provenance")


def validate_epoch_report(report, expected_epoch):
    if (
        type(report.get("epoch")) is not int
        or report["epoch"] != expected_epoch
        or report.get("source_commit") != SOURCE
    ):
        raise ValueError("audit report epoch/source differs from current native")
