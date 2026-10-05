"""Fail-closed formal6 adapters; these checks do not register or authorize a run."""

from pathlib import Path

SOURCE = "428a30f5e1658f3cf159844db547ff0147ade5a9"
SEEDS = (20261625, 20261626)
END = 1791180000
HELPERS = {
    "own6_adapter_controls.py",
    "own6_audit_support.py",
    "own6_audit_core.py",
    "own6_full_audit.py",
    "own6_fresh_cli_replay.py",
}


def validate_spec(spec):
    if (
        spec.get("schema") != "ufuk-search-acting-formal6-audit-manifest-v1"
        or spec.get("status") != "frozen-before-formal-execution"
        or spec.get("qualification_ledger_slot") != 6
        or spec.get("source_commit") != SOURCE
    ):
        raise ValueError("requires separately frozen formal6/new-v2 source manifest")
    epochs = spec["fixed_epochs"]
    if type(epochs) is not int or epochs < 8:
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
        or spec["original_training_deadline_epoch"]
        != first + spec["whole_training_seconds"]
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
        or cfg["schedule"]
        != {"minibatch_size": 1024, "passes": 4, "max_gradient_norm": 5.0}
        or cfg["learning_rate"] != 0.000025
        or cfg["weight_decay"] != 0.0001
    ):
        raise ValueError(
            "requires exact controlled sparse search-acting-v2 shape/objective"
        )
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
    """Bind receipt numbering to native epoch, never a loop's collection index."""
    if (
        type(report.get("epoch")) is not int
        or report["epoch"] != expected_epoch
        or report.get("source_commit") != SOURCE
    ):
        raise ValueError("audit report epoch/source differs from current native")
