"""Pure prospective bindings and conjunctive gates; synthetic data only, no inference."""

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import ownv1_strength_config as config


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen():
    return {
        "schema": "ufuk-ownv1-strength-bindings-v1",
        "status": "frozen-before-formal-execution",
        "qualification_ledger_slot": 4,
        "MAX_families": 8,
        "seeds": list(config.SEEDS),
        "books_sha256": {str(k): v for k, v in config.BOOKS.items()},
        "fixed_epochs": 24,
        "source_commit": "a" * 40,
        "native_schema": "torch-ownsearch-native-cuda-v1",
        "games_per_arm": 96,
        "arm_wall_seconds": 3600,
        "runtime_scope": "same-A100-CPU-one-thread-Torch2.11-all-six-arms",
    }


def test_all_four_book_maps_same_new_seed_and_no_MC_schema_epoch_assumption():
    for name in (
        "ownv1_baseline",
        "ownv1_final_arms",
        "ownv1_strength_analysis",
        "ownv1_all_gates",
    ):
        assert load(name).BOOKS == config.BOOKS
    c = frozen()
    config.validate_config(c)
    for key, bad in [
        ("source_commit", "2312652dc52a894e9726f48321117cf114270355"),
        ("seeds", [20261205, 20261206]),
        ("native_schema", "torch-fullgame-native-cuda-v1"),
        ("fixed_epochs", 7),
        ("MAX_families", 9),
    ]:
        mutated = {**c, key: bad}
        with pytest.raises(AssertionError):
            config.validate_config(mutated)
    mutated = copy.deepcopy(c)
    mutated["books_sha256"][str(config.SEEDS[0])] = config.BOOKS[config.SEEDS[1]]
    with pytest.raises(AssertionError):
        config.validate_config(mutated)


def test_conjunctive_gate_strict_numeric_boundaries_not_boolean_flags_only():
    g = load("ownv1_all_gates")
    row = {
        "strength_pass": True,
        "gates": {"synthetic": True},
        "direct": {"mean": 0.7, "ci_adjusted_98_75": [0.6, 0.8]},
        "sf_paired_delta": {"mean": 0.2, "ci_adjusted_98_75": [0.1, 0.3]},
        "final_sf_mean": 0.3,
        "caps": {"direct": 0, "final_sf": 0, "initial_sf": 0},
        "direct_adversarial_caps": {"ci_adjusted_98_75": [0.6, 0.8]},
        "sf_delta_adversarial_caps": {"ci_adjusted_98_75": [0.1, 0.3]},
    }
    strength = {
        "replicated_strength_pass": True,
        "seed_results": [{**copy.deepcopy(row), "seed": seed} for seed in config.SEEDS],
    }
    g.require_strength_gates(strength)
    for field, key, boundary in [("direct", "mean", 0.60), ("sf_paired_delta", "mean", 0.10)]:
        mutated = copy.deepcopy(strength)
        mutated["seed_results"][1][field][key] = boundary
        with pytest.raises(AssertionError):
            g.require_strength_gates(mutated)
    mutated = copy.deepcopy(strength)
    mutated["seed_results"][0]["sf_delta_adversarial_caps"]["ci_adjusted_98_75"][0] = 0
    with pytest.raises(AssertionError):
        g.require_strength_gates(mutated)


def test_manifest_binding_rejects_actual_source_or_fixedE_drift(tmp_path):
    import json
    from types import SimpleNamespace

    path = tmp_path / "manifest.json"
    value = {"source_commit": "a" * 40, "fixed_epochs": 24, "qualification_ledger_slot": 4}
    path.write_text(json.dumps(value))
    config.validate_binding(SimpleNamespace(manifest=path), frozen())
    for key, bad in [
        ("source_commit", "b" * 40),
        ("fixed_epochs", 40),
        ("qualification_ledger_slot", 3),
    ]:
        path.write_text(json.dumps({**value, key: bad}))
        with pytest.raises(AssertionError):
            config.validate_binding(SimpleNamespace(manifest=path), frozen())
