import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
REPO_ROOT = next(
    (p for p in ROOT.resolve().parents if (p / "pyproject.toml").is_file()), Path.cwd()
)
REPO = REPO_ROOT / "experiments/ufuk"
MC = ROOT / "confirmation3/fixtures"


def load(path, name):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "slot,prefix,folder",
    [(4, "ownv1", "ownsearch-method4"), (5, "own5", "search-acting-method5")],
)
def test_exact_empty_case_and_missing_real_nodes(slot, prefix, folder):
    module = load(ROOT / f"method{slot}/{prefix}_strength_analysis.py", f"a{slot}")
    arm = json.loads((MC / "two-real-games.json").read_text())["arm"]
    entry = {"book": str(MC / "book.json"), "final_sha256": arm["candidate_sha256"]}
    empty = next(g for g in arm["games"] if g["stockfish_nodes_by_move"] == [])
    module.validate_arm({**arm, "games": [empty]}, entry["book"], entry["final_sha256"])
    module.audit_trajectories(
        {**arm, "games": [empty]}, json.loads(Path(entry["book"]).read_text()), True
    )
    changed = copy.deepcopy(next(g for g in arm["games"] if g["stockfish_nodes_by_move"]))
    changed["stockfish_nodes_by_move"] = []
    with pytest.raises(ValueError):
        module.validate_arm({**arm, "games": [changed]}, entry["book"], entry["final_sha256"])


@pytest.mark.parametrize(
    "slot,prefix,folder",
    [(4, "ownv1", "ownsearch-method4"), (5, "own5", "search-acting-method5")],
)
def test_statistical_ast_and_every_original_gate_assert_preserved(slot, prefix, folder):
    binding = load(ROOT / f"method{slot}/analysis_repair_binding.py", f"b{slot}")
    for suffix in ("strength_analysis", "all_gates"):
        binding.unchanged_checks(
            REPO / folder / f"{prefix}_{suffix}.py",
            ROOT / f"method{slot}/{prefix}_{suffix}.py",
            suffix == "strength_analysis",
        )


@pytest.mark.parametrize(
    "slot,prefix,seeds",
    [(4, "ownv1", [20261425, 20261426]), (5, "own5", [20261525, 20261526])],
)
def test_unchanged_strict_strength_boundaries(slot, prefix, seeds):
    gate = load(ROOT / f"method{slot}/{prefix}_all_gates.py", f"g{slot}")
    rows = [
        {
            "seed": s,
            "strength_pass": True,
            "gates": {"all": True},
            "direct": {"mean": 0.61, "ci_adjusted_98_75": [0.51, 0.7]},
            "sf_paired_delta": {"mean": 0.11, "ci_adjusted_98_75": [0.01, 0.2]},
            "final_sf_mean": 0.25,
            "caps": dict(direct=0, final_sf=0, initial_sf=0),
            "direct_adversarial_caps": {"ci_adjusted_98_75": [0.51, 0.7]},
            "sf_delta_adversarial_caps": {"ci_adjusted_98_75": [0.01, 0.2]},
        }
        for s in seeds
    ]
    valid = {"replicated_strength_pass": True, "seed_results": rows}
    gate.require_strength_gates(valid)
    for field, key, value in [
        ("direct", "mean", 0.6),
        ("sf_paired_delta", "mean", 0.1),
        ("caps", "final_sf", 0.05001),
    ]:
        changed = copy.deepcopy(valid)
        changed["seed_results"][0][field][key] = value
        with pytest.raises(AssertionError):
            gate.require_strength_gates(changed)
