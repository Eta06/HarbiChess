"""Versioned analysis-only binding; original producer, registrations and Q stay immutable."""

import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unchanged_checks(original, repaired, analysis):
    old = ast.parse(original.read_text())
    new = ast.parse(repaired.read_text())

    def functions(tree):
        return {
            n.name: ast.dump(n, include_attributes=False)
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
        }

    before, after = functions(old), functions(new)
    names = (
        ("digest", "block_scores", "intervals", "assess_seed", "audit_trajectories")
        if analysis
        else ("sha", "read", "require_strength_gates")
    )
    assert all(before[name] == after[name] for name in names)
    if not analysis:
        old = normalize_primary_count(old)
        old_asserts = {
            ast.dump(n, include_attributes=False)
            for n in ast.walk(old)
            if isinstance(n, ast.Assert)
        }
        new_asserts = {
            ast.dump(n, include_attributes=False)
            for n in ast.walk(new)
            if isinstance(n, ast.Assert)
        }
        assert old_asserts <= new_asserts


def normalize_primary_count(tree):
    class CorrectMetadata(ast.NodeTransformer):
        def visit_Compare(self, node):
            self.generic_visit(node)
            if (
                isinstance(node.left, ast.Subscript)
                and isinstance(node.left.value, ast.Name)
                and node.left.value.id == "strength"
                and isinstance(node.left.slice, ast.Constant)
                and node.left.slice.value == "primary_comparisons"
                and len(node.ops) == 1
                and isinstance(node.ops[0], ast.Eq)
                and len(node.comparators) == 1
                and isinstance(node.comparators[0], ast.Constant)
                and node.comparators[0].value == 5
            ):
                node.comparators[0].value = 4
            return node

    return CorrectMetadata().visit(tree)


def bind_cli(namespace):
    if not __debug__:
        raise RuntimeError("Required assertions disabled")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--qualification-config", type=Path, required=True)
    parser.add_argument("--qualification-config-sha256", required=True)
    parser.add_argument("--analysis-supplement", type=Path, required=True)
    parser.add_argument("--analysis-supplement-sha256", required=True)
    parser.add_argument("--original-helpers", type=Path, required=True)
    parser.add_argument("--original-registration", type=Path, required=True)
    parser.add_argument("--previous-analysis-repair", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    assert sha(args.qualification_config) == args.qualification_config_sha256
    assert sha(args.analysis_supplement) == args.analysis_supplement_sha256
    supplement = json.loads(args.analysis_supplement.read_text())
    assert supplement["schema"] == "own5-analysis-only-empty-engine-and-primary-count-repair-v3"
    assert supplement["status"] == "registered-before-any-method4-or5-heldout-outcomes"
    assert supplement["original_qualification_config_sha256"] == args.qualification_config_sha256
    assert sha(args.original_registration) == supplement["original_registration_sha256"]
    slot = supplement["qualification_ledger_slot"]
    assert slot == 5
    assert supplement["metadata_repair_inventory"] == {"primary_comparisons": {"old": 5, "new": 4}}
    assert sha(args.previous_analysis_repair) == supplement["previous_v2_supplement_sha256"]
    previous = json.loads(args.previous_analysis_repair.read_text())
    assert previous["repair_helper_sha256"] == supplement["previous_v2_helper_sha256"]
    for filename, digest in previous["repair_helper_sha256"].items():
        assert sha(args.previous_analysis_repair.parent / filename) == digest
    prefix = "ownv1" if slot == 4 else "own5"
    binding = importlib.import_module(prefix + "_strength_config")
    q = binding.validate_config(json.loads(args.qualification_config.read_text()))
    assert q["qualification_ledger_slot"] == slot
    original_reg = json.loads(args.original_registration.read_text())
    assert original_reg["qualification_ledger_slot"] == slot
    assert original_reg["source_commit"] == q["source_commit"] == supplement["source_commit"]
    assert original_reg["fixed_epochs"] == q["fixed_epochs"] == supplement["fixed_epochs"]
    assert q["seeds"] == supplement["seeds"]
    repaired = Path(namespace["__file__"])
    for name, digest in supplement["original_helper_sha256"].items():
        assert q["helper_sha256"][name] == digest
        assert sha(args.original_helpers / name) == digest
    for name, digest in supplement["repair_helper_sha256"].items():
        assert sha(repaired.parent / name) == digest
    assert repaired.name in supplement["repair_helper_sha256"]
    assert (
        sha(Path(binding.__file__))
        == supplement["original_helper_sha256"][prefix + "_strength_config.py"]
    )
    for suffix in ("strength_analysis", "all_gates"):
        name = prefix + "_" + suffix + ".py"
        unchanged_checks(
            args.original_helpers / name,
            repaired.parent / name,
            suffix == "strength_analysis",
        )
    package = importlib.util.find_spec("harbichess")
    assert package is not None and package.origin is not None
    checkout = Path(package.origin).resolve().parents[2]
    assert (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
        == q["source_commit"]
    )
    assert not subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=checkout, text=True
    ).strip()
    namespace.update(
        Q=q,
        SEEDS=binding.SEEDS,
        BOOKS=binding.BOOKS,
        REPAIR_SHA=args.analysis_supplement_sha256,
        ORIGINAL_Q_SHA=args.qualification_config_sha256,
    )
    sys.argv = [sys.argv[0], *remaining]
