import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
ORIGINAL = Path("/workspace/HarbiChess/experiments/ufuk/own-terminal-visited-loss/frozen-helper-v1")


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def function_map(path):
    return {
        n.name: ast.dump(n, include_attributes=False)
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef)
    }


def test_every_statistical_function_exact_AST():
    old = function_map(ORIGINAL / "own8_strength_analysis.py")
    new = function_map(ROOT / "cpu_contingency_strength_analysis.py")
    assert old.keys() == new.keys()
    for name in old.keys() - {"main"}:
        assert old[name] == new[name], name
    assert (
        function_map(ORIGINAL / "own8_all_gates.py")["require_strength_gates"]
        == function_map(ROOT / "cpu_contingency_all_gates.py")["require_strength_gates"]
    )


def test_cpu_seed_schema_books_fail_closed():
    module = load("cpu_contingency_strength_config")
    cfg = dict(
        schema="ufuk-cpu-contingency-strength-bindings-v1",
        status="frozen-before-formal-execution",
        qualification_ledger_slot=8,
        MAX_families=8,
        seeds=list(module.SEEDS),
        books_sha256={str(k): v for k, v in module.FROZEN_BOOKS.items()},
        books_provenance_sha256="a" * 64,
        fixed_epochs=8,
        source_commit=module.SOURCE,
        native_schema="torch-search-acting-native-cpu-v3",
        games_per_arm=96,
        arm_wall_seconds=3600,
        runtime_scope="same-PRIMARY-CPU-one-thread-Torch2.14.1-all-six-arms",
        protocol_id="cpu-contingency-v1",
        runtime_torch_version="2.14.1+cpu",
        ledger_slot8_replacement_receipt_sha256="b" * 64,
    )
    module.validate_config(cfg)
    for key, value in [
        ("seeds", [20261825, 20261826]),
        ("native_schema", "torch-search-acting-native-cuda-v3"),
        ("runtime_scope", "same-A100-CPU-one-thread-Torch2.11-all-six-arms"),
        ("ledger_slot8_replacement_receipt_sha256", "ROOT_UNKNOWN"),
    ]:
        with pytest.raises(AssertionError):
            module.validate_config({**cfg, key: value})
    with pytest.raises(AssertionError):
        module.validate_config({**cfg, "books_sha256": {str(k): "c" * 64 for k in module.SEEDS}})


def test_both_arena_argparse_use_real_imported_SEEDS():
    for name in ("baseline", "final_arms"):
        module = load(f"cpu_contingency_{name}")
        assert module.SEEDS == (20261925, 20261926)
        text = (ROOT / f"cpu_contingency_{name}.py").read_text()
        assert "choices=SEEDS" in text and "c.SEEDS" not in text
        assert "15 * 1024**3" in text and "256 * 1024**2" in text


def test_no_false_CUDA_export_parity():
    text = (ROOT / "cpu_contingency_parity.py").read_text()
    assert "torch.cuda.is_available" not in text and '"cuda:0"' not in text
    assert "restore_native(entry)" in text and "TorchSearchActingLearner.resume" in text
    assert 'choices=("cpu", "mlx")' in text
    allgates = (ROOT / "cpu_contingency_all_gates.py").read_text()
    assert '"cpu_export_parity", "cpu"' in allgates
    assert "cuda_parity" not in allgates
