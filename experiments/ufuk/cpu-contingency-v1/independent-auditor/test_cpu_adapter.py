import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_actual_64_actor_chronology_and_masks():
    core = load("cpu_contingency_audit_core")
    groups = [dict(first_collection_index=i * 64, actor_batch_size=64) for i in range(256)]
    selected = core.choose_chronological_groups(groups, 256, 64)
    assert selected == set(range(8)) | set(range(248, 256))
    assert core.actor_packet_groups([0, 63, 64, 16383], 16384, 64) == [
        list(range(64)),
        list(range(64, 128)),
        list(range(16320, 16384)),
    ]
    groups[0]["actor_batch_size"] = 128
    with pytest.raises(AssertionError):
        core.choose_chronological_groups(groups, 256, 64)


def test_no_old_width_constant_inside_real_audit():
    tree = ast.parse((ROOT / "cpu_contingency_audit_core.py").read_text())
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "audit_epoch"
    )
    assert not any(
        isinstance(n, ast.Constant) and n.value in (128, 32768, "cuda:0")
        for n in ast.walk(function)
    )
    text = ast.unparse(function)
    assert "actors * config.epoch_steps" in text
    assert "actor_packet_groups(prescribed, len(epoch.actions), actors)" in text
    assert "choose_chronological_groups(groups, total_steps, actors)" in text
    assert "schedule_and_all_actor_search_rngs_verified" in text


def test_cpu_schema_and_new_seed_binding():
    controls = load("cpu_contingency_adapter_controls")
    assert controls.NATIVE_SCHEMA == "torch-search-acting-native-cpu-v3"
    assert controls.SEEDS == (20261925, 20261926)
    controls.validate_epoch_report({"epoch": 1, "source_commit": controls.SOURCE}, 1)
    with pytest.raises(ValueError):
        controls.validate_epoch_report({"epoch": 32, "source_commit": controls.SOURCE}, 1)


def test_actual_cpu_qualifier_keeps_six_mutations():
    text = (ROOT / "cpu_contingency_qualify_e1.py").read_text()
    assert 'training["cuda_rng"] == []' in text
    assert 'torch.__version__ == "2.14.1+cpu"' in text
    assert "torch.cuda.get_device_name" not in text and "torch.cuda.is_available" not in text
    tree = ast.parse(text)
    assignments = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "rejected" for t in n.targets)
    ]
    assert assignments
    assert "visited_loss" in text
    assert "len(rejected) == 6 and all(rejected.values())" in text
    assert 'report["optimizer_committed"] > 0' in text
    assert "Actual CPU E1 retained model must change" in text
