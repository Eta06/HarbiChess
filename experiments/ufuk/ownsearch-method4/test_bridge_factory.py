"""Mock file/path/config tests only: no SSH, models, MLX or inference execution."""

import json
from pathlib import Path

import ownv1_formal_config_factory as factory
import ownv1_mlx_receipt_bridge as bridge
import pytest
from ownv1_strength_config import SEEDS


def test_bridge_both_exact_E_paths_hashes_and_source_no_MC_alias(tmp_path):
    mirror = tmp_path / "mirror"
    config = {"qualification_ledger_slot": 4, "fixed_epochs": 24, "source_commit": "a" * 40}
    runs = {
        str(seed): f"/content/harbichess-runs/ownsearch-method4-seed-{seed}/run" for seed in SEEDS
    }
    manifest = {**config, "seeds": []}
    for seed in SEEDS:
        remote = Path(runs[str(seed)]) / "checkpoints/epoch-00000024/model.safetensors"
        local = mirror / remote.relative_to("/content")
        local.parent.mkdir(parents=True)
        local.write_bytes(f"synthetic-path-only-not-model-{seed}".encode())
        manifest["seeds"].append(
            {"seed": seed, "candidate": str(remote), "candidate_sha256": bridge.sha(local)}
        )
    localized = bridge.localize_manifest(manifest, config, mirror, runs)
    assert [row["seed"] for row in localized["seeds"]] == list(SEEDS)
    for bad in ({**manifest, "fixed_epochs": 40}, {**manifest, "source_commit": "b" * 40}):
        with pytest.raises(AssertionError):
            bridge.localize_manifest(bad, config, mirror, runs)
    mutated = json.loads(json.dumps(manifest))
    mutated["seeds"][0]["candidate"] = mutated["seeds"][0]["candidate"].replace(
        "00000024", "00000040"
    )
    with pytest.raises(AssertionError):
        bridge.localize_manifest(mutated, config, mirror, runs)
    command = bridge.parity_command(
        "python", "ownv1_parity.py", "q.json", "a" * 64, "models.json", "probes.json", "out.json"
    )
    assert command[command.index("--backend") + 1] == "mlx"
    assert "--qualification-config" in command and "--qualification-config-sha256" in command


def test_factory_E_explicit_and_typed_no_strength_or_adaptive_selection():
    assert factory.schedules(24) == [1, 2, 6, 12, 18, 24]
    with pytest.raises(AssertionError):
        factory.schedules(7)
    assert factory.schedules(8) == [1, 2, 3, 4, 6, 8]
    configs = factory.configs(24)
    assert tuple(map(int, configs)) == SEEDS
    assert all(row["learning_rate"] == 0.000025 for row in configs.values())


def test_factory_rejects_infrastructure_only_zero_committed_updates():
    profile = {
        "source_commit": factory.SOURCE,
        "status": "pass-one-development-epoch-and-readonly-audit",
        "finished_epoch": 1,
        "absolute_deadline_epoch": 2,
        "audit_result": {"optimizer_committed": 0},
    }
    qualification = {
        "source_commit": factory.SOURCE,
        "status": "pass-actualCUDA-E1-full-data-original-groups-raw-packets-and-targeted-mutations",
        "audit_report": {"epoch": 1, "raw_actor_replayed": 32768, "optimizer_committed": 0},
    }
    with pytest.raises(AssertionError):
        factory.check_training_evidence(profile, qualification)


def test_factory_model_change_uses_actual_storage_not_metadata_or_numeric_equality(monkeypatch):
    import numpy as np
    import safetensors.numpy

    def unchanged(path):
        return {"weight": np.array([0.0], dtype=np.float32)}

    monkeypatch.setattr(safetensors.numpy, "load_file", unchanged)
    with pytest.raises(AssertionError):
        factory.actual_model_changed("mock-path-no-files-read")

    def signedzero_change(path):
        value = -0.0 if "00000001" in path else 0.0
        return {"weight": np.array([value], dtype=np.float32)}

    monkeypatch.setattr(safetensors.numpy, "load_file", signedzero_change)
    assert factory.actual_model_changed("mock-path-no-files-read") == ["weight"]
