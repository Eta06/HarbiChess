import gzip
import json
import shutil
import subprocess
import sys

import pytest
import torch
from test_prepared_oracle import make_dataset

from harbichess.backends.torch_network import TorchChessNetwork, save_weights
from harbichess.core.network_config import NetworkConfig
from harbichess.training.controlled_oracle_train import common_panels, run
from harbichess.training.merge_oracle import merge
from harbichess.training.oracle_data import read_game
from harbichess.training.prepared_oracle import PreparedOracle, digest, prepare
from harbichess.training.torch_checkpoint import load_checkpoint

GROUPS = (("anchor", 0, 100000), ("broader", 200000, 300000))


def inputs(tmp_path):
    anchor, broader, merged = (tmp_path / name for name in ("anchor", "broader", "merged"))
    make_dataset(anchor)
    make_dataset(broader)
    merge(merged, anchor, broader)
    anchor_cache, merged_cache = tmp_path / "anchor-cache", tmp_path / "merged-cache"
    prepare(anchor_cache, source=anchor)
    prepare(merged_cache, source=merged)
    return anchor, merged, anchor_cache, merged_cache


def test_common_validation_is_identical_for_both_data_arms_and_excludes_union_overlap(tmp_path):
    anchor, merged, anchor_cache, merged_cache = inputs(tmp_path)
    common = PreparedOracle(merged_cache, source=merged)
    controls = PreparedOracle(anchor_cache, source=anchor)
    arguments = dict(train_cap=None, validation_cap=1, seed=101)
    old_train, old_panels, old_info = common_panels(controls, common, GROUPS, **arguments)
    new_train, new_panels, new_info = common_panels(common, common, GROUPS, **arguments)
    assert old_train.size == 3 and new_train.size == 5
    assert old_info["groups"] == new_info["groups"]
    assert old_info["common_validation_overlap_removed"] == 2
    union_training = set(common.position_key[common.split == 0])
    for name in old_panels:
        left, right = old_panels[name], new_panels[name]
        assert left.panel.rows.tolist() == right.panel.rows.tolist()
        assert all(common.position_key[i] not in union_training for i in left.panel.rows)
        a, b = left.select((0,)), right.select((0,))
        for field in ("inputs", "policies", "wdl", "legal_masks", "value_weights"):
            assert torch.equal(getattr(a, field), getattr(b, field))


@pytest.mark.parametrize(
    "groups",
    [
        (),
        (("all", 0, 300000), ("overlap", 0, 100000)),
        (("missing", 0, 100000),),
        (("bad", -1, 300000),),
    ],
)
def test_invalid_common_validation_groups_refused(tmp_path, groups):
    _, merged, _, cache = inputs(tmp_path)
    common = PreparedOracle(cache, source=merged)
    with pytest.raises(ValueError):
        common_panels(common, common, groups, train_cap=None, validation_cap=None, seed=1)


def test_training_keys_missing_from_common_validation_source_refused(tmp_path):
    anchor, merged, _, merged_cache = inputs(tmp_path)
    original = anchor
    anchor = tmp_path / "changed-arm"
    shutil.copytree(original, anchor)
    game = read_game(anchor / "train.json.gz")
    # A valid heldout row becomes training in a separate arm; common union must
    # reject this arm rather than expose its training position as validation.
    row = read_game(anchor / "validation.json.gz")["rows"][1]
    row["split"], row["family"] = "train", 0
    game["rows"].append(row)
    (anchor / "train.json.gz").write_bytes(gzip.compress(json.dumps(game).encode(), mtime=0))
    manifest = json.loads((anchor / "dataset.json").read_text())
    manifest["rows"] += 1
    manifest["files"]["train.json.gz"] = digest(anchor / "train.json.gz")
    (anchor / "dataset.json").write_text(json.dumps(manifest))
    different = tmp_path / "different-cache"
    prepare(different, source=anchor)
    common = PreparedOracle(merged_cache, source=merged)
    with pytest.raises(ValueError, match="training keys"):
        common_panels(
            PreparedOracle(different, source=anchor),
            common,
            GROUPS,
            train_cap=None,
            validation_cap=None,
            seed=1,
        )


def test_real_dual_validation_fresh_process_resume_preserves_optimizer_rng_and_selection(tmp_path):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    _, merged, _, cache = inputs(tmp_path)
    weights = tmp_path / "initial.safetensors"
    torch.manual_seed(20261026)
    network = TorchChessNetwork(
        NetworkConfig(trunk_channels=4, residual_blocks=1, value_hidden=4),
        architecture="pairwise",
        invariant={"channels": 4, "blocks": 1, "hidden": 4},
    )
    save_weights(weights, network)
    straight, restarted = tmp_path / "straight", tmp_path / "restarted"
    kwargs = dict(
        validation_dataset=merged,
        validation_cache=cache,
        groups=GROUPS,
        max_steps=4,
        interval=1,
        patience=10,
        batch_size=2,
    )
    direct = run(straight, merged, cache, weights, **kwargs)
    command = [
        sys.executable,
        "-m",
        "harbichess.training.controlled_oracle_train",
        str(restarted),
        str(merged),
        str(cache),
        str(weights),
        str(merged),
        str(cache),
        "--group",
        "anchor",
        "0",
        "100000",
        "--group",
        "broader",
        "200000",
        "300000",
        "--max-steps",
        "4",
        "--interval",
        "1",
        "--patience",
        "10",
        "--batch-size",
        "2",
    ]
    subprocess.run([*command, "--stop-at", "2"], check=True, capture_output=True, text=True)
    subprocess.run(
        [*command, "--resume", str(restarted / "checkpoints/step-000002")],
        check=True,
        capture_output=True,
        text=True,
    )
    resumed = json.loads((restarted / "result-step-000004.json").read_text())
    assert direct["sample_trace_sha256"] == resumed["sample_trace_sha256"]
    assert direct["initial_validation"] == resumed["initial_validation"]
    assert direct["best_validation"] == resumed["best_validation"]
    loaded = []
    for directory in (straight, restarted):
        checkpoint = directory / "checkpoints/step-000004"
        manifest = json.loads((checkpoint / "checkpoint.json").read_text())
        learner, actual, _ = load_checkpoint(checkpoint, expected_run_config=manifest["run_config"])
        assert actual["transfer"] == "full-training"
        assert all(not name.startswith("material_value_linear.") for name in actual["trainable"])
        assert {"anchor", "broader"} == actual["run_state"]["evaluations"][-1]["groups"].keys()
        loaded.append((learner, actual, torch.load(checkpoint / "training.pt", weights_only=True)))
    a, b = loaded
    assert a[1]["run_state"] == b[1]["run_state"]
    assert torch.equal(a[2]["torch_rng"], b[2]["torch_rng"])
    for name, value in a[0].network.state_dict().items():
        assert torch.equal(value, b[0].network.state_dict()[name])
        if name.startswith("material_value_linear."):
            assert torch.equal(value, network.state_dict()[name])
    assert any(
        not torch.equal(value, network.state_dict()[name])
        for name, value in a[0].network.state_dict().items()
    )
    for identifier, entry in a[0].optimizer.state_dict()["state"].items():
        for name, value in entry.items():
            assert torch.equal(value, b[0].optimizer.state_dict()["state"][identifier][name])
    with pytest.raises(ValueError, match="configuration"):
        run(
            restarted,
            merged,
            cache,
            weights,
            resume=restarted / "checkpoints/step-000002",
            **{**kwargs, "seed": 1},
        )
