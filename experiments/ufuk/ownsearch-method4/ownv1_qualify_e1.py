"""Actual CUDA E1 DEVELOPMENT qualification of formal audit code; never formal E>=8."""

import argparse
import copy
import gzip
import json
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import ownv1_full_audit as audit
from ownv1_audit_support import check_source, guard, publish, sha

END = 1791170400


def reject(check, *args):
    try:
        check(*args)
    except AssertionError:
        return True
    raise RuntimeError("Corrupt actual-data mutation was accepted")


def main():
    if not __debug__:
        raise RuntimeError("Required audit assertions disabled")
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--deadline-epoch", type=float, required=True)
    a = p.parse_args()
    started = time.time()
    assert sha(a.manifest) == a.manifest_sha256
    spec = json.loads(a.manifest.read_text())
    assert spec["schema"] == "ufuk-ownv1-E1-auditor-development-qualification-v1"
    assert spec["scope"] == "existing-development-E1-not-formal-method4"
    assert a.deadline_epoch == spec["started_epoch"] + spec["whole_seconds"] < END
    assert 0 < spec["whole_seconds"] <= 900 and started < a.deadline_epoch
    run, checkout = Path(spec["run"]), Path(spec["checkout"])
    check_source(checkout, spec["source_commit"])
    assert spec["source_commit"] == "a278bba67bce962cb9294f0d24040e02e9acf1f4"
    for name, digest in spec["helper_sha256"].items():
        assert sha(Path(__file__).with_name(name)) == digest
    assert spec["helper_sha256"][Path(__file__).name] == sha(__file__)
    guard(a.deadline_epoch, a.output.parent)
    paths = {key: Path(value["path"]).resolve() for key, value in spec["inputs"].items()}
    assert set(paths) == {"initial_weights", "book", "experiment_config", "protocol"}
    for key, path in paths.items():
        assert sha(path) == spec["inputs"][key]["sha256"]
    assert spec["inputs"]["initial_weights"]["sha256"] == (
        "e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03"
    )
    cp = run / "checkpoints"
    development = json.loads(Path(spec["development_controller_result"]).read_text())
    assert development["status"] == "pass-one-development-epoch-and-readonly-audit"
    assert development["source_commit"] == spec["source_commit"]
    assert development["finished_epoch"] <= spec["original_profile_deadline_epoch"]
    immutable = {str(path): sha(path) for path in run.rglob("*") if path.is_file()}
    meta = json.loads((run / "metadata.json").read_text())
    assert meta["schema"] == "ownsearch-supervised-run-v1"
    assert meta["source_commit"] == spec["source_commit"]
    assert meta["max_epochs"] == meta["checkpoint_interval"] == 1
    assert meta["absolute_deadline_epoch"] == spec["original_profile_deadline_epoch"]
    import torch

    import harbichess.training.torch_ownsearch_checkpoint as cm
    import harbichess.training.torch_ownsearch_learner as lm
    from harbichess.backends.torch_network import load_weights
    from harbichess.selfplay.online_actor import OnlineActorConfig
    from harbichess.selfplay.online_epoch import deserialize_policy_epoch
    from harbichess.training.ownsearch_targets import OwnSearchConfig
    from harbichess.training.torch_fullgame_ppo import FullGamePPOTrainConfig
    from harbichess.training.torch_ownsearch_core import OwnSearchObjective
    from harbichess.training.torch_ownsearch_learner import (
        TorchOwnSearchConfig,
        TorchOwnSearchLearner,
        canonical,
        tensor_bits_equal,
    )

    assert torch.cuda.is_available(), "Actual A100 CUDA required; no fallback or skip"
    assert "A100" in torch.cuda.get_device_name()
    assert Path(lm.__file__).resolve().is_relative_to(checkout.resolve() / "src")
    cfg = json.loads(paths["experiment_config"].read_text())
    assert cfg == spec["frozen_config"] and cfg["learning_rate"] == 0.000025
    assert cfg["device"] == "cuda:0" and cfg["actors"]["games"] == 128
    assert cfg["epoch_steps"] == 256 and cfg["actors"]["temperature"] == 1
    for key, cls in [
        ("actors", OnlineActorConfig),
        ("search", OwnSearchConfig),
        ("objective", OwnSearchObjective),
        ("schedule", FullGamePPOTrainConfig),
    ]:
        cfg[key] = cls(**cfg[key])
    config = TorchOwnSearchConfig(**cfg)
    pool = lm.read_online_train_book(paths["book"])

    def cached_book(path):
        assert Path(path).resolve() == paths["book"]
        assert sha(path) == spec["inputs"]["book"]["sha256"]
        return pool

    lm.read_online_train_book = cm.read_online_train_book = cached_book
    initial = TorchOwnSearchLearner.resume(
        cp / "epoch-00000000", config=config, input_paths=paths, source_commit=spec["source_commit"]
    )
    assert initial.epoch == 0 and initial.closed
    original = load_weights(paths["initial_weights"]).eval()
    for name, value in original.state_dict().items():
        for model in (initial.online, initial.behavior, initial.base):
            audit.compare_frozen_storage(
                value.cpu(), model.state_dict()[name].cpu(), tensor_bits_equal
            )
    del original
    for epoch in (0, 1):
        guard(a.deadline_epoch, run)
        training = torch.load(
            cp / f"epoch-{epoch:08d}" / "training.pt", map_location="cpu", weights_only=True
        )
        required = {
            "online",
            "base",
            "behavior",
            "optimizer",
            "actor_rng",
            "sampler_seed_rng",
            "schedule_rng",
            "search_rngs",
            "last_sampler_rng_state",
            "python_rng",
            "numpy_rng",
            "torch_rng",
            "cuda_rng",
            "state",
        }
        assert set(training) == required and len(training["search_rngs"]) == 128
        assert len(training["cuda_rng"]) == torch.cuda.device_count()
        assert training["torch_rng"].dtype == torch.uint8
        assert isinstance(training["optimizer"]["state"], dict)
        del training
    aa = SimpleNamespace(
        run=run, source_commit=spec["source_commit"], deadline_epoch=a.deadline_epoch
    )
    final, report = audit.audit_epoch(aa, config, paths, initial, 1, True)
    assert report["epoch"] == 1
    assert report["raw_actor_replayed"] == 32768
    assert (
        report["prescribed_neural_roots_verified"]
        == report["raw_actor_packet_roots_verified"]
        == 18
    )
    assert report["exact_fullgroup_neural_roots_recomputed"] >= 18
    assert report["optimizer_updates_performed_by_audit"] == 0
    record = json.loads(gzip.decompress(final.last_epoch_gzip))
    epoch = deserialize_policy_epoch(canonical(record["collection"]))
    index = next(
        i for i in report["prescribed_collection_indices"] if len(epoch.actions[i].base_policy) >= 2
    )
    row = epoch.actions[index]
    reference = report["raw_actor_packet_recomputed_reference"][index]
    reference_args = tuple(reference[name] for name in ("policy", "base_policy", "wdl", "base_wdl"))
    audit.compare_actor_packet(row, *reference_args)
    packet = SimpleNamespace(
        **{
            name: getattr(row, name)
            for name in ("policy", "base_policy", "online_pre_wdl", "base_wdl", "behavior_policy")
        }
    )
    probabilities = list(packet.base_policy)
    mass = min(probabilities[1] / 2, 1e-6)
    assert mass > 0
    probabilities[0] += mass
    probabilities[1] -= mass
    packet.base_policy = tuple(probabilities)
    rejected = {
        "raw_base_policy": reject(
            audit.compare_actor_packet,
            packet,
            *reference_args,
        )
    }
    groups = defaultdict(list)
    for action in epoch.actions:
        groups[(action.transition.source_id, action.transition.game_index)].append(action)
    expected = None
    for rows in groups.values():
        outcome = final.actors.rules.outcome(
            rows[-1].transition.post, claim_draw=config.actors.claim_draw
        )
        if outcome is not None:
            value = outcome.value_for(final.actors.rules.view(rows[0].transition.pre).side_to_move)
            expected = tuple(float(value == v) for v in (1, 0, -1))
            break
    assert expected is not None, "Actual development archive must include a terminal witness"
    audit.compare_terminal_packet(expected, expected)
    rejected["terminal_WDL"] = reject(
        audit.compare_terminal_packet, expected[1:] + expected[:1], expected
    )
    root = record["own_search"]["roots"][0]
    corrupt = copy.deepcopy(root)
    corrupt["search_policy"] = [*root["search_policy"]]
    corrupt["search_policy"][0] += 1e-6
    rejected["search_policy"] = reject(audit.compare_search_packet, corrupt, root, canonical)
    tensor = final.base.state_dict()["material_value_linear.weight"].detach().cpu().contiguous()
    corrupt_tensor = tensor.clone()
    raw = corrupt_tensor.reshape(-1).view(torch.uint8)
    raw[0] ^= 1
    rejected["frozen_storage_onebit"] = reject(
        audit.compare_frozen_storage, corrupt_tensor, tensor, tensor_bits_equal
    )
    assert all(rejected.values())
    guard(a.deadline_epoch, run)
    assert immutable == {str(path): sha(path) for path in run.rglob("*") if path.is_file()}
    for key, path in paths.items():
        assert sha(path) == spec["inputs"][key]["sha256"]
    check_source(checkout, spec["source_commit"])
    publish(
        a.output,
        {
            "schema": "ufuk-ownv1-E1-auditor-development-qualification-result-v1",
            "status": (
                "pass-actualCUDA-E1-full-data-original-groups-raw-packets-and-targeted-mutations"
            ),
            "scope": "DEVELOPMENT ONLY: native0+1, not allformal E+1 or strength eligibility",
            "source_commit": spec["source_commit"],
            "manifest_sha256": sha(a.manifest),
            "helper_sha256": spec["helper_sha256"],
            "training_native_untouched_sha256": immutable,
            "actual_device_name": torch.cuda.get_device_name(),
            "torch_version": torch.__version__,
            "audit_report": report,
            "targeted_actual_data_mutations_rejected": rejected,
            "mutation_scope": "Same production comparison functions on actual validated "
            "packets/tensors; "
            "no four complete corrupt-native replays and no on-disk mutation",
            "optimizer_updates_performed_by_qualification": 0,
            "new_selfplay_transitions_generated": 0,
            "started_epoch": started,
            "finished_epoch": time.time(),
            "absolute_deadline_epoch": a.deadline_epoch,
        },
    )


if __name__ == "__main__":
    main()
