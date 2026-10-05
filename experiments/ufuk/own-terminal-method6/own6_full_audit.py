"""Incremental formal6 v2 full-native/actual-mu/UNKNOWN/chronological own-search audit."""

import argparse
import json
import time
from pathlib import Path

from own6_adapter_controls import validate_epoch_report, validate_spec
from own6_audit_core import audit_epoch
from own6_audit_support import check_source, guard, publish, sha


def checked_audit_epoch(a, config, paths, initial, index, neural):
    final, report = audit_epoch(a, config, paths, initial, index, neural)
    validate_epoch_report(report, index)
    assert final.epoch == index
    return final, report


def main():
    if not __debug__:
        raise RuntimeError("Required assertions disabled")
    p = argparse.ArgumentParser()
    for name in ("manifest", "run", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--deadline-epoch", type=float, required=True)
    p.add_argument("--manifest-sha256", required=True)
    a = p.parse_args()
    assert sha(a.manifest) == a.manifest_sha256
    spec = json.loads(a.manifest.read_text())
    validate_spec(spec)
    assert a.run.resolve() == Path(spec["run"]).resolve()
    for name, digest in spec["helper_sha256"].items():
        assert sha(Path(__file__).with_name(name)) == digest
    a.checkout = Path(spec["producer_checkout"])
    a.source_commit = spec["source_commit"]
    epochs = spec["fixed_epochs"]
    assert type(epochs) is int and epochs >= 8
    assert spec["qualification_ledger_slot"] == 6
    first = spec["original_training_started_epoch"]
    assert (
        spec["original_training_deadline_epoch"]
        == first + spec["whole_training_seconds"]
    )
    assert spec["absolute_audit_cutoff_epoch"] == 1791170700
    assert a.deadline_epoch == min(first + spec["whole_audit_seconds"], 1791170700)
    assert first <= time.time() < a.deadline_epoch
    selected_epochs = spec["neural_audit_epochs"]
    assert len(set(selected_epochs)) == len(selected_epochs) == 6
    assert all(1 <= index <= epochs for index in selected_epochs)
    assert selected_epochs == sorted(selected_epochs) and selected_epochs[:2] == [1, 2]
    check_source(a.checkout, a.source_commit)
    guard(a.deadline_epoch, a.output.parent)
    paths = {
        name: Path(info["path"]).resolve() for name, info in spec["inputs"].items()
    }
    assert set(paths) == {"initial_weights", "book", "experiment_config", "protocol"}
    for name, path in paths.items():
        assert sha(path) == spec["inputs"][name]["sha256"]
    assert (
        sha(paths["initial_weights"])
        == "e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03"
    )
    a.weights, a.book, a.config, a.protocol = (
        paths[name]
        for name in ("initial_weights", "book", "experiment_config", "protocol")
    )
    import torch

    import harbichess.training.torch_search_acting_checkpoint as cm
    import harbichess.training.torch_search_acting_learner as lm
    from harbichess.backends.torch_network import load_weights
    from harbichess.selfplay.online_actor import OnlineActorConfig
    from harbichess.training.ownsearch_targets import OwnSearchConfig
    from harbichess.training.torch_fullgame_ppo import FullGamePPOTrainConfig
    from harbichess.training.torch_ownsearch_core import OwnSearchObjective
    from harbichess.training.torch_search_acting_learner import (
        TorchSearchActingConfig,
        TorchSearchActingLearner,
        tensor_bits_equal,
    )
    from harbichess.training.torch_search_acting_run import clean_source

    clean_source(a.source_commit)
    assert Path(lm.__file__).resolve().is_relative_to(a.checkout.resolve() / "src")
    cfg = json.loads(a.config.read_text())
    assert cfg == spec["frozen_config"] and cfg["seed"] in (20261625, 20261626)
    for name, cls in (
        ("actors", OnlineActorConfig),
        ("objective", OwnSearchObjective),
        ("search", OwnSearchConfig),
        ("schedule", FullGamePPOTrainConfig),
    ):
        cfg[name] = cls(**cfg[name])
    config = TorchSearchActingConfig(**cfg)
    assert config.device == "cuda:0" and torch.cuda.is_available()
    assert config.actors.games == 128 and config.epoch_steps == 256
    # Parse and validate original immutable pool once. Official loader remains unmodified
    # except its exact same reader result is memoized; no native validation is skipped.
    pool = lm.read_online_train_book(a.book)

    def cached_book(path):
        assert (
            Path(path).resolve() == a.book.resolve()
            and sha(path) == spec["inputs"]["book"]["sha256"]
        )
        return pool

    lm.read_online_train_book = cm.read_online_train_book = cached_book

    def ready(path):
        while not path.is_file():
            guard(a.deadline_epoch, a.output.parent)
            time.sleep(1)
        assert not path.is_symlink()
        guard(a.deadline_epoch, a.output.parent)

    cp = a.run / "checkpoints"
    ready(cp / "epoch-00000000/checkpoint.json")
    meta = json.loads((a.run / "metadata.json").read_text())
    assert (
        meta["schema"] == "search-acting-supervised-run-v2"
        and meta["source_commit"] == a.source_commit
    )
    assert meta["absolute_deadline_epoch"] == spec["original_training_deadline_epoch"]
    assert meta["max_epochs"] == epochs and meta["checkpoint_interval"] == 1
    assert meta["config"] == spec["frozen_config"]
    assert set(meta["inputs"]) == set(paths)
    for name, path in paths.items():
        assert (a.run / meta["inputs"][name]["relative_path"]).resolve() == path
        assert meta["inputs"][name]["sha256"] == spec["inputs"][name]["sha256"]
    initial = TorchSearchActingLearner.resume(
        cp / "epoch-00000000",
        config=config,
        input_paths=paths,
        source_commit=a.source_commit,
    )
    assert initial.epoch == 0 and initial.closed
    original = load_weights(a.weights).eval()
    for name, value in original.state_dict().items():
        for model in (initial.online, initial.base, initial.behavior):
            assert tensor_bits_equal(value.cpu(), model.state_dict()[name].cpu())
    del original
    receipts = []
    receipt0 = a.output.parent / "epoch-00000000-receipt.json"
    publish(
        receipt0,
        {
            "epoch": 0,
            "source_commit": a.source_commit,
            "seed": config.seed,
            "native_artifact_sha256": {
                str(path.relative_to(a.run)): sha(path)
                for path in (cp / "epoch-00000000").iterdir()
                if path.is_file()
            },
            "optimizer_updates_performed_by_audit": 0,
        },
    )
    receipts.append(receipt0)
    reports = []
    for index in range(1, epochs + 1):
        ready(cp / f"epoch-{index:08d}/checkpoint.json")
        ready(a.run / "journal" / f"epoch-{index:08d}.json.gz")
        initial, report = checked_audit_epoch(
            a, config, paths, initial, index, index in selected_epochs
        )
        path = a.output.parent / f"epoch-{index:08d}-receipt.json"
        publish(path, report)
        receipts.append(path)
        reports.append(report)
    assert sorted(path.name for path in cp.glob("epoch-*")) == [
        f"epoch-{index:08d}" for index in range(epochs + 1)
    ]
    assert sorted(
        path.name for path in (a.run / "journal").glob("epoch-*.json.gz")
    ) == [f"epoch-{index:08d}.json.gz" for index in range(1, epochs + 1)]
    for path in receipts:
        receipt = json.loads(path.read_text())
        assert receipt["source_commit"] == a.source_commit
        for relative, digest in receipt["native_artifact_sha256"].items():
            guard(a.deadline_epoch, a.output.parent)
            assert sha(a.run / relative) == digest
    for name, path in paths.items():
        assert sha(path) == spec["inputs"][name]["sha256"]
    check_source(a.checkout, a.source_commit)
    assert sha(a.manifest) == a.manifest_sha256
    for name, digest in spec["helper_sha256"].items():
        assert sha(Path(__file__).with_name(name)) == digest
    guard(a.deadline_epoch, a.output.parent)
    publish(
        a.output,
        {
            "schema": "ufuk-search-acting-allnative-data-audit-v2",
            "status": "pass-all-fixed-search-acting-v2-native-own-data-search-ledgers",
            "source_commit": a.source_commit,
            "seed": config.seed,
            "fixed_epochs": epochs,
            "audited_native_checkpoints": epochs + 1,
            "independently_replayed_fresh_transitions": 32768 * epochs,
            "schedule_and_all128search_rngs_verified": True,
            "actual_mixed_mu_replayed": True,
            "raw_policy_KL_reference": "KL(frozen-raw-pi||current-raw-pi)",
            "neural_witness_K": 8,
            "prescribed_neural_roots_verified": sum(
                row["prescribed_neural_roots_verified"] for row in reports
            ),
            "raw_actor_packet_roots_verified": sum(
                row["raw_actor_packet_roots_verified"] for row in reports
            ),
            "neural_search_roots_recomputed": sum(
                row["exact_fullgroup_neural_roots_recomputed"] for row in reports
            ),
            "duplicate_neural_positions": sum(
                row["duplicate_neural_positions"] for row in reports
            ),
            "neural_audit_epochs": selected_epochs,
            "sample_chain_sha256": initial.sample_chain_sha256,
            "optimizer_updates_performed_by_this_audit": 0,
            "manifest_sha256": sha(a.manifest),
            "audit_sha256": sha(Path(__file__)),
            "core_sha256": sha(Path(__file__).with_name("own6_audit_core.py")),
            "common_sha256": sha(Path(__file__).with_name("own6_audit_support.py")),
            "controls_sha256": sha(
                Path(__file__).with_name("own6_adapter_controls.py")
            ),
            "immutable_percheckpoint_audit_receipt_sha256": {
                str(path): sha(path) for path in receipts
            },
            "reports": reports,
            "original_first_clock": first,
            "deadline_epoch": a.deadline_epoch,
            "finished_epoch": time.time(),
            "sampling_scope": (
                "policy root-uniform includingUNKNOWN; terminalvalue game-uniform "
                "thenposition-uniform; "
                "rawNN-referenceKLgame-balanced; chronological128 roots+selected masks replay "
                "with108designated roots "
                "and reportedextras; no optimizer/strength claim"
            ),
        },
    )


if __name__ == "__main__":
    main()
