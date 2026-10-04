"""Prospective explicit invocation only: strict all41 native and own-data audit.

Run incrementally alongside training, reading only atomic immutable epoch
publications; device/runtime remain original. Does no optimizer updates.
Manifest supplies frozen source, seed, four inputs and config. Official
checkpoint loader is used alongside independent actor sampling/terminal replay.
"""

import argparse
import gzip
import hashlib
import json
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

import harbichess.training.torch_fullgame_checkpoint as cm
import harbichess.training.torch_fullgame_learner as lm
from harbichess.backends.torch_network import sha256
from harbichess.chess.actions import legal_action_indices
from harbichess.selfplay.online_actor import OnlineActorConfig, OnlineActors
from harbichess.selfplay.online_epoch import deserialize_policy_epoch
from harbichess.training.fullgame_own_targets import build_fullgame_targets
from harbichess.training.torch_fullgame_learner import (
    TorchFullGameConfig,
    TorchFullGameLearner,
    canonical,
)
from harbichess.training.torch_fullgame_ppo import (
    FullGamePPOConfig,
    FullGamePPOTrainConfig,
    make_torch_epoch_inference,
    torch_model_digest,
)
from harbichess.training.torch_fullgame_run import clean_source

NEURAL_AUDIT_EPOCHS = (1, 2, 10, 20, 30, 40)
NEURAL_TOLERANCE = 2e-5


def select_neural_rows(actions, rules):
    # State-only prospective selection: nine distinct full histories per mover,
    # spread evenly over sorted history length; no labels/logits/outcomes used.
    unique = {}
    for row in actions:
        unique.setdefault(row.transition.pre, row)
    selected = []
    for side in ("white", "black"):
        rows = [
            row for row in unique.values() if rules.view(row.transition.pre).side_to_move == side
        ]
        rows.sort(
            key=lambda row: (
                row.transition.pre.ply,
                row.transition.pre.root_fen,
                tuple(m.uci for m in row.transition.pre.moves),
                row.transition.source_id,
                row.transition.game_index,
            )
        )
        assert len(rows) >= 9, "Neural audit needs nine distinct histories per mover"
        selected.extend(rows[(index * (len(rows) - 1)) // 8] for index in range(9))
    return selected


def check_neural_rows(native, rows, guard):
    assert len(rows) == 18
    guard()
    infer = make_torch_epoch_inference(
        native.behavior, native.base, native.encoder, device=native.config.device
    )
    result = infer(
        tuple(row.transition.pre for row in rows),
        tuple(row.legal_actions for row in rows),
    )
    errors = {}
    fields = (
        ("policy", "policy"),
        ("policy", "behavior_policy"),
        ("wdl", "online_pre_wdl"),
        ("base_policy", "base_policy"),
        ("base_wdl", "base_wdl"),
    )
    for output_field, receipt_field in fields:
        maximum = 0.0
        for computed, row in zip(getattr(result, output_field), rows, strict=True):
            expected = getattr(row, receipt_field)
            difference = float(np.max(np.abs(np.asarray(computed) - np.asarray(expected))))
            maximum = max(maximum, difference)
            assert np.allclose(computed, expected, atol=NEURAL_TOLERANCE, rtol=NEURAL_TOLERANCE), (
                f"Native neural receipt mismatch {receipt_field}: {difference}"
            )
        errors[receipt_field] = maximum
    guard()
    return {
        "samples": 18,
        "selection": (
            "nine distinct full histories per mover spread over history length"
            "; fixed state-only order"
        ),
        "max_absolute_errors": errors,
        "atol": NEURAL_TOLERANCE,
        "rtol": NEURAL_TOLERANCE,
        "device": native.config.device,
        "behavior_digest": torch_model_digest(native.behavior),
        "base_digest": torch_model_digest(native.base),
        "histories": [
            {
                "source_id": r.transition.source_id,
                "game_index": r.transition.game_index,
                "slot": r.transition.slot,
                "root_fen": r.transition.pre.root_fen,
                "moves": [m.uci for m in r.transition.pre.moves],
                "legal_actions": list(r.legal_actions),
            }
            for r in rows
        ],
    }


def bits(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.equal(
        a.detach().cpu().reshape(-1).view(torch.uint8),
        b.detach().cpu().reshape(-1).view(torch.uint8),
    )


def unused_storage_witness(base, online, behavior):
    """Accept zero-valued frozen weights; assert storage preservation, not nonzero count."""
    unused = {
        name: value for name, value in base.items() if name.startswith("material_value_linear.")
    }
    assert set(unused) == {"material_value_linear.weight", "material_value_linear.bias"}
    assert sum(value.numel() for value in unused.values()) == 21
    result = {}
    for name, value in unused.items():
        bits(value, online[name])
        bits(value, behavior[name])
        storage = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        result[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "elements": value.numel(),
            "nonzero_elements": int(torch.count_nonzero(value).item()),
            "zero_elements": int((value == 0).sum().item()),
            "negative_zero_elements": int(((value == 0) & torch.signbit(value)).sum().item()),
            "storage_sha256": hashlib.sha256(storage.numpy().tobytes()).hexdigest(),
        }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--deadline-epoch", type=float, required=True)
    a = p.parse_args()
    started = time.time()
    assert 0 < a.deadline_epoch - started <= 21000

    def guard():
        if time.time() >= a.deadline_epoch:
            raise TimeoutError("Original independent audit wholedeadline exhausted")
        memory = Path("/sys/fs/cgroup/memory.current")
        if memory.exists() and int(memory.read_text()) > 64 * 1024**3:
            raise RuntimeError("registered64GiB globalmemoryceiling")
        if shutil.disk_usage(a.output.parent).free < 8 * 1024**3:
            raise RuntimeError("registered8GiB diskfloor")

    def wait_ready(path):
        while not path.exists():
            guard()
            time.sleep(1)
        assert not path.is_symlink()
        guard()

    spec = json.loads(a.manifest.read_text())
    source = spec["source_commit"]
    clean_source(source)
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == source
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    paths = {name: Path(row["path"]).resolve() for name, row in spec["inputs"].items()}
    for name, path in paths.items():
        assert sha256(path) == spec["inputs"][name]["sha256"]
    assert (
        sha256(paths["initial_weights"])
        == "e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03"
    )
    cfg = json.loads(paths["experiment_config"].read_text())
    cfg["actors"] = OnlineActorConfig(**cfg["actors"])
    cfg["objective"] = FullGamePPOConfig(**cfg["objective"])
    cfg["schedule"] = FullGamePPOTrainConfig(**cfg["schedule"])
    config = TorchFullGameConfig(**cfg)
    assert (
        config.seed in (20261205, 20261206)
        and config.actors.games == 128
        and config.epoch_steps == 256
    )
    expected = spec["frozen_config"]
    assert json.loads(paths["experiment_config"].read_text()) == expected
    wait_ready(a.run / "metadata.json")
    # _publish_once writes metadata in-place exclusively (fsync, not rename).
    # Native0 is published by atomic directory rename only AFTER metadata has
    # fully returned, so native0 availability is the safe metadata barrier.
    wait_ready(a.run / "checkpoints/epoch-00000000/checkpoint.json")
    meta = json.loads((a.run / "metadata.json").read_text())
    assert (
        meta["source_commit"] == source
        and meta["max_epochs"] == 40
        and meta["checkpoint_interval"] == 1
    )
    assert meta["absolute_deadline_epoch"] == spec["original_training_deadline_epoch"]
    assert a.deadline_epoch == min(meta["absolute_deadline_epoch"] + 3000, 1791170400.0)
    expectednative = [f"epoch-{i:08d}" for i in range(41)]
    # Prefix grows atomically during training; requireexactfinalinventory at closure.
    # Validate 4096-root train pool once. Cached returns preserve official actors
    # restore, exact book hash and source/input pins; no state validation skipped.
    pool = lm.read_online_train_book(paths["book"])

    def cached(path):
        assert Path(path).resolve() == paths["book"]
        return pool

    lm.read_online_train_book = cm.read_online_train_book = cached
    kw = dict(config=config, input_paths=paths, source_commit=source)
    initial = TorchFullGameLearner.resume(a.run / "checkpoints/epoch-00000000", **kw)
    initial_manifest_sha256 = sha256(a.run / "checkpoints/epoch-00000000/checkpoint.json")
    unused_witness = unused_storage_witness(
        initial.base.state_dict(),
        initial.online.state_dict(),
        initial.behavior.state_dict(),
    )
    checkpoint0_receipt = {
        "unused_material_initial_storage": unused_witness,
        "schema": "ufuk-method2-incremental-native0-audit-v1",
        "epoch": 0,
        "native_manifest_sha256": initial_manifest_sha256,
        "native_artifact_sha256": json.loads(
            (a.run / "checkpoints/epoch-00000000/checkpoint.json").read_text()
        )["artifacts"],
        "source_commit": source,
        "seed": config.seed,
        "fresh_transitions": 0,
        "optimizer_accepted_updates": 0,
        "actor_cursor_sha256": hashlib.sha256(canonical(initial.actors.cursor())).hexdigest(),
        "native_global_Adam_RNG_strictload": True,
        "audit_optimizer_updates": 0,
    }
    with (a.output.parent / f"{a.output.name}.epoch-00000000.json").open("x") as f:
        json.dump(checkpoint0_receipt, f, indent=2)
        f.write("\n")
    base = {name: value.detach().cpu().clone() for name, value in initial.base.state_dict().items()}
    previous_online = {
        name: value.detach().cpu().clone() for name, value in initial.online.state_dict().items()
    }
    unused = {
        name: value for name, value in base.items() if name.startswith("material_value_linear.")
    }
    assert sum(x.numel() for x in unused.values()) == 21
    # Actual e8 contains one +0.0 weight. All 21 raw-storage values remain
    # frozen and are checked above at native0 and below at every epoch.
    actor_rng = random.Random()
    actor_rng.setstate(initial.actors.rng.getstate())
    actors = OnlineActors(pool, config=config.actors, rng=actor_rng, cursor=initial.actors.cursor())
    seed_rng = random.Random()
    seed_rng.setstate(initial.sampler_seed_rng.getstate())
    chain = hashlib.sha256(b"").hexdigest()
    attempted = accepted = 0
    counts = {}
    receipts = []
    neural_receipts = []
    for index in range(1, 41):
        wait_ready(a.run / "journal" / f"epoch-{index:08d}.json.gz")
        wait_ready(a.run / "checkpoints" / f"epoch-{index:08d}" / "checkpoint.json")
        guard()
        data = (a.run / "journal" / f"epoch-{index:08d}.json.gz").read_bytes()
        record = json.loads(gzip.decompress(data))
        claimed = record.pop("sample_chain_sha256")
        assert record["previous_sample_chain_sha256"] == chain and record["epoch"] == index
        chain = hashlib.sha256(bytes.fromhex(chain) + canonical(record)).hexdigest()
        assert chain == claimed
        epoch = deserialize_policy_epoch(canonical(record["collection"]))
        assert (
            epoch.requested_steps == 256
            and epoch.start_actor_step == (index - 1) * 256
            and epoch.end_actor_step == index * 256
            and len(epoch.actions) == 32768
        )
        assert record["sampler_seed"] == seed_rng.randrange(2**63)
        for offset in range(0, len(epoch.actions), 128):
            guard()
            rows = epoch.actions[offset : offset + 128]
            assert [x.transition.slot for x in rows] == list(range(128))
            for row in rows:
                assert row.legal_actions == tuple(
                    legal_action_indices(actors.rules.inspect(row.transition.pre))
                )
            actual = actors.step(tuple(x.policy for x in rows))
            assert actual == tuple(x.transition for x in rows), (
                "Independent full-history actor/RNG transition differs"
            )
        assert actors.close_policy_epoch() == epoch.truncations
        assert (
            actors.cursor() == epoch.next_actor_cursor
            and actors.rng.getstate() == epoch.next_actor_rng_state
        )
        labels = build_fullgame_targets(actors.rules, epoch, claim_draw=config.actors.claim_draw)
        for name, value in record["target_counts"].items():
            assert getattr(labels, name) == value
            if type(value) is int:
                counts[name] = counts.get(name, 0) + value
        assert len(labels.targets) == record["training"]["trained_transitions"]
        # All completed games independently propagated; rule legality/terminal
        # continuity of UNKNOWN episodes is also checked by target builder.
        attempted += record["training"]["optimizer_steps_attempted"]
        accepted += record["training"]["optimizer_steps_committed"]
        native = TorchFullGameLearner.resume(a.run / "checkpoints" / f"epoch-{index:08d}", **kw)
        assert (
            native.epoch == index
            and native.sample_chain_sha256 == chain
            and native.last_epoch_gzip == data
        )
        assert (
            native.actors.cursor() == actors.cursor()
            and native.actors.rng.getstate() == actors.rng.getstate()
        )
        assert native.sampler_seed_rng.getstate() == seed_rng.getstate()
        assert (
            native.optimizer_attempted_updates == attempted
            and native.optimizer_accepted_updates == accepted
            and native.optimizer_rejected_updates == attempted - accepted
        )
        assert epoch.model_digest == torch_model_digest(native.behavior)
        if index in NEURAL_AUDIT_EPOCHS:
            sample_receipt = check_neural_rows(
                native, select_neural_rows(epoch.actions, actors.rules), guard
            )
            sample_receipt["epoch"] = index
            neural_receipts.append(sample_receipt)
        for name, value in previous_online.items():
            bits(value, native.behavior.state_dict()[name])
        for name, value in base.items():
            bits(value, native.base.state_dict()[name])
        for name, value in unused.items():
            bits(value, native.online.state_dict()[name])
            bits(value, native.behavior.state_dict()[name])
        receipts.append(
            {
                "epoch": index,
                "checkpoint_manifest_sha256": sha256(
                    a.run / "checkpoints" / f"epoch-{index:08d}" / "checkpoint.json"
                ),
                "journal_sha256": sha256(a.run / "journal" / f"epoch-{index:08d}.json.gz"),
                "native_artifact_sha256": json.loads(
                    (a.run / "checkpoints" / f"epoch-{index:08d}" / "checkpoint.json").read_text()
                )["artifacts"],
                "complete_games": labels.complete_games,
                "trained_transitions": len(labels.targets),
            }
        )
        previous_online = {
            name: value.detach().cpu().clone() for name, value in native.online.state_dict().items()
        }
        del native
        guard()
        progress = {
            "schema": "ufuk-method2-incremental-audit-progress-v1",
            "audited_through_epoch": index,
            "audited_fresh_transitions": index * 32768,
            "source_commit": source,
            "seed": config.seed,
            "sample_chain_sha256": chain,
            "latest_checkpoint": receipts[-1],
            "actor_steps": index * 256,
            "optimizer_attempted": attempted,
            "optimizer_accepted": accepted,
            "optimizer_rejected": attempted - accepted,
            "audit_optimizer_updates": 0,
            "neural_receipt_validation": neural_receipts,
            "original_audit_deadline_epoch": a.deadline_epoch,
            "finished_epoch": time.time(),
        }
        with (a.output.parent / f"{a.output.name}.epoch-{index:08d}.json").open("x") as f:
            json.dump(progress, f, indent=2)
            f.write("\n")
    assert (
        sorted(
            x.name
            for x in (a.run / "checkpoints").iterdir()
            if x.is_dir() and x.name.startswith("epoch-")
        )
        == expectednative
    )
    assert sorted(x.name for x in (a.run / "journal").glob("epoch-*.json.gz")) == [
        f"epoch-{i:08d}.json.gz" for i in range(1, 41)
    ]
    assert accepted > 0
    receipt_files = [
        a.output.parent / f"{a.output.name}.epoch-{index:08d}.json" for index in range(41)
    ]
    assert all(path.is_file() and not path.is_symlink() for path in receipt_files)
    initial_check = json.loads(receipt_files[0].read_text())
    assert initial_check["epoch"] == 0 and initial_check["native_manifest_sha256"] == sha256(
        a.run / "checkpoints/epoch-00000000/checkpoint.json"
    )
    assert (
        initial_check["source_commit"] == source
        and initial_check["seed"] == config.seed
        and initial_check["audit_optimizer_updates"] == 0
        and initial_check["fresh_transitions"] == 0
    )
    for name, digest in initial_check["native_artifact_sha256"].items():
        guard()
        assert sha256(a.run / "checkpoints/epoch-00000000" / name) == digest
    for index, path in enumerate(receipt_files[1:], 1):
        proof = json.loads(path.read_text())
        assert (
            proof["audited_through_epoch"] == index
            and proof["audited_fresh_transitions"] == index * 32768
            and proof["audit_optimizer_updates"] == 0
        )
        assert (
            proof["source_commit"] == source
            and proof["seed"] == config.seed
            and proof["actor_steps"] == index * 256
        )
        native_state = json.loads(
            (a.run / "checkpoints" / f"epoch-{index:08d}" / "checkpoint.json").read_text()
        )["state"]
        assert (
            proof["sample_chain_sha256"] == native_state["sample_chain_sha256"]
            and proof["optimizer_attempted"] == native_state["optimizer_attempted_updates"]
            and proof["optimizer_accepted"] == native_state["optimizer_accepted_updates"]
            and proof["optimizer_rejected"] == native_state["optimizer_rejected_updates"]
        )
        assert proof["latest_checkpoint"]["checkpoint_manifest_sha256"] == sha256(
            a.run / "checkpoints" / f"epoch-{index:08d}" / "checkpoint.json"
        )
        assert proof["latest_checkpoint"]["journal_sha256"] == sha256(
            a.run / "journal" / f"epoch-{index:08d}.json.gz"
        )
        for name, digest in proof["latest_checkpoint"]["native_artifact_sha256"].items():
            guard()
            assert sha256(a.run / "checkpoints" / f"epoch-{index:08d}" / name) == digest
    for name, path in paths.items():
        assert sha256(path) == spec["inputs"][name]["sha256"]
    result = {
        "schema": "ufuk-method2-fullnative-terminal-audit-v1",
        "status": "pass-all41native-all40epochs-independent-own-data",
        "source_commit": source,
        "seed": config.seed,
        "manifest_sha256": sha256(a.manifest),
        "audit_sha256": sha256(Path(__file__)),
        "audited_native_checkpoints": 41,
        "immutable_percheckpoint_audit_receipt_sha256": {
            str(path): sha256(path) for path in receipt_files
        },
        "independent_native0_manifest_sha256": initial_manifest_sha256,
        "audit_schedule": (
            "incrementalimmutable-per-epoch whiletraining; exactsame allnative"
            "/allactor/allterminal and selectedneuralchecks; originalshared210"
            "00maximum with3000tail, no per-epoch reset"
        ),
        "independently_replayed_fresh_transitions": 1310720,
        "terminal_data_counts": counts,
        "optimizer_attempts": attempted,
        "optimizer_accepted": accepted,
        "sample_chain_sha256": chain,
        "checkpoints": receipts,
        "deadline_epoch": a.deadline_epoch,
        "started_epoch": started,
        "finished_epoch": time.time(),
        "optimizer_updates_performed_by_this_audit": 0,
        "neural_receipt_validation": neural_receipts,
        "neural_audit_epochs": list(NEURAL_AUDIT_EPOCHS),
        "neural_receipt_scope": (
            "Only selected108history rows perseed independently evaluated on a"
            "ctual device; other rows neural values not recomputed. All actor/"
            "legal/terminal histories are replayed. Final CPU-CUDA and TorchCP"
            "U-MLXCPU18full/masked sameweights parity remain separate mandator"
            "y gates."
        ),
        "limits": (
            "Official native validates Adam/all global RNG containers; indepen"
            "dent epoch1->2 audit must additionally regenerate next epoch and "
            "compare every native payload bit. This full audit itself does not"
            " prove optimizer trajectory or strength."
        ),
    }
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "checkpoints"}))


if __name__ == "__main__":
    main()
