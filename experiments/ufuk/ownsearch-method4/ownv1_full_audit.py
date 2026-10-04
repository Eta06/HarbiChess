"""Prospective incremental OWN-v1 all-native/raw-data/schedule audit; no optimizer."""

import argparse
import gzip
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from ownv1_audit_support import check_source, guard, publish, sha


def check_chain(record, parent, expected):
    value = dict(record)
    digest = value.pop("sample_chain_sha256")
    assert value["previous_sample_chain_sha256"] == parent
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert hashlib.sha256(bytes.fromhex(parent) + payload).hexdigest() == digest == expected


def prescribe_roots(epoch, groups, rules):
    candidates = list(dict.fromkeys([*groups[0], *groups[-1]]))
    by_mover = {True: [], False: []}
    seen = set()
    for index in candidates:
        row = epoch.actions[index]
        if row.transition.pre in seen:
            continue
        seen.add(row.transition.pre)
        by_mover[rules.view(row.transition.pre).side_to_move.value == "white"].append(index)
    assert all(len(rows) >= 9 for rows in by_mover.values()), (
        "Insufficient prescribed mover coverage; no adaptive alternate groups"
    )
    selected = []
    for rows in by_mover.values():
        rows.sort(key=lambda index: (epoch.actions[index].transition.pre.ply, index))
        selected += [rows[round(i * (len(rows) - 1) / 8)] for i in range(9)]
    assert len(set(selected)) == 18
    return sorted(selected)


def compare_terminal_packet(actual, expected):
    assert actual == expected


def compare_search_packet(actual, expected, canonical):
    assert canonical(actual) == canonical(expected)


def compare_frozen_storage(actual, expected, bits_equal):
    assert bits_equal(actual, expected)


def actor_packet_groups(prescribed, count, actors=128):
    assert count % actors == 0 and len(set(prescribed)) == len(prescribed)
    assert all(type(index) is int and 0 <= index < count for index in prescribed)
    return [
        list(range(start, start + actors))
        for start in sorted({index // actors * actors for index in prescribed})
    ]


def compare_actor_packet(row, policy, base_policy, wdl, base_wdl):
    def normalized(values):
        assert all(math.isfinite(x) and x >= 0 for x in values)
        total = math.fsum(values)
        assert math.isclose(total, 1.0, abs_tol=1e-6)
        return tuple(x / total for x in values)

    policy, base_policy = normalized(policy), normalized(base_policy)
    assert policy == row.policy and base_policy == row.base_policy
    assert normalized(wdl) == row.online_pre_wdl and normalized(base_wdl) == row.base_wdl
    logs = [math.log(v) if v > 0 else -math.inf for v in policy]
    scaled = [math.exp(v - max(logs)) for v in logs]
    assert tuple(v / math.fsum(scaled) for v in scaled) == row.behavior_policy


def audit_epoch(a, config, paths, initial, index, neural):
    import torch

    from harbichess.chess.actions import legal_action_indices
    from harbichess.selfplay.online_epoch import _tuplify, deserialize_policy_epoch
    from harbichess.training.fullgame_own_targets import (
        GameBalancedSampler,
        build_fullgame_targets,
    )
    from harbichess.training.ownsearch_targets import (
        make_search_ledger,
    )
    from harbichess.training.torch_ownsearch_learner import (
        TorchOwnSearchLearner,
        canonical,
        tensor_bits_equal,
    )
    from harbichess.training.torch_ownsearch_run import clean_source

    cp = a.run / "checkpoints"
    artifacts = [f for n in (index - 1, index) for f in (cp / f"epoch-{n:08d}").iterdir()]
    journal = a.run / "journal" / f"epoch-{index:08d}.json.gz"
    artifacts.append(journal)
    before = {str(f.relative_to(a.run)): sha(f) for f in artifacts}
    before_inputs = {k: sha(v) for k, v in paths.items()}
    initial_models = {
        name: {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        for name, model in [("online", initial.online), ("base", initial.base)]
    }
    schedule_before = initial.schedule_rng.getstate()
    search_before = [r.getstate() for r in initial.search_rngs]
    sampler_seed = initial.sampler_seed_rng.randrange(2**63)
    sampler_after = initial.sampler_seed_rng.getstate()
    final = TorchOwnSearchLearner.resume(
        cp / f"epoch-{index:08d}",
        config=config,
        input_paths=paths,
        source_commit=a.source_commit,
    )
    guard(a.deadline_epoch, a.run)
    assert final.config.device == "cuda:0" and final.closed and final.epoch == index
    assert initial.epoch == index - 1 and initial.closed
    assert final.last_epoch_gzip == journal.read_bytes()
    record = json.loads(gzip.decompress(journal.read_bytes()))
    epoch = deserialize_policy_epoch(canonical(record["collection"]))
    assert record["schema"] == "torch-fresh-sparse-ownsearch-v1" and record["epoch"] == index
    assert len(epoch.actions) == record["fresh_transitions"] == 32768
    assert final.actors.steps == record["actor_steps"] == 256 * index
    assert record["total_fresh_transitions"] == 32768 * index
    assert (
        record["sampler_seed"] == sampler_seed
        and final.sampler_seed_rng.getstate() == sampler_after
    )
    ledger = record["own_search"]
    assert _tuplify(ledger["schedule_rng_before"]) == schedule_before
    assert [_tuplify(r) for r in ledger["search_rng_before"]] == search_before
    schedule_rng = random.Random()
    schedule_rng.setstate(schedule_before)
    schedule_pending = {}
    seen_offsets, expected_selected = [], []
    # Independently consume raw behavior probabilities through the original actor RNG.
    for offset in range(0, len(epoch.actions), config.actors.games):
        guard(a.deadline_epoch, a.run)
        rows = epoch.actions[offset : offset + config.actors.games]
        assert [row.transition.slot for row in rows] == list(range(128))
        active = set()
        for row in rows:
            tr = row.transition
            opening = initial.actors.openings[initial.actors.games[tr.slot].opening_index]
            block, block_offset = divmod(tr.pre.ply - opening.state.ply, config.search.block_plies)
            key = (tr.source_id, tr.game_index, tr.slot, block)
            active.add(key)
            if key not in schedule_pending:
                schedule_pending[key] = schedule_rng.randrange(config.search.block_plies)
                seen_offsets.append(
                    dict(
                        source_id=key[0],
                        game_index=key[1],
                        slot=key[2],
                        block=block,
                        chosen_offset=schedule_pending[key],
                        first_collection_index=offset + tr.slot,
                    )
                )
            if block_offset == schedule_pending[key]:
                expected_selected.append(offset + tr.slot)
            logs = [math.log(v) if v > 0 else -math.inf for v in row.policy]
            scaled = [math.exp(v - max(logs)) for v in logs]
            mu = tuple(v / math.fsum(scaled) for v in scaled)
            assert mu == row.behavior_policy
            assert (
                tuple(legal_action_indices(initial.actors.rules.inspect(row.transition.pre)))
                == row.legal_actions
            )
        schedule_pending = {key: value for key, value in schedule_pending.items() if key in active}
        actual = initial.actors.step(tuple(row.policy for row in rows))
        assert actual == tuple(row.transition for row in rows)
    assert ledger["selected_indices"] == expected_selected and ledger["offsets"] == seen_offsets
    assert schedule_rng.getstate() == _tuplify(ledger["schedule_rng_after"])
    assert len(ledger["roots"]) == len(expected_selected)
    from harbichess.training.ownsearch_targets import validate_search_ledger

    validate_search_ledger(epoch, ledger, config.search, rules=final.actors.rules)
    assert canonical(initial.actors.close_policy_epoch()) == canonical(epoch.truncations)
    assert initial.actors.cursor() == final.actors.cursor() == epoch.next_actor_cursor
    assert (
        initial.actors.rng.getstate() == final.actors.rng.getstate() == epoch.next_actor_rng_state
    )
    # Terminal labels independently derived from final exact-rules outcomes and each mover.
    groups = defaultdict(list)
    for row in epoch.actions:
        groups[(row.transition.source_id, row.transition.game_index)].append(row)
    expected = {}
    unknown = 0
    for rows in groups.values():
        guard(a.deadline_epoch, a.run)
        outcome = final.actors.rules.outcome(
            rows[-1].transition.post, claim_draw=config.actors.claim_draw
        )
        if outcome is None:
            unknown += len(rows)
            continue
        for row in rows:
            tr = row.transition
            value = outcome.value_for(final.actors.rules.view(tr.pre).side_to_move)
            expected[tr] = tuple(float(value == v) for v in (1, 0, -1))
    targets = build_fullgame_targets(final.actors.rules, epoch, claim_draw=config.actors.claim_draw)
    assert len(targets.targets) == len(expected) == record["training"]["trained_transitions"]
    assert targets.excluded_actions == unknown == record["target_counts"]["excluded_actions"]
    for target in targets.targets:
        compare_terminal_packet(target.target_wdl, expected[target.transition])
    from dataclasses import fields

    assert record["target_counts"] == {
        field.name: getattr(targets, field.name)
        for field in fields(targets)
        if field.name != "targets"
    }
    known_groups = defaultdict(list)
    for target in targets.targets:
        known_groups[(target.source_id, target.game_index)].append(target)
    inverse_game_weights = {
        str(key): 1 / (len(known_groups) * len(rows)) for key, rows in known_groups.items()
    }
    assert (
        not known_groups
        or abs(
            sum(inverse_game_weights[str(key)] * len(rows) for key, rows in known_groups.items())
            - 1
        )
        < 1e-12
    )
    # Reproduce accepted sampler draws; rejected passes restore both pass-start RNGs.
    training = record["training"]
    assert (
        final.optimizer_attempted_updates
        == initial.optimizer_attempted_updates + training["optimizer_steps_attempted"]
    )
    assert (
        final.optimizer_accepted_updates
        == initial.optimizer_accepted_updates + training["optimizer_steps_committed"]
    )
    assert (
        final.optimizer_rejected_updates
        == initial.optimizer_rejected_updates + training["optimizer_steps_discarded"]
    )
    assert training["accepted_passes"] + training["rejected_passes"] <= config.schedule.passes
    assert training["rejected_passes"] in (0, 1)
    policy_rng = random.Random(sampler_seed)
    value_sampler = (
        GameBalancedSampler(targets.targets, seed=sampler_seed ^ 0xBA71)
        if targets.targets
        else None
    )
    sampler_initial = dict(
        policy=policy_rng.getstate(),
        value=value_sampler.rng.getstate() if value_sampler else None,
    )
    assert canonical(sampler_initial) == canonical(training["sampler_rng_state_before_epoch"])
    assert (
        training["optimizer_steps_committed"]
        == training["accepted_passes"] * training["minibatches_per_pass"]
    )
    assert (
        training["optimizer_steps_attempted"]
        == (training["accepted_passes"] + training["rejected_passes"])
        * training["minibatches_per_pass"]
    )
    for _ in range(training["optimizer_steps_committed"]):
        guard(a.deadline_epoch, a.run)
        policy_rng.choices(ledger["roots"], k=config.schedule.minibatch_size)
        if value_sampler:
            value_sampler.sample(config.schedule.minibatch_size)
    sampler_final = dict(
        policy=policy_rng.getstate(),
        value=value_sampler.rng.getstate() if value_sampler else None,
    )
    assert canonical(sampler_final) == canonical(final.last_sampler_rng_state)
    # Every base/behavior tensor and all unused material storage bits, including signed zero.
    material_elements = material_nonzero = 0
    material_sha = {}
    import hashlib

    for name, value in final.base.state_dict().items():
        compare_frozen_storage(value.cpu(), initial_models["base"][name], tensor_bits_equal)
        assert tensor_bits_equal(
            final.behavior.state_dict()[name].cpu(), initial_models["online"][name]
        )
        if name.startswith("material_value_linear."):
            for model in (final.online, final.base, final.behavior):
                assert tensor_bits_equal(
                    model.state_dict()[name].cpu(), initial_models["base"][name]
                )
            material_elements += value.numel()
            material_nonzero += int(torch.count_nonzero(value))
            material_sha[name] = hashlib.sha256(
                value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            ).hexdigest()
    assert material_elements == 21
    # Recreate the ORIGINAL full wavefront composition, never scalar/subset batch shapes.
    queues = {slot: [] for slot in range(config.actors.games)}
    for i in ledger["selected_indices"]:
        queues[epoch.actions[i].transition.slot].append(i)
    batches = []
    while any(queues.values()):
        batches.append([rows.pop(0) for rows in queues.values() if rows])
    chosen = {0, len(batches) - 1} if neural else set()
    prescribed = prescribe_roots(epoch, batches, final.actors.rules) if neural else []
    # Raw actor packets use the ORIGINAL complete128 actor-step batch, both models.
    # Neither scalar inference nor a batch of only selected rows preserves FP32 kernels.
    raw_groups = actor_packet_groups(prescribed, len(epoch.actions))
    raw_references = {}
    if raw_groups:
        from harbichess.training.torch_fullgame_ppo import make_torch_epoch_inference

        infer = make_torch_epoch_inference(
            final.behavior, final.base, final.encoder, device=config.device
        )
        prescribed_set = set(prescribed)
        for indices in raw_groups:
            guard(a.deadline_epoch, a.run)
            rows = [epoch.actions[i] for i in indices]
            output = infer(
                tuple(row.transition.pre for row in rows), tuple(row.legal_actions for row in rows)
            )
            for position, collection_index in enumerate(indices):
                if collection_index in prescribed_set:
                    raw_references[collection_index] = {
                        "policy": output.policy[position],
                        "base_policy": output.base_policy[position],
                        "wdl": output.wdl[position],
                        "base_wdl": output.base_wdl[position],
                    }
                    compare_actor_packet(
                        rows[position],
                        output.policy[position],
                        output.base_policy[position],
                        output.wdl[position],
                        output.base_wdl[position],
                    )
    raw_duplicate_positions = 2 * 128 * len(raw_groups)
    rngs = [random.Random() for _ in search_before]
    for rng, state in zip(rngs, search_before, strict=True):
        rng.setstate(state)
    root_by_index = {root["collection_index"]: root for root in ledger["roots"]}
    audited = neural_positions = 0
    for index, indices in enumerate(batches):
        guard(a.deadline_epoch, a.run)
        if index not in chosen:
            for i in indices:
                row = epoch.actions[i]
                for _ in row.legal_actions:
                    rngs[row.transition.slot].random()
            continue
        subset = make_search_ledger(
            epoch,
            SimpleNamespace(selected=indices, offsets=[]),
            final.behavior,
            final.actors.rules,
            config.search,
            rngs,
            device=config.device,
            guard=lambda: guard(a.deadline_epoch, a.run),
            claim_draw=config.actors.claim_draw,
        )
        for root in subset["roots"]:
            compare_search_packet(root, root_by_index[root["collection_index"]], canonical)
        audited += len(indices)
        neural_positions += subset["neural_positions"]
    assert [r.getstate() for r in rngs] == [_tuplify(r) for r in ledger["search_rng_after"]]
    assert before == {str(f.relative_to(a.run)): sha(f) for f in artifacts}
    assert before_inputs == {k: sha(v) for k, v in paths.items()}
    check_chain(record, initial.sample_chain_sha256, final.sample_chain_sha256)
    assert initial.schedule_rng.getstate() == _tuplify(ledger["schedule_rng_before"])
    assert final.schedule_rng.getstate() == _tuplify(ledger["schedule_rng_after"])
    assert [r.getstate() for r in final.search_rngs] == [
        _tuplify(x) for x in ledger["search_rng_after"]
    ]
    assert record["training"]["policy_target_rows"] == len(ledger["roots"])
    assert canonical(training["sampler_rng_state"]) == canonical(sampler_final)
    guard(a.deadline_epoch, a.run)
    clean_source(a.source_commit)
    report = dict(
        epoch=index,
        native_artifact_sha256=before,
        journal_sha256=sha(journal),
        raw_actor_replayed=32768,
        known_value_rows=len(expected),
        unknown_value_rows=unknown,
        search_target_rows=len(ledger["roots"]),
        optimizer_attempts=training["optimizer_steps_attempted"],
        optimizer_committed=training["optimizer_steps_committed"],
        sample_chain_sha256=final.sample_chain_sha256,
        schedule_and_all128search_rngs_verified=True,
        sampler_policy_root_uniform_value_game_uniform=True,
        prescribed_neural_roots_verified=len(prescribed),
        prescribed_collection_indices=prescribed,
        exact_fullgroup_neural_roots_recomputed=audited,
        duplicate_neural_positions=neural_positions + raw_duplicate_positions,
        raw_actor_packet_roots_verified=len(prescribed),
        raw_actor_packet_recomputed_reference=raw_references,
        raw_actor_full128_groups=len(raw_groups),
        raw_actor_duplicate_neural_positions_both_models=raw_duplicate_positions,
        neural_selection=(
            "18history-only designated roots in FIRST+LAST ORIGINAL FULLwavefrontgroups; "
            "allgroup roots exactcompared"
        ),
        material_elements=material_elements,
        material_nonzero=material_nonzero,
        material_storage_sha256=material_sha,
        source_commit=a.source_commit,
        optimizer_updates_performed_by_audit=0,
    )
    return final, report


def main():
    if not __debug__:
        raise RuntimeError("Required assertions disabled")
    p = argparse.ArgumentParser()
    for name in ("manifest", "run", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--deadline-epoch", type=float, required=True)
    a = p.parse_args()
    spec = json.loads(a.manifest.read_text())
    a.checkout = Path(spec["producer_checkout"])
    a.source_commit = spec["source_commit"]
    epochs = spec["fixed_epochs"]
    assert type(epochs) is int and epochs >= 8
    assert spec["qualification_ledger_slot"] == 4
    first = spec["original_training_started_epoch"]
    assert spec["original_training_deadline_epoch"] == first + spec["whole_training_seconds"]
    assert a.deadline_epoch == min(first + spec["whole_audit_seconds"], 1791170400)
    assert first <= time.time() < a.deadline_epoch
    selected_epochs = spec["neural_audit_epochs"]
    assert len(set(selected_epochs)) == len(selected_epochs) == 6
    assert all(1 <= index <= epochs for index in selected_epochs)
    assert selected_epochs == sorted(selected_epochs) and selected_epochs[:2] == [1, 2]
    check_source(a.checkout, a.source_commit)
    guard(a.deadline_epoch, a.output.parent)
    paths = {name: Path(info["path"]).resolve() for name, info in spec["inputs"].items()}
    assert set(paths) == {"initial_weights", "book", "experiment_config", "protocol"}
    for name, path in paths.items():
        assert sha(path) == spec["inputs"][name]["sha256"]
    assert (
        sha(paths["initial_weights"])
        == "e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03"
    )
    a.weights, a.book, a.config, a.protocol = (
        paths[name] for name in ("initial_weights", "book", "experiment_config", "protocol")
    )
    import torch

    import harbichess.training.torch_ownsearch_checkpoint as cm
    import harbichess.training.torch_ownsearch_learner as lm
    from harbichess.backends.torch_network import load_weights
    from harbichess.selfplay.online_actor import OnlineActorConfig
    from harbichess.training.ownsearch_targets import OwnSearchConfig
    from harbichess.training.torch_fullgame_ppo import FullGamePPOTrainConfig
    from harbichess.training.torch_ownsearch_core import OwnSearchObjective
    from harbichess.training.torch_ownsearch_learner import (
        TorchOwnSearchConfig,
        TorchOwnSearchLearner,
        tensor_bits_equal,
    )
    from harbichess.training.torch_ownsearch_run import clean_source

    clean_source(a.source_commit)
    assert Path(lm.__file__).resolve().is_relative_to(a.checkout.resolve() / "src")
    cfg = json.loads(a.config.read_text())
    assert cfg == spec["frozen_config"] and cfg["seed"] in (20261425, 20261426)
    for name, cls in (
        ("actors", OnlineActorConfig),
        ("objective", OwnSearchObjective),
        ("search", OwnSearchConfig),
        ("schedule", FullGamePPOTrainConfig),
    ):
        cfg[name] = cls(**cfg[name])
    config = TorchOwnSearchConfig(**cfg)
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
        meta["schema"] == "ownsearch-supervised-run-v1" and meta["source_commit"] == a.source_commit
    )
    assert meta["absolute_deadline_epoch"] == spec["original_training_deadline_epoch"]
    assert meta["max_epochs"] == epochs and meta["checkpoint_interval"] == 1
    initial = TorchOwnSearchLearner.resume(
        cp / "epoch-00000000", config=config, input_paths=paths, source_commit=a.source_commit
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
        initial, report = audit_epoch(a, config, paths, initial, index, index in selected_epochs)
        path = a.output.parent / f"epoch-{index:08d}-receipt.json"
        publish(path, report)
        receipts.append(path)
        reports.append(report)
    assert sorted(path.name for path in cp.glob("epoch-*")) == [
        f"epoch-{index:08d}" for index in range(epochs + 1)
    ]
    assert sorted(path.name for path in (a.run / "journal").glob("epoch-*.json.gz")) == [
        f"epoch-{index:08d}.json.gz" for index in range(1, epochs + 1)
    ]
    for path in receipts:
        receipt = json.loads(path.read_text())
        assert receipt["source_commit"] == a.source_commit
        for relative, digest in receipt["native_artifact_sha256"].items():
            guard(a.deadline_epoch, a.output.parent)
            assert sha(a.run / relative) == digest
    for name, path in paths.items():
        assert sha(path) == spec["inputs"][name]["sha256"]
    check_source(a.checkout, a.source_commit)
    guard(a.deadline_epoch, a.output.parent)
    publish(
        a.output,
        {
            "schema": "ufuk-ownsearch-allnative-data-audit-v1",
            "status": "pass-all-fixed-ownv1-native-own-data-search-ledgers",
            "source_commit": a.source_commit,
            "seed": config.seed,
            "fixed_epochs": epochs,
            "audited_native_checkpoints": epochs + 1,
            "independently_replayed_fresh_transitions": 32768 * epochs,
            "schedule_and_all128search_rngs_verified": True,
            "prescribed_neural_roots_verified": sum(
                row["prescribed_neural_roots_verified"] for row in reports
            ),
            "raw_actor_packet_roots_verified": sum(
                row["raw_actor_packet_roots_verified"] for row in reports
            ),
            "neural_search_roots_recomputed": sum(
                row["exact_fullgroup_neural_roots_recomputed"] for row in reports
            ),
            "duplicate_neural_positions": sum(row["duplicate_neural_positions"] for row in reports),
            "neural_audit_epochs": selected_epochs,
            "sample_chain_sha256": initial.sample_chain_sha256,
            "optimizer_updates_performed_by_this_audit": 0,
            "manifest_sha256": sha(a.manifest),
            "audit_sha256": sha(Path(__file__)),
            "support_sha256": sha(Path(__file__).with_name("ownv1_audit_support.py")),
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
                "behaviorKLgame-balanced; original fullwavefrontgroups replay "
                "with108designated roots "
                "and reportedextras; no optimizer/strength claim"
            ),
        },
    )


if __name__ == "__main__":
    main()
