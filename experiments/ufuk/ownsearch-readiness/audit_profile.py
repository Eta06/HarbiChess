"""Read-only one-epoch audit; first/last whole search batches selected prospectively."""

import argparse
import gzip
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from qualify_cuda_owned import check_source, guard, publish, sha


def main():
    if not __debug__:
        raise RuntimeError("optimized Python disables required audit assertions")
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkout", "run", "output", "weights", "book", "config", "protocol"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--deadline-epoch", type=float, required=True)
    a = p.parse_args()
    guard(a.deadline_epoch, a.run)
    check_source(a.checkout, a.source_commit)
    import torch

    from harbichess.chess.actions import legal_action_indices
    from harbichess.selfplay.online_actor import OnlineActorConfig
    from harbichess.selfplay.online_epoch import _tuplify, deserialize_policy_epoch
    from harbichess.training.fullgame_own_targets import (
        GameBalancedSampler,
        build_fullgame_targets,
    )
    from harbichess.training.ownsearch_targets import (
        OwnSearchConfig,
        make_search_ledger,
    )
    from harbichess.training.torch_fullgame_ppo import FullGamePPOTrainConfig
    from harbichess.training.torch_ownsearch_core import OwnSearchObjective
    from harbichess.training.torch_ownsearch_learner import (
        TorchOwnSearchConfig,
        TorchOwnSearchLearner,
        canonical,
        tensor_bits_equal,
    )
    from harbichess.training.torch_ownsearch_run import clean_source

    clean_source(a.source_commit)
    c = json.loads(a.config.read_text())
    c["actors"] = OnlineActorConfig(**c["actors"])
    c["search"] = OwnSearchConfig(**c["search"])
    c["objective"] = OwnSearchObjective(**c["objective"])
    c["schedule"] = FullGamePPOTrainConfig(**c["schedule"])
    config = TorchOwnSearchConfig(**c)
    paths = dict(
        initial_weights=a.weights,
        book=a.book,
        experiment_config=a.config,
        protocol=a.protocol,
    )
    cp = a.run / "checkpoints"
    artifacts = [f for n in (0, 1) for f in (cp / f"epoch-{n:08d}").iterdir()]
    journal = a.run / "journal/epoch-00000001.json.gz"
    artifacts.append(journal)
    before = {str(f.relative_to(a.run)): sha(f) for f in artifacts}
    before_inputs = {k: sha(v) for k, v in paths.items()}
    initial = TorchOwnSearchLearner.resume(
        cp / "epoch-00000000",
        config=config,
        input_paths=paths,
        source_commit=a.source_commit,
    )
    guard(a.deadline_epoch, a.run)
    initial_models = {
        name: {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        for name, model in [("online", initial.online), ("base", initial.base)]
    }
    schedule_before = initial.schedule_rng.getstate()
    search_before = [r.getstate() for r in initial.search_rngs]
    sampler_seed = initial.sampler_seed_rng.randrange(2**63)
    sampler_after = initial.sampler_seed_rng.getstate()
    final = TorchOwnSearchLearner.resume(
        cp / "epoch-00000001",
        config=config,
        input_paths=paths,
        source_commit=a.source_commit,
    )
    guard(a.deadline_epoch, a.run)
    assert final.config.device == "cuda:0" and final.closed and final.epoch == 1
    assert initial.epoch == 0 and initial.closed
    assert final.last_epoch_gzip == journal.read_bytes()
    record = json.loads(gzip.decompress(journal.read_bytes()))
    epoch = deserialize_policy_epoch(canonical(record["collection"]))
    assert len(epoch.actions) == record["fresh_transitions"] == 32768
    assert final.actors.steps == record["actor_steps"] == 256
    assert record["total_fresh_transitions"] == 32768
    assert (
        record["sampler_seed"] == sampler_seed
        and final.sampler_seed_rng.getstate() == sampler_after
    )
    ledger = record["own_search"]
    assert _tuplify(ledger["schedule_rng_before"]) == schedule_before
    assert [_tuplify(r) for r in ledger["search_rng_before"]] == search_before
    # Independently consume raw behavior probabilities through the original actor RNG.
    for offset in range(0, len(epoch.actions), config.actors.games):
        guard(a.deadline_epoch, a.run)
        rows = epoch.actions[offset : offset + config.actors.games]
        for row in rows:
            logs = [math.log(v) if v > 0 else -math.inf for v in row.policy]
            scaled = [math.exp(v - max(logs)) for v in logs]
            mu = tuple(v / math.fsum(scaled) for v in scaled)
            assert mu == row.behavior_policy
            assert (
                tuple(
                    legal_action_indices(
                        initial.actors.rules.inspect(row.transition.pre)
                    )
                )
                == row.legal_actions
            )
        actual = initial.actors.step(tuple(row.policy for row in rows))
        assert actual == tuple(row.transition for row in rows)
    assert canonical(initial.actors.close_policy_epoch()) == canonical(
        epoch.truncations
    )
    assert initial.actors.cursor() == final.actors.cursor() == epoch.next_actor_cursor
    assert (
        initial.actors.rng.getstate()
        == final.actors.rng.getstate()
        == epoch.next_actor_rng_state
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
    targets = build_fullgame_targets(
        final.actors.rules, epoch, claim_draw=config.actors.claim_draw
    )
    assert (
        len(targets.targets)
        == len(expected)
        == record["training"]["trained_transitions"]
    )
    assert (
        targets.excluded_actions
        == unknown
        == record["target_counts"]["excluded_actions"]
    )
    for target in targets.targets:
        assert target.target_wdl == expected[target.transition]
    # Reproduce accepted sampler draws; rejected passes restore both pass-start RNGs.
    training = record["training"]
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
    assert canonical(sampler_initial) == canonical(
        training["sampler_rng_state_before_epoch"]
    )
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
        assert tensor_bits_equal(value.cpu(), initial_models["base"][name])
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
                value.detach()
                .cpu()
                .contiguous()
                .reshape(-1)
                .view(torch.uint8)
                .numpy()
                .tobytes()
            ).hexdigest()
    assert material_elements == 21
    # Recreate the ORIGINAL full wavefront composition, never scalar/subset batch shapes.
    queues = {slot: [] for slot in range(config.actors.games)}
    for i in ledger["selected_indices"]:
        queues[epoch.actions[i].transition.slot].append(i)
    batches = []
    while any(queues.values()):
        batches.append([rows.pop(0) for rows in queues.values() if rows])
    chosen = {0, len(batches) - 1}
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
            assert canonical(root) == canonical(root_by_index[root["collection_index"]])
        audited += len(indices)
        neural_positions += subset["neural_positions"]
    assert [r.getstate() for r in rngs] == [
        _tuplify(r) for r in ledger["search_rng_after"]
    ]
    assert before == {str(f.relative_to(a.run)): sha(f) for f in artifacts}
    assert before_inputs == {k: sha(v) for k, v in paths.items()}
    guard(a.deadline_epoch, a.run)
    clean_source(a.source_commit)
    publish(
        a.output,
        dict(
            schema="ownsearch-development-independent-audit-v1",
            status="pass",
            source_commit=a.source_commit,
            native_artifact_sha256=before,
            input_sha256=before_inputs,
            epoch=1,
            fresh_transitions=32768,
            independent_raw_actor_replayed=32768,
            known_value_rows=len(expected),
            UNKNOWN_excluded_value_rows=unknown,
            search_target_rows=len(ledger["roots"]),
            exact_neural_audited_roots=audited,
            neural_audit_selection="first-and-last-complete-original-wavefront-batches",
            duplicate_audit_neural_positions=neural_positions,
            duplicate_audit_gradient_updates=0,
            material_elements=material_elements,
            material_nonzero=material_nonzero,
            material_storage_sha256=material_sha,
            all_base_behavior_and_material_storage_exact=True,
            strict_native0_and_native1_loaded=True,
            finished_epoch=time.time(),
            scope="Development infrastructure only; no candidate/strength eligibility",
        ),
    )


if __name__ == "__main__":
    main()
