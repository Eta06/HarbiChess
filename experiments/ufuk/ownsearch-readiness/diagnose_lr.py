"""One prospectively fixed learning-rate check on already audited development data."""

import argparse
import gzip
import json
import time
from pathlib import Path

from qualify_cuda_owned import check_source, guard, publish, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkout", "profile", "output", "config", "registration", "weights", "book"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--deadline-epoch", type=float, required=True)
    a = p.parse_args()
    started = time.time()
    if not __debug__ or not 0 < a.deadline_epoch - started <= 300:
        raise ValueError("One unoptimized Python invocation within original 300s ceiling")
    check_source(a.checkout, a.source_commit)
    a.output.mkdir(exist_ok=False)
    reg = json.loads(a.registration.read_text())
    bindings = {
        "profile_result": a.profile / "result.json",
        "profile_journal": a.profile / "run/journal/epoch-00000001.json.gz",
        "config": a.config,
        "weights": a.weights,
        "book": a.book,
        "helper": Path(__file__),
        "common_helper": Path(__file__).with_name("qualify_cuda_owned.py"),
    }
    actual = {k: sha(v) for k, v in bindings.items()}
    if actual != reg["input_and_helper_sha256"] or reg["source_commit"] != a.source_commit:
        raise ValueError("Prospectively pinned development inputs/source/helpers differ")
    profile = json.loads(bindings["profile_result"].read_text())
    if profile["status"] != "pass-one-development-epoch-and-readonly-audit":
        raise ValueError("Prior all-data independent profile audit missing")
    if profile["cli_result"]["optimizer_accepted_updates"] != 0:
        raise ValueError("This diagnosis requires original unchanged e8 behaviour")
    guard(a.deadline_epoch, a.output)
    result = dict(
        schema="ownsearch-fixed-lr-duplicate-development-diagnosis-v1",
        status="failed-preserved",
        source_commit=a.source_commit,
        registration_sha256=sha(a.registration),
        input_and_helper_sha256=actual,
        started_epoch=started,
        absolute_deadline_epoch=a.deadline_epoch,
        new_selfplay_transitions=0,
        new_search_roots=0,
        scope="Duplicate development fitting only; not formal training/resume/strength",
    )
    try:
        import torch

        from harbichess.selfplay.online_actor import OnlineActorConfig
        from harbichess.selfplay.online_epoch import deserialize_policy_epoch
        from harbichess.training.fullgame_own_targets import build_fullgame_targets
        from harbichess.training.ownsearch_targets import OwnSearchConfig
        from harbichess.training.torch_fullgame_ppo import (
            FullGamePPOTrainConfig,
            torch_model_digest,
        )
        from harbichess.training.torch_ownsearch_core import (
            OwnSearchObjective,
            search_train_rows,
            train_search_epoch,
        )
        from harbichess.training.torch_ownsearch_learner import (
            TorchOwnSearchConfig,
            TorchOwnSearchLearner,
            canonical,
            tensor_bits_equal,
        )

        raw = json.loads(a.config.read_text())
        if raw["learning_rate"] != 0.000025 or raw["objective"]["behavior_kl_stop"] != 0.02:
            raise ValueError("Requires one preregistered quarter LR and original KL limit")
        raw["actors"] = OnlineActorConfig(**raw["actors"])
        raw["search"] = OwnSearchConfig(**raw["search"])
        raw["objective"] = OwnSearchObjective(**raw["objective"])
        raw["schedule"] = FullGamePPOTrainConfig(**raw["schedule"])
        config = TorchOwnSearchConfig(**raw)
        if config.device != "cuda:0" or config.schedule.passes != 4:
            raise ValueError("Requires actual CUDA and original four passes")
        learner = TorchOwnSearchLearner.fresh(
            config=config,
            input_paths=dict(initial_weights=a.weights, book=a.book,
                             experiment_config=a.config, protocol=a.registration),
            source_commit=a.source_commit,
        )
        record = json.loads(gzip.decompress(bindings["profile_journal"].read_bytes()))
        epoch = deserialize_policy_epoch(canonical(record["collection"]))
        before_digest = torch_model_digest(learner.online)
        if before_digest != epoch.model_digest or len(epoch.actions) != 32768:
            raise ValueError("Archived behaviour must equal fresh immutable e8")
        # This same original sampler seed is used; no new actor/search RNG is consumed.
        sampler_seed = learner.sampler_seed_rng.randrange(2**63)
        if sampler_seed != record["sampler_seed"]:
            raise ValueError("Original deterministic minibatch seed differs")
        actor_cursor = canonical(learner.actors.cursor())
        actor_rng = learner.actors.rng.getstate()
        schedule_rng = learner.schedule_rng.getstate()
        search_rng = [r.getstate() for r in learner.search_rngs]
        targets = build_fullgame_targets(
            learner.actors.rules, epoch, claim_draw=config.actors.claim_draw
        )
        rows = search_train_rows(epoch, record["own_search"])
        if len(rows) != 4095 or len(targets.targets) != 18264:
            raise ValueError("Original audited root/value inventories differ")
        guard(a.deadline_epoch, a.output)
        training = train_search_epoch(
            learner.online, policy_rows=rows, value_targets=targets,
            optimizer=learner.optimizer, rules=learner.actors.rules,
            encoder=learner.encoder, seed=sampler_seed, device=config.device,
            objective=config.objective, schedule=config.schedule,
            guard=lambda: guard(a.deadline_epoch, a.output),
        )
        torch.cuda.synchronize()
        if actor_cursor != canonical(learner.actors.cursor()):
            raise ValueError("Diagnosis consumed actor transitions")
        assert learner.actors.rng.getstate() == actor_rng
        assert learner.schedule_rng.getstate() == schedule_rng
        assert [r.getstate() for r in learner.search_rngs] == search_rng
        for name, value in learner.online.state_dict().items():
            assert torch.isfinite(value).all()
            if name.startswith("material_value_linear."):
                assert tensor_bits_equal(value, learner.base.state_dict()[name])
        after_digest = torch_model_digest(learner.online)
        result.update(
            status="completed-duplicate-only-no-strength-claim",
            fresh_e8_before_digest=before_digest,
            diagnostic_after_digest=after_digest,
            retained_model_changed=before_digest != after_digest,
            independent_terminal_rows=len(targets.targets),
            unknown_excluded_value_rows=targets.excluded_actions,
            searched_policy_rows=len(rows),
            duplicate_minibatch_row_presentations=(
                training["optimizer_steps_attempted"] * config.schedule.minibatch_size * 2
            ),
            training={k: v for k, v in training.items() if "rng" not in k},
            all_raw_actor_schedule_search_rng_untouched=True,
            all21_unused_material_storage_preserved=True,
            native_checkpoint_created=False,
        )
        guard(a.deadline_epoch, a.output)
        check_source(a.checkout, a.source_commit)
        if actual != {k: sha(v) for k, v in bindings.items()}:
            raise ValueError("Pinned input changed during duplicate fitting")
    except BaseException as exc:
        result["error"] = repr(exc)
        raise
    finally:
        result["finished_epoch"] = time.time()
        result["whole_seconds"] = result["finished_epoch"] - started
        publish(a.output / "result.json", result)


if __name__ == "__main__":
    main()
