"""Fresh-process density journal/rule/actor replay and full native audit; no SGD."""

import argparse
import gzip
import hashlib
import json
from dataclasses import fields
from pathlib import Path

import torch

from harbichess.backends.torch_network import load_weights, sha256
from harbichess.selfplay.online_actor import OnlineActorConfig
from harbichess.training.fullgame_own_targets import build_fullgame_targets
from harbichess.training.ownsearch_targets import OwnSearchConfig
from harbichess.training.search_acting_epoch import (
    deserialize_search_acting_epoch,
    validate_search_acting,
)
from harbichess.training.torch_fullgame_ppo import FullGamePPOTrainConfig
from harbichess.training.torch_ownsearch_core import OwnSearchObjective
from harbichess.training.torch_search_acting_learner import (
    TorchSearchActingConfig,
    TorchSearchActingLearner,
    canonical,
    tensor_bits_equal,
)
from harbichess.training.torch_search_acting_run import clean_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "config", "protocol", "weights", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    clean_source(protocol["source_commit"])
    torch.set_num_threads(1)
    obj = json.loads(args.config.read_text())
    assert obj == protocol["configs"][args.run.name]
    for name, cls in (
        ("actors", OnlineActorConfig),
        ("search", OwnSearchConfig),
        ("schedule", FullGamePPOTrainConfig),
        ("objective", OwnSearchObjective),
    ):
        obj[name] = cls(**obj[name])
    config = TorchSearchActingConfig(**obj)
    paths = {
        "initial_weights": args.weights,
        "book": Path(protocol["book"]["path"]),
        "experiment_config": args.config,
        "protocol": args.protocol,
    }
    assert sha256(paths["book"]) == protocol["book"]["sha256"]
    assert sha256(args.weights) == protocol["initial_e8_sha256"]
    checkpoint = args.run / f"checkpoints/epoch-{protocol['epochs']:08d}"
    protected = [p for p in checkpoint.iterdir() if p.is_file()]
    protected += sorted((args.run / "journal").glob("*.json.gz"))
    before = {str(p): sha256(p) for p in protected}
    learner = TorchSearchActingLearner.resume(
        checkpoint, config=config, input_paths=paths, source_commit=protocol["source_commit"]
    )
    assert learner.closed and learner.epoch == protocol["epochs"]
    assert learner.optimizer_accepted_updates > 0
    initial = load_weights(args.weights).state_dict()
    assert any(not tensor_bits_equal(initial[n], p) for n, p in learner.online.state_dict().items())
    assert all(tensor_bits_equal(initial[n], p) for n, p in learner.base.state_dict().items())
    for n, p in learner.online.state_dict().items():
        assert torch.isfinite(p).all()
        if n.startswith("material_value_linear."):
            assert tensor_bits_equal(initial[n], p)
    rows = []
    chain = hashlib.sha256(b"").hexdigest()
    accepted = attempted = 0
    previous_cursor = previous_actor_rng = previous_schedule_rng = previous_search_rng = None
    for i in range(1, protocol["epochs"] + 1):
        path = args.run / f"journal/epoch-{i:08d}.json.gz"
        record = json.loads(gzip.decompress(path.read_bytes()))
        assert record["epoch"] == i and record["previous_sample_chain_sha256"] == chain
        unhashed = {k: v for k, v in record.items() if k != "sample_chain_sha256"}
        chain = hashlib.sha256(bytes.fromhex(chain) + canonical(unhashed)).hexdigest()
        assert chain == record["sample_chain_sha256"]
        epoch = deserialize_search_acting_epoch(canonical(record["collection"]))
        assert (
            len(epoch.actions)
            == record["fresh_transitions"]
            == config.actors.games * config.epoch_steps
        )
        assert record["actor_steps"] == i * config.epoch_steps
        assert record["total_fresh_transitions"] == i * len(epoch.actions)
        ledger = record["own_search"]
        assert ledger["schema"] == "pre-action-masked-search-behavior-v4"
        if previous_cursor is not None:
            assert ledger["actor_cursor_before"] == previous_cursor
            assert ledger["actor_rng_before"] == previous_actor_rng
            assert ledger["schedule_rng_before"] == previous_schedule_rng
            assert ledger["search_rng_before"] == previous_search_rng
        # Shared validator replays exogenous schedules, all exact certificates,
        # actions/actual-mu/RNG/cursor. This is not an independent NN-search rerun.
        validate_search_acting(epoch, ledger, config.search, actors=learner.actors)
        own = build_fullgame_targets(learner.actors.rules, epoch, claim_draw=True)
        counts = {f.name: getattr(own, f.name) for f in fields(own) if f.name != "targets"}
        assert counts == record["target_counts"]
        selected = {root["collection_index"] for root in ledger["roots"]}
        assert len(selected) == len(ledger["roots"])
        if config.search.block_plies == 1:
            assert selected == set(range(len(epoch.actions)))
        assert record["training"]["policy_target_rows"] == len(selected)
        assert record["training"]["trained_transitions"] == len(own.targets)
        accepted += record["training"]["optimizer_steps_committed"]
        attempted += record["training"]["optimizer_steps_attempted"]
        assert record["optimizer_accepted_updates"] == accepted
        assert record["optimizer_attempted_updates"] == attempted
        assert record["optimizer_rejected_updates"] == attempted - accepted
        matches = sum(
            epoch.actions[root["collection_index"]].transition.action.uci == root["selected_action"]
            for root in ledger["roots"]
        )
        rows.append(
            {
                "epoch": i,
                "journal_sha256": sha256(path),
                "legal_transitions": len(epoch.actions),
                "search_roots": len(selected),
                "actual_move_equals_deterministic_selected": matches,
                "known_value_rows": len(own.targets),
                "target_counts": counts,
                "accepted_updates": accepted,
                "attempted_updates": attempted,
            }
        )
        previous_cursor = json.loads(canonical(epoch.next_actor_cursor))
        previous_actor_rng = json.loads(canonical(epoch.next_actor_rng_state))
        previous_schedule_rng = ledger["schedule_rng_after"]
        previous_search_rng = ledger["search_rng_after"]
    assert chain == learner.sample_chain_sha256
    assert learner.last_epoch_gzip == protected[-1].read_bytes()
    assert accepted == learner.optimizer_accepted_updates
    assert attempted == learner.optimizer_attempted_updates
    assert before == {str(p): sha256(p) for p in protected}
    manifest = json.loads((checkpoint / "checkpoint.json").read_text())
    result = {
        "schema": "cpu-density-fresh-replay-fullnative-v1",
        "status": "pass-data-actor-RNG-certificates-native-not-strength",
        "source_commit": protocol["source_commit"],
        "protocol_sha256": sha256(args.protocol),
        "helper_sha256": sha256(Path(__file__)),
        "run": str(args.run),
        "rows": rows,
        "final_artifacts": manifest["artifacts"],
        "full_native_strict_resume": True,
        "full_actor_replay_and_rules": True,
        "all_files_preserved": True,
        "NN_search_independent_rerun": False,
        "scope": (
            "fresh-process shared rules/certificate/actor validator; "
            "not independent chronological NN search reproduction; "
            "that remains open for a formal confirmation"
        ),
        "GPU_used": False,
        "strength_success_claimed": False,
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "run": args.run.name,
                "transitions": sum(r["legal_transitions"] for r in rows),
                "accepted_updates": accepted,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
