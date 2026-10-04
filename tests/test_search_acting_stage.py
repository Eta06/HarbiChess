import copy
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import chess
import pytest
import torch

from harbichess.core.state import ChessState
from harbichess.selfplay.online_actor import (
    ActorOpening,
    OnlineActorConfig,
    OnlineActors,
)
from harbichess.training.ownsearch_targets import OwnSearchConfig, SearchSchedule
from harbichess.training.torch_fullgame_ppo import FullGamePPOTrainConfig
from harbichess.training.torch_ownsearch_core import OwnSearchObjective, ownsearch_loss
from harbichess.training.torch_search_acting_learner import (
    TorchSearchActingConfig,
    TorchSearchActingLearner,
    tensor_bits_equal,
)

HERE = Path(__file__).resolve().parent.parent
WEIGHTS = Path(
    os.environ.get(
        "HARBICHESS_TEST_E8",
        str(
            HERE
            / "artifacts/ufuk-a100-mirror-20261004/harbichess-inputs/initial-e8.safetensors"
        ),
    )
)
SOURCE = (
    "a" * 40
)  # Synthetic unit-native marker; clean CLI requires a real new producer pin.


def fixture(tmp_path):
    data = dict(
        seed=20261405,
        actors=dict(games=4, max_additional_plies=8, claim_draw=True, temperature=1.0),
        objective=dict(
            policy_weight=1.0,
            value_weight=0.2,
            policy_anchor_weight=0.03,
            value_anchor_weight=0.02,
            behavior_kl_stop=1.0,
        ),
        search=dict(
            simulations=16,
            max_considered_actions=4,
            gumbel_scale=0.0,
            value_scale=0.1,
            maxvisit_init=50.0,
            block_plies=8,
        ),
        schedule=dict(minibatch_size=4, passes=2, max_gradient_norm=5.0),
        epoch_steps=16,
        learning_rate=1e-4,
        weight_decay=1e-4,
        device="cpu",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(data) + "\n")
    book = tmp_path / "book.json"
    book.write_text(
        json.dumps(
            dict(
                schema=1,
                splits=dict(
                    train=[
                        dict(
                            source_game="white",
                            root_ply=0,
                            opening=dict(
                                root_fen="7k/5Q2/6K1/8/8/8/8/8 w - - 0 1", moves=[]
                            ),
                        ),
                        dict(
                            source_game="black",
                            root_ply=0,
                            opening=dict(
                                root_fen="8/8/8/8/8/6k1/5q2/7K b - - 0 1", moves=[]
                            ),
                        ),
                        dict(
                            source_game="broad",
                            root_ply=0,
                            opening=dict(root_fen=chess.STARTING_FEN, moves=[]),
                        ),
                    ]
                ),
            )
        )
    )
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        '{"scope":"local CPU unit infrastructure only, not production"}\n'
    )
    return config_path, dict(
        initial_weights=WEIGHTS,
        book=book,
        experiment_config=config_path,
        protocol=protocol,
    )


def config(path):
    data = json.loads(path.read_text())
    data["actors"] = OnlineActorConfig(**data["actors"])
    data["objective"] = OwnSearchObjective(**data["objective"])
    data["search"] = OwnSearchConfig(**data["search"])
    data["schedule"] = FullGamePPOTrainConfig(**data["schedule"])
    return TorchSearchActingConfig(**data)


def test_schedule_before_outcomes_uses_separate_rng_and_both_parities():
    opening = ActorOpening("broad", ChessState(chess.STARTING_FEN))
    selected = []
    for seed in range(80):
        actors = OnlineActors(
            (opening,), config=OnlineActorConfig(1, 8, True, 1.0), rng=random.Random(99)
        )
        schedule = SearchSchedule(rng=random.Random(seed), block_plies=8)
        actor_before = actors.rng.getstate()
        schedule.before_actor_step(actors)
        assert actors.rng.getstate() == actor_before
        chosen = schedule.offsets[0]["chosen_offset"]
        selected.append(chosen)
        for step in range(8):
            if step:
                schedule.before_actor_step(actors)
            moves = actors.rules.legal_moves(actors.games[0].state)
            actors.step((tuple(1 / len(moves) for _ in moves),))
        assert schedule.selected == [chosen]
        assert len(schedule.offsets) == 1
    assert set(selected) == set(range(8))
    assert any(x % 2 == 0 for x in selected) and any(x % 2 == 1 for x in selected)


def test_unknown_and_illegal_zero_gradient_terminal_white_black_pov():
    objective = OwnSearchObjective(1.0, 1.0, 0.0, 0.0, 1.0)
    p = torch.zeros(2, 3, requires_grad=True)
    v = torch.zeros(2, 3, requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, False, True]])
    target = torch.tensor([[0.3, 0.7, 0.0], [0.4, 0.0, 0.6]])
    loss, _ = ownsearch_loss(
        p, v, mask, target, target, torch.full((2, 3), 1 / 3), config=objective
    )
    loss.backward()
    assert torch.equal(v.grad, torch.zeros_like(v)) and torch.equal(
        p.grad[~mask], torch.zeros_like(p.grad[~mask])
    )
    terminal = torch.zeros(2, 3, requires_grad=True)
    # Same white win, white mover WDL=W and black mover WDL=L.
    loss, _ = ownsearch_loss(
        p,
        v,
        mask,
        target,
        target,
        torch.full((2, 3), 1 / 3),
        terminal_logits=terminal,
        terminal_targets=torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        config=objective,
    )
    loss.backward()
    assert terminal.grad[0, 0] < 0 and terminal.grad[1, 2] < 0


def test_actual_e8_native_partial_rejection_and_unused_storage(tmp_path):
    torch.set_num_threads(1)
    path, inputs = fixture(tmp_path)
    learner = TorchSearchActingLearner.fresh(
        config=config(path), input_paths=inputs, source_commit=SOURCE
    )
    unused = {
        n: p.detach().clone()
        for n, p in learner.online.named_parameters()
        if n.startswith("material_value_linear.")
    }
    assert (
        sum(p.numel() for p in unused.values()) == 21
        and sum(torch.count_nonzero(p).item() for p in unused.values()) == 20
    )
    learner.closed = False
    with pytest.raises(ValueError, match="partial"):
        learner.checkpoint(tmp_path / "partial")
    assert not (tmp_path / "partial").exists()
    learner.closed = True
    record = learner.train_epoch()
    assert (
        record["training"]["policy_target_rows"] > 0
        and record["training"]["optimizer_steps_committed"] > 0
    )
    assert record["target_counts"]["excluded_actions"] > 0
    assert any(
        not math.isclose(
            row["transition"]["policy_probability"],
            row["transition"]["behavior_probability"],
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        for row in record["collection"]["actions"]
    )
    assert learner.base.training is False and all(
        not p.requires_grad for p in learner.base.parameters()
    )
    for n, p in unused.items():
        assert (
            tensor_bits_equal(p, learner.online.state_dict()[n])
            and tensor_bits_equal(p, learner.base.state_dict()[n])
            and tensor_bits_equal(p, learner.behavior.state_dict()[n])
        )
    learner.checkpoint(tmp_path / "native1")
    restored = TorchSearchActingLearner.resume(
        tmp_path / "native1",
        config=config(path),
        input_paths=inputs,
        source_commit=SOURCE,
    )
    assert restored.schedule_rng.getstate() == learner.schedule_rng.getstate()
    assert [r.getstate() for r in restored.search_rngs] == [
        r.getstate() for r in learner.search_rngs
    ]
    assert restored.last_epoch_gzip == learner.last_epoch_gzip
    with pytest.raises(ValueError, match="source"):
        TorchSearchActingLearner.resume(
            tmp_path / "native1",
            config=config(path),
            input_paths=inputs,
            source_commit="0" * 40,
        )


PROCESS = r"""
import json,sys,torch
from pathlib import Path
from harbichess.selfplay.online_actor import OnlineActorConfig
from harbichess.training.ownsearch_targets import OwnSearchConfig
from harbichess.training.torch_fullgame_ppo import FullGamePPOTrainConfig
from harbichess.training.torch_ownsearch_core import OwnSearchObjective
from harbichess.training.torch_search_acting_learner import (
 TorchSearchActingConfig,TorchSearchActingLearner,
)
root,cfg,weights,source,mode=sys.argv[1:];root=Path(root);cfg=Path(cfg)
c=json.loads(cfg.read_text());c['actors']=OnlineActorConfig(**c['actors']);c['objective']=OwnSearchObjective(**c['objective']);c['search']=OwnSearchConfig(**c['search']);c['schedule']=FullGamePPOTrainConfig(**c['schedule'])
inputs=dict(initial_weights=Path(weights),book=cfg.parent/'book.json',experiment_config=cfg,protocol=cfg.parent/'protocol.json')
torch.set_num_threads(1)
l=(
 TorchSearchActingLearner.resume(
  root/('native2' if mode=='strictload' else 'native1'),config=TorchSearchActingConfig(**c),
  input_paths=inputs,source_commit=source
 ) if mode in ('resume','strictload') else TorchSearchActingLearner.fresh(
  config=TorchSearchActingConfig(**c),input_paths=inputs,source_commit=source
 )
)
root.mkdir(exist_ok=True)
while l.epoch<(1 if mode=='pause' else 2):
 l.train_epoch();(root/f'journal{l.epoch}.gz').write_bytes(l.last_epoch_gzip);l.checkpoint(root/f'native{l.epoch}')
print(json.dumps(dict(epoch=l.epoch,accepted=l.optimizer_accepted_updates,search_roots=json.loads(__import__('gzip').decompress(l.last_epoch_gzip))['training']['policy_target_rows'])))
"""


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="actual CUDA unavailable on local host; root qualification required",
            ),
        ),
    ],
)
def test_real_fresh_process_two_vs_pause_one_resume_two_native_bytes(tmp_path, device):
    path, _ = fixture(tmp_path)
    raw = json.loads(path.read_text())
    raw["device"] = device
    path.write_text(json.dumps(raw) + "\n")
    env = dict(
        os.environ,
        PYTHONPATH=str(HERE / "src"),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )
    for root, mode in (
        (tmp_path / "whole", "whole"),
        (tmp_path / "split", "pause"),
        (tmp_path / "split", "resume"),
        (tmp_path / "whole", "strictload"),
        (tmp_path / "split", "strictload"),
    ):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                PROCESS,
                str(root),
                str(path),
                str(WEIGHTS),
                SOURCE,
                mode,
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stderr
    for name in (
        "model.safetensors",
        "base.safetensors",
        "behavior.safetensors",
        "training.pt",
        "actor.json",
        "last-frozen-epoch.json.gz",
    ):
        assert (tmp_path / "whole/native2" / name).read_bytes() == (
            tmp_path / "split/native2" / name
        ).read_bytes(), name
    assert (tmp_path / "whole/journal1.gz").read_bytes() == (
        tmp_path / "split/journal1.gz"
    ).read_bytes()
    assert (tmp_path / "whole/journal2.gz").read_bytes() == (
        tmp_path / "split/journal2.gz"
    ).read_bytes()
    native = json.loads((tmp_path / "whole/native2/checkpoint.json").read_text())
    assert (
        native["schema"]
        == (
            "torch-search-acting-native-cuda-v2"
            if device == "cuda:0"
            else "torch-search-acting-native-cpu-v2"
        )
        and native["state"]["pending_search_schedule"] == "closed-empty"
    )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="actual CUDA unavailable locally; mandatory A100 gate",
            ),
        ),
    ],
)
def test_rejected_pass_restores_model_adam_globals_and_both_sampler_rngs(
    tmp_path, monkeypatch, device
):
    import numpy as np

    import harbichess.training.torch_ownsearch_core as core

    torch.set_num_threads(1)
    path, inputs = fixture(tmp_path)
    raw = json.loads(path.read_text())
    raw["device"] = device
    path.write_text(json.dumps(raw) + "\n")
    learner = TorchSearchActingLearner.fresh(
        config=config(path), input_paths=inputs, source_commit=SOURCE
    )
    retained = learner.train_epoch()
    assert retained["training"]["optimizer_steps_committed"] > 0
    model = copy.deepcopy(learner.online.state_dict())
    optimizer = copy.deepcopy(learner.optimizer.state_dict())
    globals_before = (
        random.getstate(),
        copy.deepcopy(np.random.get_state()),
        torch.get_rng_state().clone(),
    )
    cuda_before = torch.cuda.get_rng_state_all() if device == "cuda:0" else []
    real = core._whole_dataset_kls
    calls = 0

    def force_reject(*args, **kwargs):
        nonlocal calls
        calls += 1
        value = real(*args, **kwargs)
        if calls > 1:
            random.random()
            np.random.random()
            torch.rand(3)
            if device == "cuda:0":
                torch.rand(3, device=device)
            return learner.config.objective.behavior_kl_stop + 1, value[1]
        return value

    monkeypatch.setattr(core, "_whole_dataset_kls", force_reject)
    record = learner.train_epoch()
    assert (
        record["training"]["optimizer_steps_attempted"] > 0
        and record["training"]["optimizer_steps_committed"] == 0
    )
    assert (
        record["training"]["sampler_rng_state"]
        == record["training"]["sampler_rng_state_before_epoch"]
    )
    actual_optimizer = learner.optimizer.state_dict()
    assert optimizer["param_groups"] == actual_optimizer["param_groups"]
    for key, state in optimizer["state"].items():
        for name, value in state.items():
            actual_value = actual_optimizer["state"][key][name]
            assert (
                tensor_bits_equal(value, actual_value)
                if isinstance(value, torch.Tensor)
                else value == actual_value
            )
    assert all(
        tensor_bits_equal(value, learner.online.state_dict()[name])
        for name, value in model.items()
    )
    assert random.getstate() == globals_before[0]
    actual = np.random.get_state()
    assert (
        actual[0] == globals_before[1][0]
        and np.array_equal(actual[1], globals_before[1][1])
        and actual[2:] == globals_before[1][2:]
    )
    assert torch.equal(torch.get_rng_state(), globals_before[2])
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            cuda_before,
            torch.cuda.get_rng_state_all() if device == "cuda:0" else [],
            strict=True,
        )
    )


def test_closed_ledger_schedule_tamper_rejected(tmp_path):
    from harbichess.training.search_acting_epoch import (
        deserialize_search_acting_epoch as deserialize_policy_epoch,
    )
    from harbichess.training.search_acting_epoch import validate_search_acting

    torch.set_num_threads(1)
    path, inputs = fixture(tmp_path)
    learner = TorchSearchActingLearner.fresh(
        config=config(path), input_paths=inputs, source_commit=SOURCE
    )
    record = learner.train_epoch()
    epoch = deserialize_policy_epoch(json.dumps(record["collection"]).encode())
    changed = copy.deepcopy(record["own_search"])
    changed["offsets"][0]["chosen_offset"] = (
        changed["offsets"][0]["chosen_offset"] + 1
    ) % 8
    with pytest.raises(ValueError, match="schedule"):
        validate_search_acting(
            epoch, changed, learner.config.search, actors=learner.actors
        )


def test_zero_search_selection_still_trains_fresh_known_outcomes(tmp_path):
    from dataclasses import replace

    torch.set_num_threads(1)
    path, inputs = fixture(tmp_path)
    settings = config(path)
    settings = replace(settings, search=replace(settings.search, block_plies=999999))
    # This synthetic local config must agree with its immutable config input.
    raw = json.loads(path.read_text())
    raw["search"]["block_plies"] = 999999
    path.write_text(json.dumps(raw) + "\n")
    learner = TorchSearchActingLearner.fresh(
        config=settings, input_paths=inputs, source_commit=SOURCE
    )
    result = learner.train_epoch()
    assert result["training"]["policy_target_rows"] == 0
    assert result["training"]["trained_transitions"] > 0
    assert result["training"]["optimizer_steps_committed"] > 0
    assert result["own_search"]["neural_positions"] == 0
    learner.checkpoint(tmp_path / "native")
    restored = TorchSearchActingLearner.resume(
        tmp_path / "native", config=settings, input_paths=inputs, source_commit=SOURCE
    )
    assert restored.epoch == 1


@pytest.mark.parametrize(
    "fen", ["7k/6Q1/5K2/8/8/8/8/8 b - - 0 1", "8/8/8/8/8/5k2/6q1/7K w - - 0 1"]
)
def test_known_terminal_root_never_calls_network_both_movers(fen):
    from harbichess.chess.rules import PythonChessRules
    from harbichess.search.full_gumbel import FullGumbelConfig
    from harbichess.search.ownsearch_wavefront import WavefrontGumbel

    class NoNetwork:
        def evaluate_many(self, states):
            assert not states
            return ()

    result = WavefrontGumbel(
        NoNetwork(),
        PythonChessRules(),
        FullGumbelConfig(simulations=16, max_considered_actions=4),
    ).search_many([ChessState(fen)], [random.Random(7)])[0]
    assert (
        result.root_value == -1.0
        and result.simulations == 0
        and not result.action_weights
    )


def test_same_record_policy_CE_zero_ablation_changes_only_policy_term():
    from dataclasses import replace

    p = torch.tensor([[0.2, -0.2]], requires_grad=True)
    v = torch.zeros(1, 3, requires_grad=True)
    mask = torch.ones(1, 2, dtype=torch.bool)
    target = torch.tensor([[0.1, 0.9]])
    base = torch.tensor([[0.5, 0.5]])
    objective = OwnSearchObjective(1.0, 0.2, 0.03, 0.02, 1.0)
    active, terms = ownsearch_loss(
        p, v, mask, target, base, torch.full((1, 3), 1 / 3), config=objective
    )
    control, _ = ownsearch_loss(
        p,
        v,
        mask,
        target,
        base,
        torch.full((1, 3), 1 / 3),
        config=replace(objective, policy_weight=0.0),
    )
    assert torch.equal(active - control, terms["policy"])
    gradients = torch.autograd.grad(terms["policy"], p)[0]
    assert gradients[0, 0] > 0 and gradients[0, 1] < 0


def test_preaction_masked_search_and_actual_mu_raw_pi_archive(tmp_path, monkeypatch):
    import harbichess.training.search_acting_epoch as collector

    torch.set_num_threads(1)
    path, inputs = fixture(tmp_path)
    learner = TorchSearchActingLearner.fresh(
        config=config(path), input_paths=inputs, source_commit=SOURCE
    )
    original = collector.MaskedRootEvaluator
    starts = []

    class Witness(original):
        def __init__(self, evaluator, states, mask):
            assert tuple(states) == learner.actors.states
            starts.append((learner.actors.steps, tuple(mask)))
            super().__init__(evaluator, states, mask)

    monkeypatch.setattr(collector, "MaskedRootEvaluator", Witness)
    record = learner.train_epoch()
    assert starts and all(len(mask) == 4 and any(mask) for _, mask in starts)
    selected = {r["collection_index"]: r for r in record["own_search"]["roots"]}
    assert record["own_search"]["groups"]
    assert record["own_search"]["neural_batch_sizes"][0] == 4
    for i, row in enumerate(record["collection"]["actions"]):
        assert row["policy"] != []
        expected = collector.actor_mu(
            selected[i]["search_policy"] if i in selected else row["policy"]
        )
        assert list(expected) == row["behavior_policy"]
    assert any(
        row["policy"] != row["behavior_policy"]
        for i, row in enumerate(record["collection"]["actions"])
        if i in selected
    )
    epoch = collector.deserialize_search_acting_epoch(
        json.dumps(record["collection"]).encode()
    )
    for field in ("actor_rng_before", "groups", "behavior"):
        changed = copy.deepcopy(record["own_search"])
        if field == "actor_rng_before":
            changed[field] = random.Random(999).getstate()
        elif field == "groups":
            changed[field][0]["selected_slots"] = []
        else:
            changed[field] = "raw-current-T1-not-search-action-policy"
        with pytest.raises(ValueError):
            collector.validate_search_acting(
                epoch, changed, learner.config.search, actors=learner.actors
            )


def test_v1_native_and_non_e8_fresh_rejected(tmp_path):
    from harbichess.training.torch_ownsearch_learner import (
        TorchOwnSearchConfig,
        TorchOwnSearchLearner,
    )

    torch.set_num_threads(1)
    path, inputs = fixture(tmp_path)
    settings = config(path)
    from dataclasses import asdict

    raw = asdict(settings)
    raw["actors"] = settings.actors
    raw["objective"] = settings.objective
    raw["search"] = settings.search
    raw["schedule"] = settings.schedule
    old = TorchOwnSearchLearner.fresh(
        config=TorchOwnSearchConfig(**raw), input_paths=inputs, source_commit=SOURCE
    )
    old.checkpoint(tmp_path / "old-v1")
    with pytest.raises(ValueError, match="schema"):
        TorchSearchActingLearner.resume(
            tmp_path / "old-v1",
            config=settings,
            input_paths=inputs,
            source_commit=SOURCE,
        )
    inputs = dict(inputs, initial_weights=tmp_path / "old-v1/model.safetensors")
    with pytest.raises(ValueError, match="immutable e8"):
        TorchSearchActingLearner.fresh(
            config=settings, input_paths=inputs, source_commit=SOURCE
        )


def test_completed_own_outcomes_and_caps_use_rules_not_search_value(tmp_path):
    from harbichess.training.fullgame_own_targets import build_fullgame_targets
    from harbichess.training.search_acting_epoch import (
        deserialize_search_acting_epoch as deserialize_policy_epoch,
    )

    torch.set_num_threads(1)
    path, inputs = fixture(tmp_path)
    learner = TorchSearchActingLearner.fresh(
        config=config(path), input_paths=inputs, source_commit=SOURCE
    )
    record = learner.train_epoch()
    epoch = deserialize_policy_epoch(json.dumps(record["collection"]).encode())
    targets = build_fullgame_targets(learner.actors.rules, epoch, claim_draw=True)
    assert targets.excluded_actions > 0 and targets.complete_games > 0
    selected = {r["collection_index"] for r in record["own_search"]["roots"]}
    assert selected  # Caps may carry CE, while their entire episode has no WDL label.
    endings = {
        (r.transition.source_id, r.transition.game_index): r.transition.post
        for r in epoch.actions
    }
    movers = set()
    for target in targets.targets:
        mover = learner.actors.rules.view(target.transition.pre).side_to_move
        value = learner.actors.rules.outcome(
            endings[(target.source_id, target.game_index)], claim_draw=True
        ).value_for(mover)
        assert target.target_wdl == (
            float(value == 1),
            float(value == 0),
            float(value == -1),
        )
        movers.add(mover.value)
    assert movers == {"white", "black"}


def test_real_e8_raw_reference_kl_zero_despite_search_mu_at_same_point02_gate(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    import harbichess.training.torch_search_acting_learner as module
    from harbichess.training.torch_fullgame_ppo import _whole_dataset_kls

    torch.set_num_threads(1)
    path, inputs = fixture(tmp_path)
    raw = json.loads(path.read_text())
    raw["objective"]["behavior_kl_stop"] = 0.02
    raw["learning_rate"] = 2.5e-5
    path.write_text(json.dumps(raw) + "\n")
    learner = TorchSearchActingLearner.fresh(
        config=config(path), input_paths=inputs, source_commit=SOURCE
    )
    collection = {}
    measurements = {}
    real_collect = module.collect_search_acting_epoch
    real_train = module.train_search_epoch

    def capture_collection(*args, **kwargs):
        epoch, ledger = real_collect(*args, **kwargs)
        collection.update(epoch=epoch, ledger=ledger)
        return epoch, ledger

    def check_actual_optimizer_rows(network, *, policy_rows, **kwargs):
        epoch, ledger = collection["epoch"], collection["ledger"]
        assert policy_rows
        actual_mu_rows = []
        for row, receipt in zip(policy_rows, ledger["roots"], strict=True):
            actor_row = epoch.actions[receipt["collection_index"]]
            assert row.behavior_policy == actor_row.policy  # RAW frozen NN pi.
            actual_mu_rows.append(
                replace(row, behavior_policy=actor_row.behavior_policy)
            )
        raw_kl, _ = _whole_dataset_kls(
            network, learner.encoder, learner.actors.rules, policy_rows, device="cpu"
        )
        mu_kl, _ = _whole_dataset_kls(
            network, learner.encoder, learner.actors.rules, actual_mu_rows, device="cpu"
        )
        measurements.update(raw_kl=raw_kl, actual_mu_kl=mu_kl)
        assert abs(raw_kl) < 1e-6
        assert mu_kl > learner.config.objective.behavior_kl_stop == 0.02
        return real_train(network, policy_rows=policy_rows, **kwargs)

    monkeypatch.setattr(module, "collect_search_acting_epoch", capture_collection)
    monkeypatch.setattr(module, "train_search_epoch", check_actual_optimizer_rows)
    record = learner.train_epoch()
    assert measurements["actual_mu_kl"] > 0.02
    assert record["training"]["optimizer_steps_committed"] > 0
    assert record["training"]["behavior_kl_reference"] == (
        "frozen-raw-network-policy-on-searched-roots"
    )
    for row, archived in zip(
        collection["epoch"].actions, record["collection"]["actions"], strict=True
    ):
        assert list(row.behavior_policy) == archived["behavior_policy"]
        assert list(row.policy) == archived["policy"]
