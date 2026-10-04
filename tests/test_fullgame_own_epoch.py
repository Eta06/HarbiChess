import copy
import random
from dataclasses import replace

import numpy as np
import pytest
import torch

import harbichess.training.torch_fullgame_ppo as fullgame_ppo
from harbichess.backends.torch_network import TorchChessNetwork
from harbichess.chess.actions import legal_action_indices, move_to_action
from harbichess.chess.encoding import BoardEncoder
from harbichess.chess.rules import PythonChessRules
from harbichess.core.network_config import NetworkConfig
from harbichess.selfplay.online_actor import ActorOpening, OnlineActorConfig, OnlineActors
from harbichess.selfplay.online_epoch import (
    EpochPolicyOutput,
    collect_policy_epoch,
    deserialize_policy_epoch,
    serialize_policy_epoch,
)
from harbichess.training.fullgame_own_targets import (
    GameBalancedSampler,
    build_fullgame_targets,
)
from harbichess.training.torch_fullgame_ppo import (
    FullGamePPOConfig,
    FullGamePPOTrainConfig,
    fullgame_ppo_loss,
    make_torch_epoch_inference,
    torch_model_digest,
    train_fullgame_policy_epoch,
)


def setup_actor(*, cap=4):
    rules = PythonChessRules()
    config = OnlineActorConfig(games=1, max_additional_plies=cap, claim_draw=False, temperature=1)
    actors = OnlineActors(
        (ActorOpening("heldout-free-train-opening", rules.initial_state()),),
        config=config,
        rng=random.Random(41),
    )
    return rules, config, actors


def fool_mate_policy(rules, states, legal):
    moves = ("f2f3", "e7e5", "g2g4", "d8h4")
    policies, wdls, bases = [], [], []
    for state, support in zip(states, legal, strict=True):
        board = rules.inspect(state)
        target = moves[state.ply]
        chosen = move_to_action(board, board.parse_uci(target))
        index = support.index(chosen)
        policy = [0.0] * len(support)
        policy[index] = 1.0
        policies.append(tuple(policy))
        bases.append(tuple(1 / len(support) for _ in support))
        wdls.append((0.4, 0.3, 0.3))
    return EpochPolicyOutput(
        tuple(policies), tuple(wdls), tuple(bases), tuple(wdls)
    )


def test_frozen_epoch_collects_terminal_episode_and_explicitly_closes_next_game():
    rules, _, actors = setup_actor()
    epoch = collect_policy_epoch(
        actors,
        steps=4,
        infer=lambda states, legal: fool_mate_policy(rules, states, legal),
        model_digest=lambda: "sha256:frozen-e8-snapshot",
    )
    assert epoch.schema == "full-history-frozen-policy-epoch-v1"
    assert epoch.start_actor_step == 0 and epoch.end_actor_step == 4
    assert len(epoch.actions) == 4
    assert epoch.truncations == ()
    assert deserialize_policy_epoch(serialize_policy_epoch(epoch)) == epoch
    assert serialize_policy_epoch(deserialize_policy_epoch(serialize_policy_epoch(epoch))) == (
        serialize_policy_epoch(epoch)
    )
    assert len({row.transition.game_index for row in epoch.actions}) == 1
    labels = build_fullgame_targets(rules, epoch, claim_draw=False)
    assert labels.complete_games == labels.loss_games == 1
    assert labels.draw_games == labels.win_games == 0
    assert labels.normal_cap_games == labels.epoch_truncated_games == 0
    assert len(labels.targets) == 4
    assert [row.target_wdl for row in labels.targets] == [
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
    ]
    assert all(row.game_length == 4 and row.action_ratio_old == 1 for row in labels.targets)
    assert all(row.base_value_wdl == (0.4, 0.3, 0.3) for row in labels.targets)
    assert [row.transition.pre.ply for row in labels.targets] == [0, 1, 2, 3]
    sampler_rows = tuple(
        [*labels.targets, *(
            replace(row, game_index=100, source_id="second-game")
            for row in labels.targets[:1]
        )]
    )
    sampled = GameBalancedSampler(sampler_rows, seed=77).sample(2000)
    first_fraction = sum(row.game_index == labels.targets[0].game_index for row in sampled) / len(
        sampled
    )
    assert first_fraction == pytest.approx(0.5, abs=0.05)


def test_unknown_cap_and_epoch_truncation_never_become_draw_targets():
    rules, _, actors = setup_actor(cap=1)
    epoch = collect_policy_epoch(
        actors,
        steps=1,
        infer=lambda states, legal: EpochPolicyOutput(
            tuple(tuple(1 / len(row) for _ in row) for row in legal),
            ((0.3, 0.4, 0.3),),
            tuple(tuple(1 / len(row) for _ in row) for row in legal),
            ((0.2, 0.5, 0.3),),
        ),
        model_digest=lambda: "frozen",
    )
    labels = build_fullgame_targets(rules, epoch, claim_draw=False)
    assert labels.complete_games == 0
    assert labels.normal_cap_games == 1
    assert labels.epoch_truncated_games == 0
    assert labels.excluded_actions == 1
    assert not labels.targets
    resumed = OnlineActors(
        (ActorOpening("heldout-free-train-opening", rules.initial_state()),),
        config=OnlineActorConfig(games=1, max_additional_plies=1, claim_draw=False, temperature=1),
        rng=random.Random(99),
        cursor=epoch.next_actor_cursor,
    )
    resumed.rng.setstate(epoch.next_actor_rng_state)
    assert resumed.cursor() == epoch.next_actor_cursor

    # Game 0 ends on its first action; its replacement game remains live at
    # the two-step boundary and must be explicitly excluded as unknown.
    position = rules.initial_state("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    config = OnlineActorConfig(games=1, max_additional_plies=2, claim_draw=False, temperature=1)
    actors = OnlineActors(
        (ActorOpening("mate-opening", position),), config=config, rng=random.Random(3)
    )
    calls = 0

    def mate_then_continue(states, legal):
        nonlocal calls
        calls += 1
        board = rules.inspect(states[0])
        if calls == 1:
            action = move_to_action(board, board.parse_uci("f7g7"))
        else:
            other = next(move for move in board.legal_moves if move.uci() != "f7g7")
            action = move_to_action(board, other)
        probabilities = []
        for row in legal:
            index = row.index(action) if action in row else 0
            probs = [0.0] * len(row)
            probs[index] = 1.0
            probabilities.append(tuple(probs))
        uniform = tuple(tuple(1 / len(row) for _ in row) for row in legal)
        return EpochPolicyOutput(
            tuple(probabilities),
            tuple((0.3, 0.4, 0.3) for _ in legal),
            uniform,
            tuple((0.2, 0.5, 0.3) for _ in legal),
        )

    epoch = collect_policy_epoch(
        actors, steps=2, infer=mate_then_continue, model_digest=lambda: "frozen"
    )
    labels = build_fullgame_targets(rules, epoch, claim_draw=False)
    assert labels.complete_games == 1
    assert labels.epoch_truncated_games == 1
    assert labels.excluded_actions == 1
    assert len(epoch.truncations) == 1
    assert epoch.truncations[0]["moves"] == [next(
        row.transition.action.uci for row in epoch.actions
        if row.transition.game_index == epoch.truncations[0]["game_index"]
    )]


def test_epoch_rejects_model_change_and_resume_cursor_preserves_next_actor_rng():
    rules, config, actors = setup_actor(cap=1)
    digests = iter(("first", "changed"))
    with pytest.raises(ValueError, match="model changed"):
        collect_policy_epoch(
            actors,
            steps=1,
            infer=lambda states, legal: EpochPolicyOutput(
                tuple(tuple(1 / len(row) for _ in row) for row in legal),
                ((0.3, 0.4, 0.3),),
                tuple(tuple(1 / len(row) for _ in row) for row in legal),
                ((0.2, 0.5, 0.3),),
            ),
            model_digest=lambda: next(digests),
        )

    _, _, actors = setup_actor(cap=4)
    support = legal_action_indices(actors.rules.inspect(actors.states[0]))
    actors.step((tuple(1 / len(support) for _ in support),))
    with pytest.raises(ValueError, match="fresh opening roots"):
        collect_policy_epoch(
            actors,
            steps=4,
            infer=lambda states, legal: EpochPolicyOutput(
                tuple(tuple(1 / len(row) for _ in row) for row in legal),
                ((0.3, 0.4, 0.3),),
                tuple(tuple(1 / len(row) for _ in row) for row in legal),
                ((0.2, 0.5, 0.3),),
            ),
            model_digest=lambda: "frozen",
        )

    _, _, actors = setup_actor(cap=1)
    epoch = collect_policy_epoch(
        actors,
        steps=1,
        infer=lambda states, legal: EpochPolicyOutput(
            tuple(tuple(1 / len(row) for _ in row) for row in legal),
            ((0.3, 0.4, 0.3),),
            tuple(tuple(1 / len(row) for _ in row) for row in legal),
            ((0.2, 0.5, 0.3),),
        ),
        model_digest=lambda: "frozen",
    )
    saved_rng = epoch.next_actor_rng_state
    resumed = OnlineActors(
        (ActorOpening("heldout-free-train-opening", rules.initial_state()),),
        config=config,
        rng=random.Random(0),
        cursor=epoch.next_actor_cursor,
    )
    resumed.rng.setstate(saved_rng)
    legal = tuple(
        tuple(legal_action_indices(resumed.rules.inspect(state))) for state in resumed.states
    )
    probs = tuple(tuple(1 / len(row) for _ in row) for row in legal)
    first = resumed.step(probs)
    resumed2 = OnlineActors(
        (ActorOpening("heldout-free-train-opening", rules.initial_state()),),
        config=config,
        rng=random.Random(0),
        cursor=epoch.next_actor_cursor,
    )
    resumed2.rng.setstate(saved_rng)
    legal2 = tuple(
        tuple(legal_action_indices(resumed2.rules.inspect(state))) for state in resumed2.states
    )
    probs2 = tuple(tuple(1 / len(row) for _ in row) for row in legal2)
    second = resumed2.step(probs2)
    assert first == second
    assert resumed.cursor() == resumed2.cursor()


def test_fullgame_ppo_clips_action_ratio_and_reports_both_kl_guards():
    config = FullGamePPOConfig(
        policy_clip=0.1,
        behavior_kl_stop=0.01,
        policy_anchor_weight=0.2,
        value_weight=1.0,
        value_anchor_weight=0.1,
    )
    logits = torch.tensor([[2.0, 0.0, -1.0]], requires_grad=True)
    values = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
    legal = torch.tensor([[True, True, False]])
    behavior = torch.tensor([[0.5, 0.5, 0.0]])
    base = torch.tensor([[0.5, 0.5, 0.0]])
    target = torch.tensor([[1.0, 0.0, 0.0]])
    base_wdl = torch.tensor([[0.3, 0.4, 0.3]])
    loss = fullgame_ppo_loss(
        logits,
        values,
        legal_masks=legal,
        actions=torch.tensor([0]),
        old_action_probabilities=torch.tensor([0.5]),
        advantages=torch.tensor([0.2]),
        behavior_policy=behavior,
        base_policy=base,
        terminal_wdl=target,
        base_wdl=base_wdl,
        config=config,
    )
    assert loss.policy.item() == pytest.approx(-0.22)
    assert loss.behavior_policy_kl.item() > config.behavior_kl_stop
    assert loss.base_policy_kl.item() == pytest.approx(loss.behavior_policy_kl.item())
    loss.total.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.isfinite(values.grad).all()
    with pytest.raises(ValueError, match="old behavior mass"):
        fullgame_ppo_loss(
            logits,
            values,
            legal_masks=legal,
            actions=torch.tensor([0]),
            old_action_probabilities=torch.tensor([0.4]),
            advantages=torch.tensor([0.2]),
            behavior_policy=behavior,
            base_policy=base,
            terminal_wdl=target,
            base_wdl=base_wdl,
            config=config,
        )


@pytest.mark.parametrize("passes", [1, 2])
def test_one_closed_epoch_performs_a_real_terminal_policy_and_wdl_update(passes):
    rules, _, actors = setup_actor()
    epoch = collect_policy_epoch(
        actors,
        steps=4,
        infer=lambda states, legal: fool_mate_policy(rules, states, legal),
        model_digest=lambda: "frozen-e8",
    )
    targets = build_fullgame_targets(rules, epoch, claim_draw=False)
    network = TorchChessNetwork(
        NetworkConfig(
            trunk_channels=4,
            residual_blocks=1,
            policy_channels=2,
            value_channels=2,
            value_hidden=4,
        ),
        architecture="pairwise",
    )
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-4)
    result = train_fullgame_policy_epoch(
        network,
        targets=targets,
        optimizer=optimizer,
        rules=rules,
        encoder=BoardEncoder(rules),
        seed=10,
        device="cpu",
        objective=FullGamePPOConfig(
            policy_clip=0.1,
            behavior_kl_stop=100.0,
            policy_anchor_weight=0.01,
            value_weight=1.0,
            value_anchor_weight=0.01,
        ),
        schedule=FullGamePPOTrainConfig(
            minibatch_size=4,
            passes=passes,
            max_gradient_norm=1.0,
        ),
    )
    assert result["schema"] == "fullgame-terminal-ppo-train-v1"
    assert result["trained_transitions"] == 4
    assert result["accepted_passes"] == passes
    assert result["optimizer_steps_attempted"] == passes
    assert result["sampler_rng_state_before_epoch"] == random.Random(10).getstate()
    assert result["sampler_rng_state"] != result["sampler_rng_state_before_epoch"]
    assert result["final_behavior_kl"] >= 0
    assert result["last_minibatch_losses"]["terminal_wdl"] > 0


def test_rejected_pass_rolls_back_parameters_optimizer_all_rng_and_sampler(monkeypatch):
    rules, _, actors = setup_actor()
    epoch = collect_policy_epoch(
        actors,
        steps=4,
        infer=lambda states, legal: fool_mate_policy(rules, states, legal),
        model_digest=lambda: "frozen-e8",
    )
    targets = build_fullgame_targets(rules, epoch, claim_draw=False)
    network = TorchChessNetwork(
        NetworkConfig(
            trunk_channels=4,
            residual_blocks=1,
            policy_channels=2,
            value_channels=2,
            value_hidden=4,
        ),
        architecture="pairwise",
    )
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
    model_before = copy.deepcopy(network.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    random.seed(1201)
    np.random.seed(1202)
    torch.manual_seed(1203)
    kl_calls = 0
    pass_start_rng = {}
    guard_calls = 0

    def controlled_kl(*args, **kwargs):
        nonlocal kl_calls
        kl_calls += 1
        # Collection matches at entry, but the single optimizer pass is
        # rejected by its post-pass whole-buffer KL measurement.
        return (0.0, 0.0) if kl_calls in (1, 3) else (1.0, 0.0)

    monkeypatch.setattr(fullgame_ppo, "_whole_dataset_kls", controlled_kl)

    def noisy_guard():
        nonlocal guard_calls
        guard_calls += 1
        if guard_calls == 3:
            pass_start_rng["python"] = random.getstate()
            pass_start_rng["numpy"] = copy.deepcopy(np.random.get_state())
            pass_start_rng["torch"] = torch.get_rng_state().clone()
            pass_start_rng["cuda"] = (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            )
        random.random()
        np.random.random()
        torch.rand(1)
        if torch.cuda.is_available():
            torch.rand(1, device="cuda")

    result = train_fullgame_policy_epoch(
        network,
        targets=targets,
        optimizer=optimizer,
        rules=rules,
        encoder=BoardEncoder(rules),
        seed=101,
        device="cpu",
        objective=FullGamePPOConfig(
            policy_clip=0.1,
            behavior_kl_stop=0.1,
            policy_anchor_weight=0.01,
            value_weight=1.0,
            value_anchor_weight=0.01,
        ),
        schedule=FullGamePPOTrainConfig(
            minibatch_size=4,
            passes=1,
            max_gradient_norm=1.0,
        ),
        guard=noisy_guard,
    )
    assert result["accepted_passes"] == 0
    assert result["rejected_passes"] == 1
    assert result["optimizer_steps_attempted"] == 1
    assert result["sampler_rng_state_before_epoch"] == random.Random(101).getstate()
    assert result["optimizer_steps_committed"] == 0
    assert result["optimizer_steps_discarded"] == 1
    assert result["sampler_rng_state"] == result["sampler_rng_state_before_epoch"]
    assert all(torch.equal(model_before[key], value) for key, value in network.state_dict().items())
    assert optimizer.state_dict() == optimizer_before
    assert random.getstate() == pass_start_rng["python"]
    restored_numpy = np.random.get_state()
    numpy_rng_before = pass_start_rng["numpy"]
    assert restored_numpy[0] == numpy_rng_before[0]
    assert np.array_equal(restored_numpy[1], numpy_rng_before[1])
    assert restored_numpy[2:] == numpy_rng_before[2:]
    assert torch.equal(torch.get_rng_state(), pass_start_rng["torch"])
    if pass_start_rng["cuda"] is not None:
        assert all(
            torch.equal(actual, expected)
            for actual, expected in zip(
                torch.cuda.get_rng_state_all(), pass_start_rng["cuda"], strict=True
            )
        )


def test_torch_cuda_cpu_batch_adapter_preserves_exact_legal_support():
    rules = PythonChessRules()
    network = TorchChessNetwork(
        NetworkConfig(
            trunk_channels=4,
            residual_blocks=1,
            policy_channels=2,
            value_channels=2,
            value_hidden=4,
        ),
        architecture="pairwise",
    )
    base = copy.deepcopy(network)
    digest = torch_model_digest(network)
    infer = make_torch_epoch_inference(network, base, BoardEncoder(rules), device="cpu")
    state = rules.initial_state()
    support = tuple(legal_action_indices(rules.inspect(state)))
    output = infer((state,), (support,))
    assert len(output.policy[0]) == len(support)
    assert sum(output.policy[0]) == pytest.approx(1.0, abs=1e-6)
    assert sum(output.base_policy[0]) == pytest.approx(1.0, abs=1e-6)
    assert sum(output.wdl[0]) == pytest.approx(1.0, abs=1e-6)
    assert sum(output.base_wdl[0]) == pytest.approx(1.0, abs=1e-6)
    assert torch_model_digest(network) == digest
