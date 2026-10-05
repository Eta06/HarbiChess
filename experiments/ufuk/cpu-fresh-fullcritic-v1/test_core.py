import sys
from pathlib import Path

import chess
import numpy as np
import pytest
import torch
from core import (
    assert_frozen_bits,
    bind_common_rows,
    extract_full_history_rows,
    freeze_shared_and_policy,
    frozen_bits,
    objective,
)

REPO_SRC = Path("/workspace/HarbiChess/src")
JOURNAL_PATH = Path(__file__).resolve().parents[1] / "cpu-fresh-selfplay-v2/journal_v2.py"
FEATURE_PATH = Path("/workspace/HarbiChess/experiments/ufuk/cpu-residual-value-v1/features.py")
CONSISTENCY_PATH = Path(
    Path(__file__).resolve().parents[1] / "cpu-fresh-sc-v1/own_search_consistency.py"
)
for item in (
    str(REPO_SRC),
    str(JOURNAL_PATH.parent),
    str(FEATURE_PATH.parent),
    str(CONSISTENCY_PATH.parent),
):
    if item not in sys.path:
        sys.path.insert(0, item)

import journal_v2  # noqa: E402
import own_search_consistency  # noqa: E402
from features import invariants  # noqa: E402

from harbichess.backends.torch_network import TorchChessNetwork  # noqa: E402
from harbichess.chess.actions import legal_action_indices  # noqa: E402
from harbichess.core.network_config import NetworkConfig  # noqa: E402
from harbichess.training.torch_array_encoder import TorchArrayBoardEncoder  # noqa: E402


class MateSearch:
    def search(self, board):
        move = chess.Move.from_uci("g8f6" if board.ply() == 5 else "h5f7")
        return type(
            "Result",
            (),
            {
                "move": move,
                "root_actions": board.legal_moves.count(),
                "nodes": 2,
                "evaluations": 1,
                "completed_depth": 1,
                "value": 0.25,
            },
        )()


def config(seed=0):
    root = {
        "source_id": "same-ordinary-start-family",
        "root_fen": chess.STARTING_FEN,
        "prefix": ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4"],
    }
    return {
        "nodes": 512,
        "qdepth": 2,
        "max_depth": 8,
        "exploration": 0.05,
        "total_ply_cap": 400,
        "max_actions": 2,
        "actors": 1,
        "seed": seed,
        "epoch_id": "full-critic-synthetic",
        "original_deadline_epoch": 9999999999.0,
        "model_sha256": "0" * 64,
        "source_commit": "0" * 40,
        "search_helper_sha256": "1" * 64,
        "value_helper_sha256": "2" * 64,
        "producer_sha256": journal_v2.sha(JOURNAL_PATH),
        "runner_sha256": "3" * 64,
        "evaluator_identity": "synthetic-scripted-not-qualified",
        "anchor_model_sha256": "4" * 64,
        "anchor_helper_sha256": "5" * 64,
        "anchor_target": "frozen-e8-wdl-probabilities-mover-perspective-v1",
        "excluded_training_position_keys": ["8/8/8/8/8/8/4k3/7K w - -"],
        "exclusion_book_sha256": "6" * 64,
        "roots": [root],
    }


def known_fixture():
    torch.set_num_threads(1)
    actor_config = config()
    actor = journal_v2.Actor(
        actor_config,
        lambda: MateSearch(),
        lambda board: (0.3, 0.4, 0.3),
    )
    actor.advance(2)
    assert len(actor.state["games"]) == 1
    assert actor.state["games"][0]["result"] == "1-0"
    packets = journal_v2.replay(actor.state, actor_config)
    assert len(packets) == 1
    return actor_config, actor.state


def test_default_pairwise_value_only_gradients_and_policy_storage_freeze():
    torch.set_num_threads(1)
    model = TorchChessNetwork(NetworkConfig(), architecture="pairwise").eval()
    trainable = freeze_shared_and_policy(model)
    assert len(trainable) > 10
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 43932
    snapshot = frozen_bits(model, trainable)

    board = chess.Board()
    position = TorchArrayBoardEncoder().encode_board(board)
    inputs = torch.from_numpy(position.values.copy()).reshape(1, 8, 8, 104)
    legal = torch.tensor([legal_action_indices(board)], dtype=torch.long)
    policy_logits, baseline_logits = model.masked_policy_value(inputs, legal)
    base_wdl = torch.softmax(baseline_logits.detach(), dim=1)
    labels = torch.tensor([0], dtype=torch.long)
    search = torch.tensor([0.25], dtype=torch.float32)
    parts = objective(base_wdl, baseline_logits, labels, search)
    parts["loss"].backward()

    assert policy_logits.shape == legal.shape
    grads = {name: parameter.grad for name, parameter in model.named_parameters()}
    assert all(grads[name] is not None for name in trainable)
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if name not in trainable
    )
    assert_frozen_bits(model, snapshot)
    assert torch.isfinite(parts["loss"])


def test_fullhistory_dense_features_labels_search_score_and_mover_alignment():
    actor_config, state = known_fixture()
    packet = own_search_consistency.extract(state, actor_config, journal_v2, invariants)
    full = extract_full_history_rows(
        state,
        actor_config,
        journal_v2,
        TorchArrayBoardEncoder(),
        invariants,
        expected_known_rows=len(packet["y"]),
    )
    assert full["x104"].shape == (2, 8, 8, 104)
    assert full["x840"].tobytes() == packet["x"].tobytes()
    assert full["labels"].tobytes() == packet["y"].tobytes()
    assert full["labels"].tolist() == [2, 0]
    assert full["anchors"].tobytes() == packet["anchors"].tobytes()
    assert full["normalized_search"].tobytes() == packet["normalized_search"].tobytes()
    # Black is the root mover at the first row. Preserve the recorded +0.25 score;
    # do not negate it or substitute the played action for selected-best-move target.
    assert full["ledger"][0]["mover"] == "black"
    assert full["ledger"][0]["search_target_action"] == "g8f6"
    assert full["ledger"][0]["played_action"] == "g8f6"
    assert full["ledger"][0]["root_search_score"] == 0.25
    assert full["ledger"][0]["score_perspective"] == "pre-action-root-mover"

    common = (
        torch.from_numpy(packet["x"].copy()),
        torch.from_numpy(packet["y"].copy()),
        torch.from_numpy(packet["anchors"].copy()),
        ((0, 1),),
        ("trajectory-hash-fixture",),
        (0,),
        (),
        (0, 1),
        (),
        (),
        "a" * 64,
        actor_config,
    )
    search_binding = own_search_consistency.bind_common_data(packet, common)
    bound = bind_common_rows(common, full, search_binding)
    assert bound["train_indices"] == (0, 1)
    assert bound["validation_indices"] == ()
    assert bound["same_common_sampler_and_split"]
    wrong_search = dict(search_binding)
    wrong_search["normalized_search"] = np.zeros(2, dtype=np.float32)
    with pytest.raises(ValueError, match="search targets"):
        bind_common_rows(common, full, wrong_search)


def test_objective_weights_and_alignment_guards():
    base = torch.tensor([[0.2, 0.5, 0.3], [0.4, 0.4, 0.2]])
    logits = torch.tensor([[0.0, 0.3, -0.2], [0.5, 0.0, -0.5]], requires_grad=True)
    labels = torch.tensor([1, 0], dtype=torch.long)
    search = torch.tensor([-0.1, 0.7])
    result = objective(base, logits, labels, search)
    expected = (
        0.5 * result["own_terminal_ce"]
        + 0.5 * result["own_search_score_mse"]
        + result["fixed_e8_kl"]
    )
    torch.testing.assert_close(result["loss"], expected)
    result["loss"].backward()
    assert torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError, match="align"):
        objective(base, logits.detach(), labels, search[:1])
    with pytest.raises(ValueError, match="invalid finite"):
        objective(base, logits.detach(), labels, torch.tensor([1.1, 0.0]))


def test_synthetic_full_native_pause_restore_matches_two_step_control(tmp_path):
    import copy
    import random

    from native import capture_native, restore_native

    def make_model():
        model = TorchChessNetwork(
            NetworkConfig(trunk_channels=8, residual_blocks=1, value_channels=2, value_hidden=8),
            architecture="pairwise",
            invariant={"channels": 4, "blocks": 1, "hidden": 8},
        )
        names = freeze_shared_and_policy(model)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=2e-5,
            weight_decay=0.0,
            foreach=False,
        )
        return model, names, optimizer

    torch.set_num_threads(1)
    torch.manual_seed(20)
    random.seed(20)
    np.random.seed(20)
    seed_model, _, _ = make_model()
    baseline = copy.deepcopy(seed_model.state_dict())
    seed_torch_rng = torch.get_rng_state().clone()
    seed_python_rng = random.getstate()
    seed_numpy_rng = np.random.get_state()

    board = chess.Board()
    encoder = TorchArrayBoardEncoder()
    positions = []
    for uci in ("e2e4", "e7e5", "g1f3", "b8c6"):
        positions.append(encoder.encode_board(board).values.reshape(8, 8, 104).copy())
        board.push_uci(uci)
    x = torch.from_numpy(np.asarray(positions, dtype=np.float32))
    labels = torch.tensor([0, 2, 0, 2], dtype=torch.long)
    anchors = torch.tensor([[0.3, 0.4, 0.3]] * 4)
    search = torch.tensor([0.2, -0.2, 0.4, -0.4])
    contract = {"source": "synthetic-only", "data": "synthetic-known-rows"}

    def step(model, optimizer, sampler):
        batch = [sampler.randrange(4) for _ in range(2)]
        _, logits = model._features(x[batch])
        parts = objective(anchors[batch], logits, labels[batch], search[batch])
        optimizer.zero_grad(set_to_none=True)
        parts["loss"].backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
        optimizer.step()

    # Whole run from identical E8-like initialized values.
    control, names, control_opt = make_model()
    control.load_state_dict(baseline)
    random.setstate(seed_python_rng)
    np.random.set_state(seed_numpy_rng)
    torch.set_rng_state(seed_torch_rng)
    control_sampler = random.Random(17)
    step(control, control_opt, control_sampler)
    step(control, control_opt, control_sampler)

    # Pause after one update, round-trip a native payload, restore in a new model.
    paused, paused_names, paused_opt = make_model()
    paused.load_state_dict(baseline)
    random.setstate(seed_python_rng)
    np.random.set_state(seed_numpy_rng)
    torch.set_rng_state(seed_torch_rng)
    paused_sampler = random.Random(17)
    step(paused, paused_opt, paused_sampler)
    payload = capture_native(paused, paused_opt, paused_names, contract, 1, paused_sampler)
    checkpoint = tmp_path / "native.pt"
    torch.save(payload, checkpoint)
    assert checkpoint.stat().st_size < 2 * 1024**2
    restored_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)

    resumed, resumed_names, resumed_opt = make_model()
    resumed.load_state_dict(baseline)
    resumed_sampler = random.Random(999)
    bad_contract = dict(contract, data="changed")
    with pytest.raises(ValueError, match="contract"):
        restore_native(
            restored_payload, resumed, resumed_opt, resumed_names, bad_contract, resumed_sampler
        )
    with torch.no_grad():
        resumed.stem.weight[0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="frozen E8"):
        restore_native(
            restored_payload, resumed, resumed_opt, resumed_names, contract, resumed_sampler
        )
    with torch.no_grad():
        resumed.stem.weight.copy_(baseline["stem.weight"])
    accepted = restore_native(
        restored_payload, resumed, resumed_opt, resumed_names, contract, resumed_sampler
    )
    assert accepted == 1
    step(resumed, resumed_opt, resumed_sampler)
    for (name_a, param_a), (name_b, param_b) in zip(
        control.named_parameters(), resumed.named_parameters(), strict=True
    ):
        assert name_a == name_b
        assert torch.equal(param_a, param_b), name_a
    assert control_opt.state_dict()["state"].keys() == resumed_opt.state_dict()["state"].keys()
    for state_a, state_b in zip(
        control_opt.state.values(), resumed_opt.state.values(), strict=True
    ):
        for key in state_a:
            if isinstance(state_a[key], torch.Tensor):
                assert torch.equal(state_a[key], state_b[key])
            else:
                assert state_a[key] == state_b[key]
    assert control_sampler.getstate() == resumed_sampler.getstate()
    assert random.getstate() == restored_payload["python_rng"]
    assert np.array_equal(np.random.get_state()[1], restored_payload["numpy_rng"]["keys"].numpy())
    assert torch.equal(torch.get_rng_state(), restored_payload["torch_cpu_rng"])
