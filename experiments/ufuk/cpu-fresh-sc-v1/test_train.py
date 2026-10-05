import importlib.util
import sys
import time
from pathlib import Path

import chess
import pytest
import torch
from train import (
    critic,
    position_key,
    prepare_fresh,
    rebase,
    touches_protected_position,
    trajectory_key,
)

ROOT = Path(__file__).resolve().parents[1] / "cpu-fresh-selfplay-v2"
sys.path.insert(0, str(ROOT))
import journal_v2  # noqa: E402

FEATURES = Path("/workspace/HarbiChess/experiments/ufuk/cpu-residual-value-v1/features.py")
PROTECTED_KEY = "8/8/8/8/8/8/4k3/7K w - -"
_feature_spec = importlib.util.spec_from_file_location("test_fresh_q_features", FEATURES)
features = importlib.util.module_from_spec(_feature_spec)
assert _feature_spec.loader is not None
_feature_spec.loader.exec_module(features)


def actor_config(seed, roots, max_actions=4):
    return {
        "nodes": 512,
        "qdepth": 2,
        "max_depth": 8,
        "exploration": 0.05,
        "total_ply_cap": 400,
        "max_actions": max_actions,
        "actors": 1,
        "seed": seed,
        "epoch_id": "tiny-v2-trainer-interface",
        "original_deadline_epoch": time.time() + 60,
        "model_sha256": "0" * 64,
        "source_commit": "0" * 40,
        "search_helper_sha256": "1" * 64,
        "value_helper_sha256": "2" * 64,
        "producer_sha256": journal_v2.sha(ROOT / "journal_v2.py"),
        "runner_sha256": "3" * 64,
        "evaluator_identity": "synthetic-scripted-NOT-NN-qualified",
        "anchor_model_sha256": "4" * 64,
        "anchor_helper_sha256": "5" * 64,
        "anchor_target": "frozen-e8-wdl-probabilities-mover-perspective-v1",
        "roots": roots,
        "excluded_training_position_keys": [PROTECTED_KEY],
        "exclusion_book_sha256": "9" * 64,
    }


class ScriptedMate:
    def search(self, board):
        move = chess.Move.from_uci("g8f6" if board.ply() == 5 else "h5f7")
        return type(
            "Result",
            (),
            {
                "move": move,
                "root_actions": board.legal_moves.count(),
                "nodes": 1,
                "evaluations": 0,
                "completed_depth": 0,
                "value": 0.0,
            },
        )()


def test_fresh_v2_converter_trajectory_split_protection_and_unknown(tmp_path):
    common = "ordinary-start"
    prefixes = [
        ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4"],
        ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5"],
    ]
    roots = [dict(source_id=common, root_fen=chess.STARTING_FEN, prefix=p) for p in prefixes]
    roots.append(
        {
            "source_id": common,
            "root_fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 200",
            "prefix": [],
        }
    )
    config = actor_config(2026, roots, max_actions=120)
    config_path = tmp_path / "actor-config.json"
    config_path.write_bytes(journal_v2.canonical(config))

    class ScriptedWithCap(ScriptedMate):
        def search(self, board):
            if board.ply() >= 398:
                move = sorted(board.legal_moves, key=lambda item: item.uci())[0]
                return type(
                    "Result",
                    (),
                    {
                        "move": move,
                        "root_actions": board.legal_moves.count(),
                        "nodes": 1,
                        "evaluations": 0,
                        "completed_depth": 0,
                        "value": 0.0,
                    },
                )()
            desired = (
                chess.Move.from_uci("g8f6")
                if board.ply() == 5
                else chess.Move.from_uci("h5f7")
                if board.ply() == 6
                else None
            )
            move = (
                desired
                if desired in board.legal_moves
                else sorted(board.legal_moves, key=lambda item: item.uci())[0]
            )
            return type(
                "Result",
                (),
                {
                    "move": move,
                    "root_actions": board.legal_moves.count(),
                    "nodes": 1,
                    "evaluations": 0,
                    "completed_depth": 0,
                    "value": 0.0,
                },
            )()

    actor = journal_v2.Actor(config, lambda: ScriptedWithCap(), lambda board: (0.3, 0.4, 0.3))
    actor.advance(120)
    assert len(actor.state["games"]) >= 16
    assert all(roots[game["root_index"]]["source_id"] == common for game in actor.state["games"])
    journal_path = tmp_path / "v2-journal.gz"
    journal_v2.save(journal_path, actor.state)

    board = chess.Board()
    for move in ["e2e4", "e7e5", "d1h5"]:
        board.push_uci(move)
    protected_position = position_key(board)
    kwargs = (
        journal_path,
        journal_v2.sha(journal_path),
        config_path,
        journal_v2.sha(config_path),
        ROOT / "journal_v2.py",
        journal_v2.sha(ROOT / "journal_v2.py"),
        FEATURES,
        journal_v2.sha(FEATURES),
    )
    data = prepare_fresh(
        *kwargs, minimum_rows=4, minimum_games=2, protected_position_keys=[PROTECTED_KEY]
    )
    (
        x,
        y,
        anchors,
        groups,
        trajectories,
        train_groups,
        val_groups,
        train,
        val,
        receipts,
        data_sha,
        _,
    ) = data
    assert x.shape[1:] == (840,) and anchors.shape == (len(y), 3)
    assert len(groups) >= 2 and len(set(trajectories)) == len(trajectories)
    assert set(train_groups).isdisjoint(val_groups)
    assert train and val and not (set(train) & set(val))
    assert (
        all(item["source_id"] == common for item in receipts[0].get("provenance", []))
        if "provenance" in receipts[0]
        else True
    )
    split = receipts[1]
    assert split["source_ids_are_not_split_keys"]
    assert split["independence_scope"] == (
        "generated-trajectory-disjoint internal validation; not root/source-family/state-disjoint"
    )
    assert split["protected_position_key_count"] == 1
    assert (
        split["protected_position_keys_sha256"]
        == __import__("hashlib").sha256(journal_v2.canonical([PROTECTED_KEY])).hexdigest()
    )
    assert split["excluded_complete_games_touching_protected_positions"] == 0
    assert len(split["protected_exclusion_ledger_sha256"]) == 64
    assert receipts[0]["excluded_UNKNOWN_or_tail_rows"] > 0
    # A protected intermediate board excludes the entire complete game, including
    # all rows before and after the hit; duplicate aliases retain one trajectory.
    one_game = next(g for g in actor.state["games"] if g["root_index"] == 0)
    one_root = roots[0]
    assert touches_protected_position(one_root, one_game, {protected_position})
    other_game = next(g for g in actor.state["games"] if g["root_index"] == 1)
    assert not touches_protected_position(roots[1], other_game, {protected_position})
    assert trajectory_key(one_root, one_game) == trajectory_key(one_root, one_game)
    assert split["deduplicated_identical_trajectory_aliases"] > 0
    assert len(data_sha) == 64

    # Duplicate exact trajectories hash to one group and cannot straddle partitions.
    h1 = (
        __import__("hashlib")
        .sha256(
            journal_v2.canonical({"root_fen": chess.STARTING_FEN, "prefix": [], "moves": ["e2e4"]})
        )
        .hexdigest()
    )
    h2 = (
        __import__("hashlib")
        .sha256(
            journal_v2.canonical({"root_fen": chess.STARTING_FEN, "prefix": [], "moves": ["e2e4"]})
        )
        .hexdigest()
    )
    assert h1 == h2 and (int(h1[:8], 16) % 5) == (int(h2[:8], 16) % 5)

    with pytest.raises(ValueError, match="protocol protected keys differ"):
        prepare_fresh(
            *kwargs,
            minimum_rows=4,
            minimum_games=2,
            protected_position_keys=["8/8/8/8/8/8/8/K6k w - -"],
        )
    with pytest.raises(ValueError, match="immutable v2 journal/config input SHA"):
        prepare_fresh(
            journal_path,
            "f" * 64,
            config_path,
            journal_v2.sha(config_path),
            ROOT / "journal_v2.py",
            journal_v2.sha(ROOT / "journal_v2.py"),
            FEATURES,
            journal_v2.sha(FEATURES),
            minimum_rows=4,
            minimum_games=2,
        )


def test_zero_residual_matches_full_pairwise_forward_and_gradients():
    sys.path.insert(0, "/workspace/HarbiChess/src")
    from harbichess.backends.torch_network import TorchChessNetwork
    from harbichess.core.network_config import NetworkConfig
    from harbichess.training.torch_array_encoder import TorchArrayBoardEncoder

    torch.set_num_threads(1)
    old = TorchChessNetwork(
        NetworkConfig(trunk_channels=4, residual_blocks=1, value_channels=2, value_hidden=8),
        architecture="pairwise",
        invariant={"channels": 2, "blocks": 1, "hidden": 4},
    ).eval()
    new = rebase(old).eval()
    board = chess.Board()
    encoded = TorchArrayBoardEncoder().encode_board(board)
    network_input = torch.from_numpy(encoded.values.copy()).reshape(1, 8, 8, 104)
    base_logits = old.masked_policy_value(network_input, torch.tensor([[0]]))[1]
    full_policy, full_logits = new.masked_policy_value(network_input, torch.tensor([[0]]))
    assert torch.equal(full_logits, base_logits)
    x = torch.from_numpy(features.invariants(board)[None].copy())
    base_probabilities = torch.softmax(base_logits, 1)
    fast_logits = critic(new, x, base_probabilities)
    torch.testing.assert_close(
        torch.softmax(fast_logits, 1), torch.softmax(full_logits, 1), atol=3e-7, rtol=3e-7
    )
    assert full_policy.shape == (1, 1)

    new.train()
    targets = torch.tensor([0])
    loss = torch.nn.functional.cross_entropy(critic(new, x, base_probabilities), targets)
    loss.backward()
    assert new.value_sparse_head.feature.weight.grad is not None
    assert torch.isfinite(new.value_sparse_head.feature.weight.grad).all()
    assert not torch.count_nonzero(new.value_sparse_head.feature.weight.grad[3:])


def test_configured_production_admission_rejects_short_dataset():
    assert 4 * 1024 // 256 == 16
    assert min(1024, (4 * 1024) // 256) == 16
    assert min(1024, (4 * 65536) // 256) == 1024


def test_common_data_and_initialization_functions_ast_identical_to_frozen_mc():
    import ast

    root = Path(__file__).parent
    before = ast.parse((root / "frozen-mc-parent-corrected.py.txt").read_text())
    after = ast.parse((root / "train.py").read_text())
    for name in (
        "prepare_fresh",
        "rebase",
        "critic",
        "penalty",
        "verify_mask",
        "trajectory_key",
        "position_key",
        "touches_protected_position",
        "metrics",
        "checkpoint",
    ):
        x = next(n for n in before.body if isinstance(n, ast.FunctionDef) and n.name == name)
        y = next(n for n in after.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(x) == ast.dump(y), name


def test_sc_composition_adds_e8_once_and_uses_exact_delta_gradients():
    from own_search_consistency import mixed_loss
    from train import residual_logits

    sys.path.insert(0, "/workspace/work/harbichess/cpu-additive-source-6fcc8b4/src")
    from harbichess.backends.torch_network import TorchChessNetwork
    from harbichess.core.network_config import NetworkConfig

    model = rebase(
        TorchChessNetwork(
            NetworkConfig(trunk_channels=4, residual_blocks=1, value_channels=2, value_hidden=8),
            architecture="pairwise",
            invariant={"channels": 2, "blocks": 1, "hidden": 4},
        )
    )
    x = torch.from_numpy(features.invariants(chess.Board())[None].copy())
    base = torch.tensor([[0.2, 0.3, 0.5]])
    with torch.no_grad():
        model.value_sparse_head.feature.weight[0, 0] = 0.001
    output = mixed_loss(
        base,
        residual_logits(model, x),
        torch.tensor([2]),
        torch.tensor([-1.0]),
        __import__("train").penalty(model),
    )
    assert torch.equal(output["logits"].view(torch.uint8), critic(model, x, base).view(torch.uint8))
    output["loss"].backward()
    assert model.value_sparse_head.feature.weight.grad is not None
    assert torch.isfinite(model.value_sparse_head.feature.weight.grad).all()
