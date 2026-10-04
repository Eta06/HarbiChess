import copy
import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import chess
import pytest
import torch
from test_torch_online_checkpoint import equal_tree, fixture

from harbichess.backends.torch_network import save_weights
from harbichess.chess.actions import legal_action_indices, move_to_action
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove, ChessState
from harbichess.selfplay.online_actor import OnlineActorConfig
from harbichess.training.online_objective import OnlineObjectiveConfig
from harbichess.training.torch_online_learner import (
    TorchOnlineConfig,
    TorchOnlineLearner,
    read_online_train_book,
)

SOURCE = "a" * 40


def config():
    return TorchOnlineConfig(
        seed=719,
        actors=OnlineActorConfig(4, 3, True, 1.0),
        objective=OnlineObjectiveConfig(0.03, 0.2, 0.02),
        learning_rate=1e-4,
        weight_decay=1e-4,
        max_gradient_norm=5.0,
        ema_decay=0.9,
        importance_maximum=1.0,
    )


def inputs(directory):
    return {"initial_weights": directory / "initial.safetensors", "book": directory / "book.json"}


def make_inputs(directory):
    save_weights(directory / "initial.safetensors", fixture().online)
    rules = PythonChessRules()
    state = rules.initial_state()
    rows = []
    for name, moves in [("initial", []), ("e4", ["e2e4"])]:
        current = state
        for uci in moves:
            current = rules.apply(current, ChessMove(uci))
        rows.append(
            {
                "source_game": name,
                "root_ply": len(moves),
                "opening": {"moves": moves, "fen": rules.view(current).fen},
            }
        )
    (directory / "book.json").write_text(json.dumps({"schema": 1, "splits": {"train": rows}}))


def fresh(directory):
    return TorchOnlineLearner.fresh(
        config=config(), input_paths=inputs(directory), source_commit=SOURCE
    )


def test_actual_legal_selfplay_targets_updates_fixed_base_and_ema_tracking(tmp_path):
    make_inputs(tmp_path)
    learner = fresh(tmp_path)
    initial = {key: value.clone() for key, value in learner.online.state_dict().items()}
    base = copy.deepcopy(learner.base.state_dict())
    previous_ema = copy.deepcopy(learner.ema.state_dict())
    rules = PythonChessRules()
    records = []
    for _ in range(3):
        record = learner.train_update()
        records.append(record)
        for key, actual in learner.ema.state_dict().items():
            expected = previous_ema[key] * 0.9 + learner.online.state_dict()[key] * 0.1
            torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
        previous_ema = copy.deepcopy(learner.ema.state_dict())
        assert all(math.isfinite(value) for value in record["loss"].values())
        for sample in record["samples"]:
            pre = ChessState(
                sample["root_fen"], tuple(ChessMove(move) for move in sample["pre_moves"])
            )
            assert rules.outcome(pre, claim_draw=True) is None
            post = rules.apply(pre, ChessMove(sample["action"]))
            assert sample["legal_actions"] == list(legal_action_indices(rules.inspect(pre)))
            outcome = rules.outcome(post, claim_draw=True)
            if outcome is None:
                assert sample["target_source"] == "ema-bootstrap"
                assert sample["terminal_result"] is None
                assert sample["mover_target_wdl"] == sample["ema_post_wdl"][::-1]
            else:
                assert sample["target_source"] == "observed-terminal"
                assert sample["terminal_result"] == outcome.result.value
                assert sample["ema_post_wdl"] is None
            assert sample["importance"] == pytest.approx(1.0)
    assert learner.update == learner.actors.steps == 3
    assert sum(record["transitions"] for record in records) == 12
    assert any(sample["rollout_cutoff"] for sample in records[-1]["samples"])
    assert any(
        not torch.equal(value, learner.online.state_dict()[key]) for key, value in initial.items()
    )
    for key, value in base.items():
        assert torch.equal(value, learner.base.state_dict()[key])
    for key, value in initial.items():
        if key.startswith("material_value_linear."):
            assert torch.equal(value, learner.online.state_dict()[key])


def test_fresh_process_next_real_rollouts_losses_all_models_optimizer_and_rng_exact(tmp_path):
    make_inputs(tmp_path)
    learner = fresh(tmp_path)
    for _ in range(2):
        learner.train_update()
    learner.checkpoint(tmp_path / "step2")
    expected = [learner.train_update() for _ in range(4)]
    learner.checkpoint(tmp_path / "expected-step6")
    code = """
from pathlib import Path
import json, sys
from test_torch_online_learner import config, inputs, SOURCE
from harbichess.training.torch_online_learner import TorchOnlineLearner
directory=Path(sys.argv[1])
learner=TorchOnlineLearner.resume(directory/'step2',config=config(),input_paths=inputs(directory),source_commit=SOURCE)
records=[learner.train_update() for _ in range(4)]
(directory/'restored-records.json').write_text(json.dumps(records,sort_keys=True))
learner.checkpoint(directory/'restored-step6')
"""
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            (str(Path(__file__).parent), str(Path(__file__).parents[1] / "src"))
        ),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    completed = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    actual = json.loads((tmp_path / "restored-records.json").read_text())
    assert actual == expected
    for name in ("model.safetensors", "base.safetensors", "ema.safetensors", "actor.json"):
        assert (tmp_path / "expected-step6" / name).read_bytes() == (
            tmp_path / "restored-step6" / name
        ).read_bytes()
    equal_tree(
        torch.load(tmp_path / "expected-step6/training.pt", weights_only=True),
        torch.load(tmp_path / "restored-step6/training.pt", weights_only=True),
    )


@pytest.mark.parametrize("kind", ["fen", "heldout", "duplicate", "empty"])
def test_book_provenance_rejects_wrong_history_or_source_overlap(tmp_path, kind):
    make_inputs(tmp_path)
    path = tmp_path / "book.json"
    book = json.loads(path.read_text())
    if kind == "fen":
        book["splits"]["train"][1]["opening"]["fen"] = chess.STARTING_FEN
    elif kind == "heldout":
        book["splits"]["validation"] = book["splits"]["train"][:1]
    elif kind == "duplicate":
        book["splits"]["train"][1]["source_game"] = "initial"
    else:
        book["splits"]["train"] = []
    path.write_text(json.dumps(book))
    with pytest.raises(ValueError):
        read_online_train_book(path)


def test_reject_changed_optimizer_inputs_and_resume_config(tmp_path):
    make_inputs(tmp_path)
    learner = fresh(tmp_path)
    learner.checkpoint(tmp_path / "initial-checkpoint")
    with pytest.raises(ValueError, match="mismatch"):
        TorchOnlineLearner.resume(
            tmp_path / "initial-checkpoint",
            config=replace(config(), ema_decay=0.8),
            input_paths=inputs(tmp_path),
            source_commit=SOURCE,
        )
    learner.optimizer.param_groups[0]["lr"] *= 2
    with pytest.raises(ValueError, match="optimizer settings changed"):
        learner.train_update()
    assert learner.update == learner.actors.steps == 0
    learner.optimizer.param_groups[0]["lr"] = config().learning_rate
    (tmp_path / "book.json").write_text("changed")
    with pytest.raises(ValueError, match="immutable inputs changed"):
        learner.checkpoint(tmp_path / "changed-input")
    assert not (tmp_path / "changed-input").exists()


def test_real_terminal_at_cutoff_needs_no_ema_forward_and_keeps_win_label(tmp_path, monkeypatch):
    make_inputs(tmp_path)
    fen = "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"
    book = {
        "schema": 1,
        "splits": {
            "train": [
                {
                    "source_game": "mate-probe",
                    "root_ply": 0,
                    "opening": {"root_fen": fen, "moves": [], "fen": fen},
                }
            ]
        },
    }
    (tmp_path / "book.json").write_text(json.dumps(book))
    learner = TorchOnlineLearner.fresh(
        config=replace(config(), actors=OnlineActorConfig(1, 1, True, 1.0)),
        input_paths=inputs(tmp_path),
        source_commit=SOURCE,
    )
    original = learner.online.masked_policy_value
    chosen = move_to_action(chess.Board(fen), chess.Move.from_uci("f7g7"))

    def force_mating_action(inputs, actions):
        policy, value = original(inputs, actions)
        # A unit-test actor override, not a trained model improvement.
        forced = torch.where(actions == chosen, 0.0, -1e9) + policy * 0
        return forced, value

    def reject_terminal_ema(*args):
        raise AssertionError("terminal child must not require an EMA bootstrap forward")

    monkeypatch.setattr(learner.online, "masked_policy_value", force_mating_action)
    monkeypatch.setattr(learner.ema, "masked_policy_value", reject_terminal_ema)
    (sample,) = learner.train_update()["samples"]
    assert sample["action"] == "f7g7"
    assert sample["rollout_cutoff"] is True
    assert sample["target_source"] == "observed-terminal"
    assert sample["terminal_result"] == "1-0"
    assert sample["mover_target_wdl"] == [1.0, 0.0, 0.0]
    assert sample["ema_post_wdl"] is None
