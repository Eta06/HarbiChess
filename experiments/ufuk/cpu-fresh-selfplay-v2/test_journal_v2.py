import copy
import sys
import time
from pathlib import Path

import chess
import journal_v2 as journal
import numpy as np
import pytest
import torch
from objective import additive_own_mc_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from anchor_value import FrozenWDLAnchor

ROOT = {
    "source_id": "tiny-legal-mate-root",
    "root_fen": chess.STARTING_FEN,
    "prefix": ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4"],
}


def config(seed=1):
    return {
        "nodes": 512,
        "qdepth": 2,
        "max_depth": 8,
        "exploration": 0.05,
        "total_ply_cap": 400,
        "max_actions": 2,
        "actors": 1,
        "seed": seed,
        "epoch_id": "tiny-v2-anchor-test",
        "original_deadline_epoch": time.time() + 60,
        "model_sha256": "0" * 64,
        "source_commit": "0" * 40,
        "search_helper_sha256": "1" * 64,
        "value_helper_sha256": "2" * 64,
        "producer_sha256": journal.sha(Path(__file__).with_name("journal_v2.py")),
        "runner_sha256": "3" * 64,
        "evaluator_identity": "synthetic-scripted-NOT-NN-qualified",
        "anchor_model_sha256": "4" * 64,
        "anchor_helper_sha256": "5" * 64,
        "anchor_target": "frozen-e8-wdl-probabilities-mover-perspective-v1",
        "roots": [ROOT],
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
                "value": 123.0,
            },
        )()


def test_v2_anchor_is_fullhistory_preaction_and_separate_from_search_score(tmp_path):
    seen = []

    def anchor(board):
        seen.append((board.ply(), tuple(m.uci() for m in board.move_stack)))
        return (0.6, 0.25, 0.15) if board.turn else (0.2, 0.3, 0.5)

    for seed in range(100):
        cfg = config(seed)
        actor = journal.Actor(cfg, lambda: ScriptedMate(), anchor)
        actor.advance(2)
        if actor.state["games"]:
            break
    assert actor.state["games"][0]["result"] == "1-0"
    assert [x[0] for x in seen] == [5, 6]
    assert seen[0][1] == tuple(ROOT["prefix"])
    packets = journal.replay(actor.state, cfg)
    assert packets[0][3] == "1-0"
    assert packets[0][2][0][2] == (0.2, 0.3, 0.5)
    assert actor.state["games"][0]["moves"][0]["search_value"] == 123.0
    assert actor.state["games"][0]["moves"][0]["e8_anchor_wdl"] == [0.2, 0.3, 0.5]

    path = tmp_path / "journal-v2.gz"
    journal.save(path, actor.state)
    x, y, anchors, groups, provenance = journal.shrink840(
        journal.read(path), cfg, lambda board: np.zeros(840, dtype=np.float32)
    )
    assert x.shape == (2, 840) and y.tolist() == [2, 0]
    assert anchors.shape == (2, 3)
    assert np.allclose(anchors, [[0.2, 0.3, 0.5], [0.6, 0.25, 0.15]], atol=1e-7)
    assert groups == [[0, 1]] and provenance[0]["source_id"] == ROOT["source_id"]
    feature_helper = tmp_path / "tiny_features.py"
    feature_helper.write_text(
        "import numpy as np\n"
        "def invariants(board):\n"
        "    return np.zeros(840, dtype=np.float32)\n"
    )
    converted = journal.convert_verified(
        path, journal.sha(path), cfg, feature_helper, journal.sha(feature_helper)
    )
    cx, cy, ca, cg, cp, receipt = converted
    assert np.array_equal(cx, x) and np.array_equal(cy, y)
    assert np.array_equal(ca, anchors) and cg == groups and cp == provenance
    assert receipt["schema"] == "fresh-qsearch-to-shrink840-handoff-v2"
    assert receipt["anchor_model_sha256"] == cfg["anchor_model_sha256"]
    assert receipt["anchor_helper_sha256"] == cfg["anchor_helper_sha256"]
    with pytest.raises(ValueError, match="source journal SHA mismatch"):
        journal.convert_verified(path, "0" * 64, cfg, feature_helper, journal.sha(feature_helper))


@pytest.mark.parametrize("bad", [[0, 0, 0], [float("nan"), 0, 1], [-0.1, 0.5, 0.6], [0.4, 0.4]])
def test_invalid_wdl_anchor_rejected_at_capture_and_replay(bad):
    cfg = config()
    actor = journal.Actor(cfg, lambda: ScriptedMate(), lambda board: bad)
    with pytest.raises(ValueError, match="invalid frozen e8 WDL anchor"):
        actor.advance(1)

    cfg = config()
    actor = journal.Actor(cfg, lambda: ScriptedMate(), lambda board: (0.3, 0.4, 0.3))
    for seed in range(100):
        cfg["seed"] = seed
        actor = journal.Actor(cfg, lambda: ScriptedMate(), lambda board: (0.3, 0.4, 0.3))
        actor.advance(2)
        if actor.state["games"]:
            break
    corrupted = copy.deepcopy(actor.state)
    corrupted["games"][0]["moves"][0]["e8_anchor_wdl"] = bad
    with pytest.raises(ValueError, match="invalid/missing frozen e8 anchor"):
        journal.replay(corrupted, cfg)


def test_v1_and_v2_are_separate_decoder_generations():
    old = Path(__file__).with_name("journal_v1.py").read_text()
    assert 'SCHEMA = "fresh-qsearch-selfplay-journal-v1"' in old
    assert journal.SCHEMA == "fresh-qsearch-selfplay-journal-v2"
    cfg = config()
    assert cfg["anchor_target"] == "frozen-e8-wdl-probabilities-mover-perspective-v1"


def test_immutable_config_and_whole_pause_resume_v2_journal(tmp_path):
    cfg = config(4)

    def anchors(board):
        return (0.4, 0.2, 0.4)
    whole = journal.Actor(cfg, lambda: ScriptedMate(), anchors)
    whole.advance(2)
    whole_path = tmp_path / "whole.gz"
    journal.save(whole_path, whole.state)

    paused = journal.Actor(cfg, lambda: ScriptedMate(), anchors)
    paused.advance(1)
    pause_path = tmp_path / "pause.gz"
    journal.save(pause_path, paused.state)
    resumed = journal.Actor(cfg, lambda: ScriptedMate(), anchors, journal.read(pause_path))
    resumed.advance(2)
    resumed_path = tmp_path / "resumed.gz"
    journal.save(resumed_path, resumed.state)
    assert whole_path.read_bytes() == resumed_path.read_bytes()

    changed = journal.Actor(cfg, lambda: ScriptedMate(), anchors)
    changed.state["config"]["seed"] += 1
    with pytest.raises(ValueError, match="immutable v2 actor config"):
        changed.advance(1)


def test_additive_objective_zero_residual_matches_base_and_uses_only_own_returns():
    base = torch.tensor([[0.7, 0.2, 0.1], [0.2, 0.3, 0.5]], dtype=torch.float32)
    residual = torch.zeros((2, 3), requires_grad=True)
    labels = torch.tensor([0, 2])
    total, own_mc, anchor_kl, logits = additive_own_mc_loss(base, residual, labels, 1.0)
    assert torch.allclose(torch.softmax(logits, 1), base, atol=1e-7)
    assert own_mc.item() > 0 and torch.allclose(anchor_kl, torch.zeros_like(anchor_kl), atol=1e-7)
    total.backward()
    assert residual.grad is not None and torch.isfinite(residual.grad).all()


def test_additive_objective_rejects_bad_frozen_anchor():
    with pytest.raises(ValueError, match="normalized probabilities"):
        additive_own_mc_loss(
            torch.tensor([[0.5, 0.5, 0.5]]), torch.zeros((1, 3)), torch.tensor([0])
        )


def test_anchor_adapter_softmaxes_pinned_helper_fullhistory_logits():
    seen = []

    class Value:
        def logits(self, board):
            seen.append(tuple(move.uci() for move in board.move_stack))
            return torch.tensor([[2.0, 0.0, -1.0]])

    class ValueHelper:
        @staticmethod
        def NeuralValue(weights):
            assert weights == "synthetic-weights"
            return Value()

    board = journal.board_for(ROOT)
    probabilities = FrozenWDLAnchor("synthetic-weights", ValueHelper)(board)
    assert seen == [tuple(ROOT["prefix"])]
    assert len(probabilities) == 3 and sum(probabilities) == pytest.approx(1.0)
    assert probabilities[0] > probabilities[1] > probabilities[2]
