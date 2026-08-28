from __future__ import annotations

import random
from pathlib import Path

import pytest

from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import GameOutcome, Side, TerminalResult
from harbichess.evaluation.system_teacher_qualification import (
    QualificationGame,
    RawPolicy,
    SystemTeacherConfig,
    evaluate_gate,
    summarize_control,
    summarize_games,
)
from harbichess.search.evaluator import PositionEvaluation


class FixedEvaluator:
    def __init__(self, evaluation: PositionEvaluation) -> None:
        self.evaluation = evaluation

    def evaluate(self, state):
        del state
        return self.evaluation


def test_raw_policy_selects_legal_prior_argmax() -> None:
    rules = PythonChessRules()
    state = rules.initial_state()
    moves = rules.legal_moves(state)
    evaluation = PositionEvaluation(
        priors=((moves[0], 0.1), (moves[1], 0.7), (moves[2], 0.2)),
        value=0.25,
    )
    result = RawPolicy(FixedEvaluator(evaluation)).search(
        state, rng=random.Random(3), add_root_noise=False
    )

    assert result.select_move(temperature=0.0, rng=random.Random(4)) == moves[1]
    assert sum(move.visits for move in result.moves) == 1
    assert result.root_value == pytest.approx(0.25)


def _game(pair: int, score: float | None, termination: str = "checkmate"):
    result = (
        TerminalResult.WHITE_WIN
        if score == 1.0
        else TerminalResult.BLACK_WIN
        if score == 0.0
        else TerminalResult.DRAW
    )
    return QualificationGame(
        pair_index=pair,
        candidate_side=None if score is None else Side.WHITE,
        opening_moves=("e2e4",),
        outcome=GameOutcome(result, termination),
        candidate_score=score,
        plies=40,
    )


def test_game_summaries_preserve_color_pairing_and_horizon_rates() -> None:
    games = (
        _game(0, 1.0),
        _game(0, 0.5),
        _game(1, 1.0),
        _game(1, 0.0),
    )
    summary = summarize_games(games, bootstrap_samples=100, seed=7)
    control = summarize_control(
        (_game(0, None, "max_plies"), _game(0, None, "threefold_repetition"))
    )

    assert summary["score_rate"] == pytest.approx(0.625)
    assert summary["decisive_score"] == pytest.approx(2 / 3)
    assert control["max_ply_rate"] == pytest.approx(0.5)
    assert control["threefold_rate"] == pytest.approx(0.5)


def test_system_gate_uses_strength_tactics_and_behavior() -> None:
    strong = {
        "score_rate": 0.65,
        "score_interval": {"low": 0.55, "high": 0.75},
        "decisive_score": 0.6,
        "max_ply_rate": 0.1,
        "threefold_rate": 0.1,
    }
    tactical = {
        "raw": {"solved": 2},
        "budgets": [
            {"budget": 64, "solved": 3, "cases": []},
            {
                "budget": 128,
                "solved": 4,
                "cases": [
                    {"case": "a", "solved": True},
                    {"case": "b", "solved": True},
                ],
            },
            {
                "budget": 256,
                "solved": 5,
                "cases": [
                    {"case": "a", "solved": True},
                    {"case": "b", "solved": True},
                ],
            },
        ],
    }
    passed, reasons = evaluate_gate(
        {64: strong, 128: strong, 256: strong},
        {"max_ply_rate": 0.05, "threefold_rate": 0.05},
        tactical,
    )

    assert passed
    assert reasons == ()

    failed, reasons = evaluate_gate(
        {64: strong, 128: strong, 256: {**strong, "score_rate": 0.49}},
        {"max_ply_rate": 0.05, "threefold_rate": 0.05},
        tactical,
    )
    assert not failed
    assert any("score" in reason for reason in reasons)


def test_system_teacher_config_selects_only_registered_search_kinds() -> None:
    config = SystemTeacherConfig(
        output_dir=Path("output"),
        model_path=Path("model.safetensors"),
        search_kind="full-gumbel",
    )

    assert config.max_considered_actions == 16
    assert config.gumbel_scale == 0.0
    with pytest.raises(ValueError, match="search kind"):
        SystemTeacherConfig(
            output_dir=Path("output"),
            model_path=Path("model.safetensors"),
            search_kind="root-only-imitation",
        )
