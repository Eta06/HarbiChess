from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import chess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from collector import SCHEMA, alias, collect  # noqa: E402


class ScriptedSearch:
    def __init__(self, evaluator, moves, branch=False):
        self.evaluator = evaluator
        self.moves = moves
        self.branch = branch

    def search(self, board):
        evaluated = board.copy()
        if self.branch:
            evaluated.push_uci("e2e4")
        self.evaluator(evaluated)
        ply = len(board.move_stack)
        move = (
            chess.Move.from_uci(self.moves[ply])
            if ply < len(self.moves)
            else next(iter(board.legal_moves))
        )
        legal = board.legal_moves.count()
        return SimpleNamespace(
            move=move,
            nodes=legal + 1,
            root_actions=legal,
            evaluations=1,
            completed_depth=1,
            value=0.0,
        )


def pool():
    return {
        "schema": "teacher-selected-ownq-train-roots-v2",
        "selection_status": "pass",
        "train_only": True,
        "source_selection_sha256": "a" * 64,
        "source_teacher_labels_sha256": "b" * 64,
        "rows": [
            {
                "root_id": "root-1",
                "source_row_id": "source-1",
                "trajectory_id": "trajectory-1",
                "root_fen": chess.STARTING_FEN,
                "prefix_uci": [],
                "role": "TRAIN",
            }
        ],
    }


def factory(moves, branch=False):
    return SimpleNamespace(
        evaluator=lambda _board: 0.0,
        make=lambda traced: ScriptedSearch(traced, moves, branch),
    )


def test_collects_exact_terminal_wdl_and_stops_only_after_closed_game() -> None:
    events = []
    result = collect(
        pool(),
        seed=20262905,
        protected=set(),
        factory=factory(["f2f3", "e7e5", "g2g4", "d8h4"]),
        row_limit=4,
        start_limit=1,
        plies=8,
        actor_row_limit=8,
        pool_size=1,
        on_event=events.append,
    )
    assert result["schema"] == SCHEMA
    assert result["status"] == "PASS-exact-closed-terminal-row-budget"
    assert result["training_row_ids"] == [f"root-1:{ply}" for ply in range(4)]
    end = events[-1]
    assert end["type"] == "game_end" and end["status"] == "completed-own-terminal"
    assert end["row_labels"] == [-1, 1, -1, 1]
    assert all(r["train_eligible"] for r in result["rows"])
    assert all(
        r["label_source"] == "own-frozen-parent-search-diagnostic-only" for r in result["rows"]
    )


def test_cap_prefix_is_preserved_unknown_and_never_a_draw_label() -> None:
    events = []
    result = collect(
        pool(),
        seed=20262905,
        protected=set(),
        factory=factory(["e2e4"]),
        row_limit=2,
        start_limit=1,
        plies=8,
        actor_row_limit=1,
        pool_size=1,
        on_event=events.append,
    )
    assert result["status"] == "FAILED-insufficient-closed-terminal-rows"
    end = events[-1]
    assert end["status"] == "unknown-actor-row-budget-prefix"
    assert end["row_labels"] == [None]
    assert end["training_eligible"] is False
    assert result["training_row_ids"] == []


def test_protected_evaluator_branch_excludes_whole_episode() -> None:
    board = chess.Board()
    board.push_uci("e2e4")
    protected = {alias(board)}
    events = []
    result = collect(
        pool(),
        seed=20262905,
        protected=protected,
        factory=factory(["e2e4"], branch=True),
        row_limit=1,
        start_limit=1,
        plies=8,
        actor_row_limit=8,
        pool_size=1,
        on_event=events.append,
    )
    assert result["status"] == "FAILED-insufficient-closed-terminal-rows"
    search_row = events[1]["row"]
    assert search_row["protected_search_aliases"] == [alias(board)]
    assert search_row["selected_action_played"] is False
    assert events[-1]["status"] == "excluded-protected-trajectory"
    assert events[-1]["training_eligible"] is False
    assert events[-1]["row_labels"] == [None]
    assert not result["training_row_ids"]


def test_protected_terminal_final_board_excludes_closed_game() -> None:
    board = chess.Board()
    for move in ["f2f3", "e7e5", "g2g4", "d8h4"]:
        board.push_uci(move)
    events = []
    result = collect(
        pool(),
        seed=20262905,
        protected={alias(board)},
        factory=factory(["f2f3", "e7e5", "g2g4", "d8h4"]),
        row_limit=4,
        start_limit=1,
        plies=8,
        actor_row_limit=8,
        pool_size=1,
        on_event=events.append,
    )
    assert result["status"] == "FAILED-insufficient-closed-terminal-rows"
    end = events[-1]
    assert end["final_state"]["protected"] is True
    assert end["status"] == "excluded-protected-trajectory"
    assert end["row_labels"] == [None] * 4
    assert not result["training_row_ids"]
