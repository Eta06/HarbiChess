from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import chess
import pytest
import torch

import collector
import convert
import native
from model import FEATURE_SCHEMA

ROOT = Path(__file__).resolve().parents[0]


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = load(ROOT / "search_original.py", "tdleaf_search_baseline")
PV = load(ROOT / "search_pv.py", "tdleaf_search_capture")
TARGETS = load(ROOT / "tdleaf_targets.py", "tdleaf_targets")
OBJECTIVE = load(ROOT / "tdleaf_objective.py", "tdleaf_objective")
CAPTURE_RECORD = load(ROOT / "capture_record.py", "tdleaf_capture_record")


class Static:
    def __init__(self):
        self.calls = []

    def __call__(self, board):
        self.calls.append((board.fen(), tuple(move.uci() for move in board.move_stack)))
        # Small deterministic nonzero signal in the evaluator's mover POV.
        material = sum(
            (1 if piece.color == board.turn else -1) * piece.piece_type
            for piece in board.piece_map().values()
        )
        return max(-0.8, min(0.8, material / 100.0))


@pytest.mark.parametrize(
    "fen,nodes",
    [
        (chess.STARTING_FEN, 512),
        (chess.STARTING_FEN, 8192),
        ("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3", 768),
        ("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2", 512),
    ],
)
def test_captured_search_preserves_exact_search_and_evaluator_order(fen: str, nodes: int) -> None:
    board_a = chess.Board(fen)
    board_b = chess.Board(fen)
    base_eval, pv_eval = Static(), Static()
    observed = []
    original = BASE.BudgetSearch(base_eval, nodes=nodes, quiescence_plies=2, max_depth=8).search(
        board_a
    )
    captured = PV.BudgetSearch(
        pv_eval,
        nodes=nodes,
        quiescence_plies=2,
        max_depth=8,
        input_observer=lambda board: observed.append(
            (board.fen(), tuple(move.uci() for move in board.move_stack))
        ),
    ).search(board_b)
    assert (
        captured.move,
        captured.value.hex(),
        captured.nodes,
        captured.evaluations,
        captured.completed_depth,
        captured.root_actions,
    ) == (
        original.move,
        original.value.hex(),
        original.nodes,
        original.evaluations,
        original.completed_depth,
        original.root_actions,
    )
    assert pv_eval.calls == base_eval.calls
    assert observed == pv_eval.calls
    assert tuple(board_b.move_stack) == tuple(board_a.move_stack) == ()
    assert captured.pv and captured.pv[0] == captured.move
    leaf_board = board_b.copy(stack=True)
    for move in captured.pv:
        assert move in leaf_board.legal_moves
        leaf_board.push(move)
    assert captured.leaf.ply == len(captured.pv)
    assert captured.leaf.kind in {"static", "terminal"}
    if fen.startswith("rnbqkbnr/pppp1ppp/8/4p3/6P1"):
        assert captured.move == chess.Move.from_uci("d8h4")
        assert captured.leaf.kind == "terminal"
    if captured.leaf.kind == "static":
        assert (
            leaf_board.fen(),
            tuple(move.uci() for move in leaf_board.move_stack),
        ) in pv_eval.calls
        assert (
            captured.value.hex()
            == ((-1.0 if len(captured.pv) % 2 else 1.0) * captured.leaf.value).hex()
        )
    else:
        assert leaf_board.outcome(claim_draw=True) is not None
        expected = (-1.0 if len(captured.pv) % 2 else 1.0) * captured.leaf.value
        assert captured.value.hex() == expected.hex()


def _record(root_id, index, history, played, root_score, pv=None, leaf_value=None, kind=None):
    board = chess.Board()
    for token in history:
        board.push_uci(token)
    if pv is None:
        pv = [played]
    leaf = board.copy(stack=True)
    for token in pv:
        leaf.push_uci(token)
    outcome = leaf.outcome(claim_draw=True)
    if kind is None:
        kind = "terminal" if outcome is not None else "static"
    if kind == "terminal":
        leaf_value = None
        leaf_wdl = 0 if outcome.winner is None else (1 if outcome.winner == leaf.turn else -1)
    else:
        leaf_value = (
            -root_score
            if leaf_value is None and len(pv) % 2
            else root_score
            if leaf_value is None
            else leaf_value
        )
        leaf_wdl = None
    return {
        "root_id": root_id,
        "local_ply": index,
        "episode_status": "completed-own-terminal",
        "root_fen": chess.STARTING_FEN,
        "history_uci": history,
        "history_sha256": TARGETS._sha({"root_fen": chess.STARTING_FEN, "history_uci": history}),
        "mover": "white" if board.turn else "black",
        "selected_action_uci": played,
        "played_action_uci": played,
        "played_action": True,
        "train_eligible": True,
        "protected_search_aliases": [],
        "search_root_value": root_score,
        "pv_uci": list(pv),
        "leaf_kind": kind,
        "leaf_value_mover": leaf_value,
        "leaf_terminal_mover_wdl": leaf_wdl,
        "leaf_fen": leaf.fen(),
        "leaf_history_sha256": TARGETS._sha(
            {"root_fen": chess.STARTING_FEN, "history_uci": history + list(pv)}
        ),
    }


def _four_ply_mate_rows():
    histories = [
        [],
        ["f2f3"],
        ["f2f3", "e7e5"],
        ["f2f3", "e7e5", "g2g4"],
    ]
    actions = ["f2f3", "e7e5", "g2g4", "d8h4"]
    movers = ["white", "black", "white", "black"]
    q_white = [0.2, 0.4, 0.6, -1.0]
    rows = []
    for i, (history, action, mover, white_q) in enumerate(
        zip(histories, actions, movers, q_white, strict=True)
    ):
        root_q = white_q if mover == "white" else (1.99999 if i == 3 else -white_q)
        rows.append(_record("mate-game", i, history, action, root_q))
    return rows


def test_lambda_targets_terminal_and_unknown_use_correct_pov_and_endpoint():
    rows = _four_ply_mate_rows()
    known = TARGETS.make_episode(rows, terminal_white_wdl=-1)
    assert [round(row["tdleaf_target_white"], 6) for row in known["rows"]] == [
        0.1,
        -0.2,
        -1.0,
        -1.0,
    ]
    assert [row["tdleaf_target_leaf_mover"] for row in known["rows"]] == pytest.approx(
        [-0.1, -0.2, 1.0, None]
    )
    assert [row["gradient_mask"] for row in known["rows"]] == [True, True, True, False]
    unknown_rows = [dict(row, episode_status="unknown-ply-cap") for row in rows[:3]]
    unknown = TARGETS.make_episode(unknown_rows, terminal_white_wdl=None)
    assert unknown["status"] == "targeted-unknown-bootstrap"
    assert unknown["endpoint"] == "last-recorded-preaction-search-bootstrap"
    assert unknown["rows"][-1]["tdleaf_target_white"] == 0.6


def test_terminal_pv_leaf_is_exact_and_gradient_masked():
    row = _record(
        "terminal-pv", 0, [], "f2f3", -1.99996, ["f2f3", "e7e5", "g2g4", "d8h4"], kind="terminal"
    )
    row["leaf_terminal_mover_wdl"] = -1
    # The searched terminal PV is in the game tree; actual training episode is UNKNOWN.
    row["episode_status"] = "unknown-ply-cap"
    target = TARGETS.make_episode([row], terminal_white_wdl=None)
    assert target["rows"][0]["gradient_mask"] is False
    assert target["rows"][0]["tdleaf_target_leaf_mover"] is None


def test_odd_pv_parity_maps_white_return_back_to_leaf_mover():
    row = _record("odd-pv", 0, [], "e2e4", 0.2, ["e2e4"], leaf_value=-0.2)
    row["episode_status"] = "unknown-ply-cap"
    target = TARGETS.make_episode([row], terminal_white_wdl=None)
    only = target["rows"][0]
    assert only["gradient_sign_leaf_to_white"] == -1.0
    assert only["tdleaf_target_white"] == 0.2
    assert only["tdleaf_target_leaf_mover"] == pytest.approx(-0.2)


def test_capture_row_binds_completed_result_without_another_search():
    board = chess.Board()
    evaluator = Static()
    result = PV.BudgetSearch(evaluator, nodes=512).search(board)
    row = CAPTURE_RECORD.capture_row(
        root_id="capture-fixture",
        local_ply=0,
        root_board=board,
        played_action=result.move,
        result=result,
        episode_status="unknown-ply-cap",
    )
    assert row["played_action_uci"] == row["pv_uci"][0] == result.move.uci()
    assert row["search_evaluations"] == len(evaluator.calls)
    assert row["leaf_history_sha256"] == TARGETS._sha(
        {"root_fen": chess.STARTING_FEN, "history_uci": row["pv_uci"]}
    )
    excluded = CAPTURE_RECORD.capture_row(
        root_id="protected-fixture",
        local_ply=0,
        root_board=board,
        played_action=None,
        result=result,
        episode_status="unknown-ply-cap",
        train_eligible=False,
        protected_search_aliases=[42],
    )
    assert excluded["selected_action_uci"] == result.move.uci()
    assert excluded["played_action_uci"] is None
    assert TARGETS.make_episode([excluded], terminal_white_wdl=None)["status"] == (
        "excluded-protected-or-unplayed-episode"
    )


def test_protected_episode_excluded_whole_and_unknown_cannot_be_draw():
    rows = _four_ply_mate_rows()
    rows[1]["protected_search_aliases"] = [123]
    assert (
        TARGETS.make_episode(rows, terminal_white_wdl=-1)["status"]
        == "excluded-protected-or-unplayed-episode"
    )
    unknown = [
        dict(row, episode_status="unknown-row-budget-prefix") for row in _four_ply_mate_rows()[:2]
    ]
    with pytest.raises(ValueError, match="UNKNOWN cap"):
        TARGETS.make_episode(unknown, terminal_white_wdl=0)


def test_terminal_actual_endpoint_must_not_be_hidden_as_unknown():
    rows = [dict(row, episode_status="unknown-ply-cap") for row in _four_ply_mate_rows()]
    with pytest.raises(ValueError, match="terminal game endpoint"):
        TARGETS.make_episode(rows, terminal_white_wdl=None)


def test_tdleaf_objective_is_masked_semi_gradient():
    pred = torch.tensor([0.2, -0.1, 0.4], requires_grad=True)
    target = torch.tensor([0.0, 0.5, 100.0], requires_grad=True)
    loss = OBJECTIVE.loss(pred, target, [True, True, False])
    loss.backward()
    assert target.grad is None
    assert pred.grad.tolist() == pytest.approx([0.2, -0.6, 0.0])


class FeatureShim:
    @staticmethod
    def board_indices(_board):
        return [1, 2]


class PriorShim:
    PRIOR = (0.0,)
    SCALE = 1.0

    @staticmethod
    def features(_board):
        return [0.0]


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _synthetic_ledger():
    events = []
    alias_bytes = bytearray()
    train_ids = []
    exposed = set()
    for game in range(128):
        root_id = f"fixture-{game:03d}"
        board = chess.Board()
        prefix = []
        events.append(
            {
                "type": "game_start",
                "root_id": root_id,
                "root_ordinal": game,
                "root_fen": chess.STARTING_FEN,
                "root_prefix_uci": [],
                "root_alias": convert.alias(board),
                "root_prefix_aliases": [convert.alias(board)],
            }
        )
        exposed.add(convert.alias(board))
        labels = []
        for ply in range(8):
            move = next(iter(board.legal_moves))
            before = board.copy(stack=True)
            mover = "white" if before.turn else "black"
            board.push(move)
            leaf = board.copy(stack=True)
            leaf_alias = convert.alias(leaf)
            exposed.add(convert.alias(before))
            exposed.add(leaf_alias)
            offset = len(alias_bytes)
            alias_bytes.extend(__import__("struct").pack("<q", leaf_alias))
            history_sha = hashlib.sha256(
                _canonical({"root_fen": chess.STARTING_FEN, "history_uci": prefix})
            ).hexdigest()
            pv = [move.uci()]
            leaf_sha = hashlib.sha256(
                _canonical({"root_fen": chess.STARTING_FEN, "history_uci": prefix + pv})
            ).hexdigest()
            row_id = f"{root_id}:{ply}"
            train_ids.append(row_id)
            events.append(
                {
                    "type": "search_row",
                    "row": {
                        "root_id": root_id,
                        "root_ordinal": game,
                        "root_fen": chess.STARTING_FEN,
                        "root_prefix_uci": [],
                        "local_ply": ply,
                        "history_uci": prefix.copy(),
                        "history_sha256": history_sha,
                        "fen4": " ".join(before.fen().split()[:4]),
                        "mover": mover,
                        "root_alias": convert.alias(before),
                        "search_root_value": -0.2,
                        "selected_action_uci": move.uci(),
                        "played_action_uci": move.uci(),
                        "played_action": True,
                        "selected_action_played": True,
                        "nodes": before.legal_moves.count() + 1,
                        "evaluations": 1,
                        "completed_depth": 1,
                        "root_actions": before.legal_moves.count(),
                        "actual_eval_calls": 1,
                        "protected_search_aliases": [],
                        "pv_uci": pv,
                        "leaf_kind": "static",
                        "leaf_value_mover": 0.2,
                        "leaf_search_value_mover": 0.2,
                        "leaf_terminal_mover_wdl": None,
                        "leaf_fen": leaf.fen(),
                        "leaf_history_sha256": leaf_sha,
                        "train_eligible": False,
                        "search_alias_ref": {
                            "file": "aliases-0000.bin",
                            "offset_bytes": offset,
                            "count": 1,
                        },
                    },
                }
            )
            prefix.append(move.uci())
            labels.append(None)
        events.append(
            {
                "type": "game_end",
                "root_id": root_id,
                "status": "unknown-ply-cap",
                "final_state": {
                    "fen4": " ".join(board.fen().split()[:4]),
                    "history_uci": prefix.copy(),
                    "alias": convert.alias(board),
                    "protected": False,
                },
                "training_eligible": True,
                "row_labels": labels,
            }
        )
        exposed.add(convert.alias(board))
    receipt = {
        "schema": "human-prior-tdleaf-collection-receipt-v1",
        "status": "PASS-exact-row-budget",
        "teacher_labels_used": False,
        "train_rows": 1024,
        "seed": 1,
        "starts_considered": 128,
        "all_actor_rows": 1024,
        "unique_exposure_aliases": len(exposed),
        "training_row_ids": train_ids,
    }
    return events, {"aliases-0000.bin": bytes(alias_bytes)}, receipt


def test_converter_replays_full_eligible_ledger_and_builds_static_leaf_rows():
    events, sidecars, receipt = _synthetic_ledger()
    data = convert.convert_events(
        events=events,
        sidecars=sidecars,
        protected_aliases=set(),
        receipt=receipt,
        features=FeatureShim,
        prior=PriorShim,
    )
    assert data["schema"] == "own-tdleaf-pv-leaf-sparse-training-data-v1"
    assert data["eligible_actor_rows"] == 1024
    assert data["gradient_rows"] == len(data["rows"]) == 1024
    assert all(-1 <= row["target"] <= 1 for row in data["rows"])
    assert len({round(row["target"], 6) for row in data["rows"]}) > 1
    assert data["rows"][0]["indices"] == [1, 2]


@pytest.mark.parametrize("corruption", ["pv", "receipt-order", "span"])
def test_converter_rejects_resealed_semantic_corruption(corruption):
    events, sidecars, receipt = _synthetic_ledger()
    if corruption == "pv":
        events[1]["row"]["pv_uci"] = ["a1a8"]
    elif corruption == "receipt-order":
        receipt["training_row_ids"][0], receipt["training_row_ids"][1] = (
            receipt["training_row_ids"][1],
            receipt["training_row_ids"][0],
        )
    else:
        events[1]["row"]["search_alias_ref"]["offset_bytes"] = len(sidecars["aliases-0000.bin"]) + 8
    with pytest.raises(ValueError):
        convert.convert_events(
            events=events,
            sidecars=sidecars,
            protected_aliases=set(),
            receipt=receipt,
            features=FeatureShim,
            prior=PriorShim,
        )


def test_start_selection_excludes_protected_alias_in_source_prefix():
    row = {
        "role": "TRAIN",
        "root_id": "protected-prefix-root",
        "root_fen": chess.STARTING_FEN,
        "prefix_uci": ["e2e4"],
        "trajectory_id": "source-trajectory",
        "source_row_id": "source-row",
    }
    pool = {
        "schema": collector.POOL,
        "selection_status": "pass",
        "train_only": True,
        "source_selection_sha256": "a" * 64,
        "ancestral_teacher_labels_sha256": "b" * 64,
        "rows": [row],
    }
    protected = {collector.alias(chess.Board())}
    with pytest.raises(ValueError, match="pool exhausted"):
        collector.starts(pool, 1, protected, n=1, pool_size=1)


def test_tdleaf_native_newadam_resumes_exactly_and_changes_only_trained_state():
    torch.set_num_threads(1)
    contract = {
        "phase": native.SYNTHETIC_PHASE,
        "updates": 8,
        "seed": 123,
        "math": native.MATH,
        "feature_schema": FEATURE_SCHEMA,
    }
    learner = native.Learner(contract)
    initial = learner.native()
    rows = [{"indices": [1, 2], "prior_logit": 0.0, "target": 0.25}] * 256
    learner.advance(rows, 1)
    state = learner.native()
    assert state["step"] == 1
    assert not native.bits_equal(state["model"], initial["model"])
    restored = native.Learner(contract, state=state)
    assert native.bits_equal(state, restored.native())


def test_tdleaf_synthetic_whole_and_pause_resume_fullpayload_identity(tmp_path):
    torch.set_num_threads(1)
    contract = {
        "phase": native.SYNTHETIC_PHASE,
        "updates": 8,
        "seed": 707,
        "math": native.MATH,
        "feature_schema": FEATURE_SCHEMA,
    }
    rows = [
        {
            "indices": [1 + i % 4, 10 + i % 4],
            "prior_logit": 0.01 * (i % 3),
            "target": (i % 7 - 3) / 10,
        }
        for i in range(256)
    ]
    whole = native.Learner(contract)
    whole.advance(rows, 8)
    whole_state = whole.native()

    split = native.Learner(contract)
    split.advance(rows, 4)
    pause_path = tmp_path / "pause.pt"
    torch.save(split.native(), pause_path)
    resumed = native.load_native(pause_path, contract)
    resumed.advance(rows, 8)
    resumed_path = tmp_path / "resume.pt"
    torch.save(resumed.native(), resumed_path)
    assert native.bits_equal(whole_state, native.load_native(resumed_path, contract).native())
    for step, state in (
        (0, native.Learner(contract).native()),
        (4, torch.load(pause_path, weights_only=False)),
        (8, whole_state),
    ):
        path = tmp_path / f"step-{step}.pt"
        torch.save(state, path)
        assert native.load_native(path, contract).step == step
