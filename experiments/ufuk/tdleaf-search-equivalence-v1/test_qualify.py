import random

import chess

import qualify
import search_original
import search_pv


def test_pv_instrumentation_preserves_search_and_exact_ordered_evaluator_inputs():
    board = chess.Board()
    baseline_trace = qualify.Trace(lambda b: (b.piece_type_at(chess.E4) or 0) / 6 - 0.2)
    pv_trace = qualify.Trace(lambda b: (b.piece_type_at(chess.E4) or 0) / 6 - 0.2)
    base = search_original.BudgetSearch(
        baseline_trace, nodes=512, quiescence_plies=2, max_depth=4
    ).search(board)
    pv = search_pv.BudgetSearch(
        pv_trace,
        input_observer=pv_trace.observe,
        nodes=512,
        quiescence_plies=2,
        max_depth=4,
    ).search(board)
    assert qualify.result_payload(base) == qualify.result_payload(pv)
    assert baseline_trace.aliases == pv_trace.observed_aliases
    assert baseline_trace.aliases == pv_trace.aliases
    assert baseline_trace.evaluated_inputs == pv_trace.evaluated_inputs
    assert pv_trace.observed_inputs == [
        {"fen": row["fen"], "history_uci": row["history_uci"]} for row in pv_trace.evaluated_inputs
    ]
    leaf_board = qualify.verify_pv(board, pv, pv_trace.observed_aliases)
    assert len(pv.pv) == pv.leaf.ply
    assert leaf_board.is_valid()


def test_root_selection_is_fixed_order_and_rejects_protected_actual_root():
    rng = random.Random(42)
    rows = []
    histories = set()
    while len(rows) < 4096:
        board = chess.Board()
        for _ in range(rng.randint(3, 18)):
            if board.outcome(claim_draw=True) is not None:
                break
            board.push(rng.choice(list(board.legal_moves)))
        history = [move.uci() for move in board.move_stack]
        key = tuple(history)
        if len(history) < 2 or key in histories or board.outcome(claim_draw=True) is not None:
            continue
        histories.add(key)
        rows.append(
            {
                "role": "TRAIN",
                "root_fen": chess.STARTING_FEN,
                "prefix_uci": history,
                "root_id": f"root-{len(rows)}",
            }
        )
    pool = {"rows": rows}
    start_alias = qualify.alias(chess.Board())
    selected, considered = qualify.choose_roots(pool, {start_alias})
    assert len(selected) == 24
    assert considered == 24
    assert all(qualify.alias(qualify.replay(row)) != start_alias for row in selected)
