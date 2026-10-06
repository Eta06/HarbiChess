import copy
import hashlib
import math
from types import SimpleNamespace

import chess
import pytest
from convert import verify_row


@pytest.mark.parametrize("cp,mate", [(-600, None), (0, None), (None, -2), (None, 2)])
def test_actual_black_history_proxy_and_alignment(cp, mate):
    root = chess.STARTING_FEN
    board = chess.Board(root)
    board.push_uci("e2e4")
    target = math.tanh(cp / 600) if cp is not None else (1.0 if mate > 0 else -1.0)
    selected = dict(row_id="game:1", trajectory_id="game", root_fen=root,
                    prefix_uci=["e2e4"], fen4=" ".join(board.fen().split()[:4]),
                    history_sha256=hashlib.sha256((root + "\ne2e4").encode()).hexdigest())
    row = dict(selected, ordinal=0, root_mover="black", selected_teacher_uci="e7e5",
               teacher_nodes_actual=8193, nominal_nodes=8192, node_overrun=1,
               teacher_depth=4, teacher_time_seconds=0.1, indices=[1, 7], prior_logit=0.25,
               target=target, raw_teacher=dict(cp_mover=cp, mate_mover=mate, target=target,
                                               target_is_calibrated_WDL=False))
    f = SimpleNamespace(board_indices=lambda b: [1, 7])
    p = SimpleNamespace(PRIOR=[150], SCALE=600, features=lambda b: [1])
    before = copy.deepcopy(row)
    assert verify_row(row, selected, 0, f, p) == dict(indices=[1, 7], prior_logit=0.25,
                                                    target=target)
    assert row == before
    for key, value in [("root_mover", "white"), ("indices", [7, 1]),
                       ("prior_logit", 0.2500001), ("node_overrun", 0),
                       ("prefix_uci", ["e2e5"]), ("target", -target if target else 0.1)]:
        bad = dict(row, **{key: value})
        with pytest.raises(ValueError):
            verify_row(bad, selected, 0, f, p)
