import math

import pytest

from harbichess.core.state import ChessMove
from harbichess.evaluation.replay_teacher_alignment import _argmax, _kl, _tv


def test_replay_teacher_policy_distances_and_argmax() -> None:
    first = ChessMove("e2e4")
    second = ChessMove("d2d4")
    raw = ((first, 0.75), (second, 0.25))
    teacher = ((first, 0.25), (second, 0.75))

    assert _argmax(raw) == first
    assert _argmax(teacher) == second
    assert _tv(raw, teacher) == pytest.approx(0.5)
    assert _kl(teacher, raw) == pytest.approx(
        0.25 * math.log(0.25 / 0.75) + 0.75 * math.log(0.75 / 0.25)
    )
