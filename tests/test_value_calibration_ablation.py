from types import SimpleNamespace

from harbichess.core.state import Side
from harbichess.evaluation.value_calibration_ablation import (
    _game_disjoint_halves,
    _tactical_reasons,
)


def test_calibration_split_is_deterministic_game_disjoint_and_stratified() -> None:
    records = tuple(
        SimpleNamespace(
            game_id=f"{phase}-{outcome}-{game}",
            ply=ply + offset,
            outcome_value=outcome * (1 if offset % 2 == 0 else -1),
            side_to_move=Side.WHITE if offset % 2 == 0 else Side.BLACK,
        )
        for phase, ply in (("opening", 4), ("middle", 40), ("end", 90))
        for outcome in (-1, 0, 1)
        for game in range(4)
        for offset in range(2)
    )

    first = _game_disjoint_halves(records, seed=17)
    second = _game_disjoint_halves(records, seed=17)
    fit, test, metrics = first

    assert first == second
    assert {row.game_id for row in fit}.isdisjoint(row.game_id for row in test)
    assert len({row.game_id for row in fit}) == len({row.game_id for row in test}) == 18
    assert set(metrics["strata"].values()) == {4}


def test_tactical_gate_preserves_solved_cases_and_floor() -> None:
    before = {
        "budgets": [
            {
                "solved": 5,
                "cases": [
                    {"case": str(index), "solved": index < 5} for index in range(8)
                ],
            }
        ]
    }
    safe = before
    harmful = {
        "budgets": [
            {
                "solved": 4,
                "cases": [
                    {"case": str(index), "solved": index in (0, 1, 2, 5)}
                    for index in range(8)
                ],
            }
        ]
    }

    assert _tactical_reasons(before, safe) == ()
    assert len(_tactical_reasons(before, harmful)) == 2
