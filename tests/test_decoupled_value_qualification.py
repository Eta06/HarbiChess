from pathlib import Path

import pytest

from harbichess.evaluation.decoupled_value_qualification import (
    DecoupledValueQualificationConfig,
    _tactical_gate,
)


def _tactical(*, raw: int, solved: tuple[str, ...]) -> dict[str, object]:
    cases = tuple(
        {"case": case, "solved": case in solved}
        for case in ("mate", "capture", "defense", "fork", "quiet")
    )
    return {
        "raw": {"solved": raw},
        "budgets": ({"solved": len(solved), "cases": cases},),
    }


def test_tactical_gate_preserves_solved_cases_and_minimum_count() -> None:
    baseline = _tactical(raw=2, solved=("mate", "capture", "defense", "fork"))

    assert _tactical_gate(baseline, baseline) == ()
    reasons = _tactical_gate(
        baseline,
        _tactical(raw=1, solved=("mate", "capture", "quiet")),
    )

    assert len(reasons) == 3


def test_value_qualification_rejects_invalid_hash() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        DecoupledValueQualificationConfig(
            output_dir=Path("output"),
            value_result=Path("value.json"),
            model_path=Path("model.safetensors"),
            expected_candidate_sha256="short",
        )
