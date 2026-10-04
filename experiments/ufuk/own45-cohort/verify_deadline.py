"""Read-only regression for new final-phase clocks and frozen conservative audit guards."""

import ast
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).parents[1]
FINAL = 1791180000
LEGACY = 1791170400


def constants(path):
    return [
        node.value
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Constant) and type(node.value) is int
    ]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert datetime.fromtimestamp(FINAL, UTC).isoformat() == "2026-10-05T06:00:00+00:00"
    assert datetime.fromtimestamp(FINAL, ZoneInfo("Europe/Istanbul")).isoformat() == (
        "2026-10-05T09:00:00+03:00"
    )
    assert datetime.fromtimestamp(LEGACY, UTC).isoformat() == "2026-10-05T03:20:00+00:00"
    assert int(datetime(2026, 10, 5, 6, tzinfo=UTC).timestamp()) == FINAL
    checked = {}
    for directory, prefix in [("ownsearch-method4", "ownv1"), ("search-acting-method5", "own5")]:
        for suffix in [
            "formal_config_factory",
            "owner_config_factory",
            "training_controller",
            "audit_controller",
            "posttraining",
            "strength_runtime",
            "mlx_receipt_bridge",
        ]:
            path = ROOT / directory / f"{prefix}_{suffix}.py"
            values = constants(path)
            assert FINAL in values, f"New global deadline absent: {path}"
            assert LEGACY not in values, f"Wrong old global deadline retained: {path}"
            checked[str(path.relative_to(ROOT))] = sha(path)
        path = ROOT / directory / "own45_cohort.py"
        assert FINAL in constants(path) and LEGACY not in constants(path)
        checked[str(path.relative_to(ROOT))] = sha(path)
    own4 = ROOT / "ownsearch-method4"
    own5 = ROOT / "search-acting-method5"
    qualified4 = json.loads((own4 / "E1-auditor-requalification-v2-protocol.json").read_text())
    qualified5 = json.loads((own5 / "fullshape-development-protocol.json").read_text())
    # These prospective/qualified files have independent pinned input provenance.
    # Never rewrite them to pretend a new final-phase deadline was qualified.
    for path, expected in [
        (
            own4 / "ownv1_full_audit.py",
            "e0ae056937a39faf04c05958725c80240c9fabdfcfdebe10dedc4221aa5bf2a7",
        ),
        (
            own5 / "own5_adapter_controls.py",
            "e5fb4dc3f18919885d906fffac8849607344294176cbc0fc87d7bb679ea8a7de",
        ),
        (
            own5 / "own5_qualify_e1.py",
            "a85a2745241e4a0d93652033fe1d4183530ba26e7abcbfcd79393f9a0e705743",
        ),
        (
            own5 / "own5_audit_core.py",
            "10a717eb0c0db21a2f23e502154a4fb4b16011a0df51de576e245c78f67c5155",
        ),
    ]:
        assert sha(path) == expected, f"Qualified helper bytes changed: {path}"
    assert LEGACY in constants(own4 / "ownv1_full_audit.py")
    assert LEGACY in constants(own5 / "own5_adapter_controls.py")
    assert LEGACY in constants(own5 / "own5_qualify_e1.py")
    assert "e0ae056937a39faf04c05958725c80240c9fabdfcfdebe10dedc4221aa5bf2a7" in (
        json.dumps(qualified4)
    )
    for name in ["own5_adapter_controls.py", "own5_qualify_e1.py", "own5_audit_core.py"]:
        assert sha(own5 / name) in json.dumps(qualified5)
    latest = datetime(2026, 10, 4, 23, 43, tzinfo=UTC).timestamp()
    ends = {}
    for budget in (3600, 6000, 12600):
        end = latest + budget
        assert end < LEGACY < FINAL
        ends[str(budget)] = datetime.fromtimestamp(end, UTC).isoformat()
    assert ends["12600"] == "2026-10-05T03:13:00+00:00"
    assert ends["6000"] == "2026-10-05T01:23:00+00:00"
    print(
        json.dumps(
            {
                "status": "pass-actual-file-deadlines-and-frozen-qualified-guards",
                "UTC": datetime.fromtimestamp(FINAL, UTC).isoformat(),
                "Istanbul": datetime.fromtimestamp(FINAL, ZoneInfo("Europe/Istanbul")).isoformat(),
                "new_controller_sha256": checked,
                "latest_start_UTC": "2026-10-04T23:43:00Z",
                "conservative_train_audit_budget_ends_UTC": ends,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
