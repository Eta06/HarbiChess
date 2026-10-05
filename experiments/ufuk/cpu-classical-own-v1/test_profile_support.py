"""No search/NN/model operation: actual production receipt publication only."""

import json

import pytest
from arena.profile_support import publish_result


def test_profile_output_publish_once_never_overwrites(tmp_path):
    path = tmp_path / "result.json"
    receipt = {"status": "PASS-classical-mixed-search-qualification-not-strength", "rows": []}
    actual = publish_result(path, receipt, 600.0, clock=lambda: 599.0)
    original = path.read_bytes()
    assert actual["finished_epoch"] == 599.0
    with pytest.raises(FileExistsError):
        publish_result(path, {"status": "other", "rows": []}, 600.0, clock=lambda: 599.0)
    assert path.read_bytes() == original


def test_final_profile_deadline_expired_never_passes(tmp_path):
    path = tmp_path / "late.json"
    actual = publish_result(
        path,
        {"status": "PASS-classical-mixed-search-qualification-not-strength", "rows": []},
        600.0,
        clock=lambda: 600.0,
    )
    assert actual["status"] == "failed-preserved"
    assert json.loads(path.read_text())["status"] == "failed-preserved"
    assert actual["rows"][-1]["error"].startswith("original600")
