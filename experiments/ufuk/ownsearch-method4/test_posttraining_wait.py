import json

import pytest

from ownv1_posttraining import wait_json


def test_expired_but_complete_original_receipt_is_available(tmp_path):
    path = tmp_path / "baseline-result.json"
    receipt = dict(status="completed", finished_epoch=90, original_deadline_epoch=100)
    path.write_text(json.dumps(receipt))
    assert wait_json(path, 100) == receipt
    # The caller still rejects a receipt whose completion missed its ORIGINAL ceiling.
    path.write_text(json.dumps(dict(receipt, finished_epoch=101)))
    received = wait_json(path, 100)
    with pytest.raises(AssertionError):
        assert received["finished_epoch"] <= received["original_deadline_epoch"]


@pytest.mark.parametrize("content", [None, '{"status":'])
def test_expired_missing_or_partial_receipt_fails_immediately(tmp_path, content):
    path = tmp_path / "result.json"
    if content is not None:
        path.write_text(content)
    with pytest.raises(TimeoutError):
        wait_json(path, 100)


def test_completed_failure_receipt_is_returned_without_retry(tmp_path):
    path = tmp_path / "result.json"
    receipt = dict(status="failed", error="original failure preserved")
    path.write_text(json.dumps(receipt))
    assert wait_json(path, 100) == receipt
