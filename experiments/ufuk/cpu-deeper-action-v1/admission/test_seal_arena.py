"""Receipt corruption tests only; no model construction, actual jobs or seal writes."""

import copy
import json
from pathlib import Path

import pytest
from seal_arena import validate_receipts

ROOT = Path('/workspace/work/harbichess/continuation-20261005/ACTION-training-v1-registration')


def receipts():
    return [json.loads((ROOT / name).read_bytes()) for name in
            ['proofs-finished.json', 'fits-and-freshloads-finished.json', 'fit-protocol.json']]


def test_actual_closed_receipts_structurally_bind_both_seeds():
    proof, fits, protocol = receipts()
    assert set(validate_receipts(proof, fits, protocol)) == {20262905, 20262906}


@pytest.mark.parametrize('field', ['expired', 'duplicate', 'partial', 'wrongstep', 'fakepass'])
def test_receipt_corruption_rejected(field):
    proof, fits, protocol = copy.deepcopy(receipts())
    if field == 'expired':
        fits['finished_epoch'] = fits['deadline_epoch'] + 0.1
    elif field == 'duplicate':
        fits['rows'][1] = fits['rows'][0]
    elif field == 'partial':
        proof['rows'][0]['commands'] = []
    elif field == 'wrongstep':
        fits['rows'][0]['fresh_interpreter_loads'][1]['step'] = 63
    else:
        fits['rows'][0]['fresh_interpreter_loads'][0]['receipt']['status'] = 'almost-PASS'
    with pytest.raises(ValueError):
        validate_receipts(proof, fits, protocol)
