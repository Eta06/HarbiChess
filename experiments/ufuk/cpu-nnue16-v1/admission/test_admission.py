import json

import pytest
from admission import pin, receipt, sha


def test_actual_fresh_load_receipt_byte_and_counter_binding(tmp_path):
    f = tmp_path / "native.pt"
    f.write_bytes(b"full model and Adam RNG opaque fixture")
    contract = dict(path="contract.json", sha256="f" * 64)
    cmd = ["python", "train.py", "--contract", "contract.json", "--stop", "256",
           "--resume", str(f), "--resume-sha256", sha(f), "--audit-only"]
    r = dict(status="PASS-readonly-loads-and-fixed-phase-not-strength",
             contract_sha256=contract["sha256"], first=1, finished=2, deadline=3,
             raw_zip_identity_claimed=False, native_sha256={str(f): sha(f)},
             commands=[dict(command=cmd, code=0, stderr="", stdout=json.dumps(
                 dict(status="PASS-strict-native-readonly", step=256)))])
    receipt(r, contract, [256])
    with pytest.raises(ValueError):
        receipt(r, contract, [64])
    with pytest.raises(ValueError):
        receipt(dict(r, commands=[]), contract, [256])
    f.write_bytes(b"corrupted full native")
    with pytest.raises(ValueError):
        receipt(r, contract, [256])


def test_semantic_failure_or_expired_completion_never_masked(tmp_path):
    r = dict(status="FAILED-preserved", contract_sha256="0" * 64, first=1,
             finished=4, deadline=3, raw_zip_identity_claimed=False, commands=[])
    with pytest.raises(ValueError):
        receipt(r, dict(path="c", sha256="0" * 64), [])
    with pytest.raises(FileNotFoundError):
        pin(dict(path=str(tmp_path / "absent"), sha256="0" * 64))
