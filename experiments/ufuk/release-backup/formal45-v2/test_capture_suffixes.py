import base64
import hashlib
import json
import subprocess
import sys

import pytest
import remote_evidence_review as review


@pytest.mark.parametrize("suffix", [".jsonl", ".log", ".stdout", ".stderr"])
def test_exact_public_run_log_is_hash_only(tmp_path, suffix):
    inputs = tmp_path / "inputs"
    runs = tmp_path / "runs"
    inputs.mkdir()
    runs.mkdir()
    path = runs / ("original-failed-partial" + suffix)
    path.write_bytes(b"synthetic private-looking content is never returned\n")
    request = {
        "known": [
            {"path": str(path), "expected_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        ],
        "unresolved": [],
    }
    program = review.REMOTE_PROGRAM.replace(
        "/content/harbichess-fullgame-method2-inputs", str(inputs)
    ).replace("/content/harbichess-runs", str(runs))
    completed = subprocess.run(
        [sys.executable, "-c", program, base64.b64encode(json.dumps(request).encode()).decode()],
        capture_output=True,
        check=True,
    )
    record = json.loads(completed.stdout)["records"][0]
    assert record["status"] == "exact-regular-file-read"
    assert "bytes_base64" not in record
    assert b"synthetic private-looking" not in completed.stdout
    request["known"][0].pop("expected_sha256")
    failed = subprocess.run(
        [sys.executable, "-c", program, base64.b64encode(json.dumps(request).encode()).decode()],
        capture_output=True,
    )
    assert failed.returncode != 0
