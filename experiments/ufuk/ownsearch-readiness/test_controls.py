import json
import os
import sys
import time
from pathlib import Path

import pytest
import qualify_cuda_owned as q


def test_junit_requires_actual_cuda_and_no_skip(tmp_path):
    cases = ["cpu-" + str(i) for i in range(10)] + ["resume[cuda:0]"]
    path = tmp_path / "unit.xml"
    path.write_text(
        "<testsuites><testsuite>"
        + "".join('<testcase name="' + name + '"/>' for name in cases)
        + "</testsuite></testsuites>"
    )
    assert len(q.check_junit(path)) == 11
    path.write_text(
        path.read_text().replace(
            'name="resume[cuda:0]"/>', 'name="resume[cuda:0]"><skipped/></testcase>'
        )
    )
    with pytest.raises(ValueError, match="zero skips"):
        q.check_junit(path)


def test_whole_timeout_kills_owned_grandchild(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "MEMORY", 10**20)
    monkeypatch.setattr(q, "DISK", 0)
    pidfile = tmp_path / "pid.json"
    code = (
        "import subprocess,time,json; from pathlib import Path; "
        'p=subprocess.Popen(["'
        + sys.executable
        + '","-c","import time; time.sleep(60)"]); '
        "Path(" + repr(str(pidfile)) + ").write_text(json.dumps(p.pid)); time.sleep(60)"
    )
    with pytest.raises(TimeoutError):
        q.run_owned(
            "timeout",
            [sys.executable, "-c", code],
            cwd=tmp_path,
            env=dict(os.environ),
            output=tmp_path,
            deadline=time.time() + 1,
        )
    pid = json.loads(pidfile.read_text())
    stat = Path("/proc") / str(pid) / "stat"
    assert not stat.exists() or stat.read_text().split(") ", 1)[1].split()[0] == "Z"
    record = json.loads((tmp_path / "timeout.invocation.json").read_text())
    assert record["status"] == "failed" and "TimeoutError" in record["error"]
    assert record["whole_seconds"] < 3
