"""Pure cgroup-file fixtures; no real runtime stops, OOMs or jobs."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import runtime

SOURCE = Path("/workspace/work/harbichess/cpu-additive-source-6fcc8b4")
MiB = 1024**2


def fixture(
    root,
    current=1700,
    maximum=4096,
    file=1000,
    shmem=700,
    dirty=100,
    writeback=100,
    inactive=1000,
    oom=0,
):
    (root / "memory.current").write_text(str(current * MiB))
    (root / "memory.max").write_text(str(maximum * MiB))
    (root / "memory.stat").write_text(
        f"file {file * MiB}\nshmem {shmem * MiB}\n"
        f"file_dirty {dirty * MiB}\nfile_writeback {writeback * MiB}\n"
        f"inactive_file {inactive * MiB}\n"
    )
    (root / "memory.events").write_text(f"oom {oom}\noom_kill 0\n")


def test_dirty_shared_writeback_eligible_cache_rejects_old_false_pass(tmp_path):
    fixture(tmp_path)
    cls = runtime._memory_class(SOURCE)
    b = cls(1500 * MiB, root=tmp_path)
    with pytest.raises(RuntimeError, match="active cgroup memory ceiling") as err:
        b.check()
    assert err.value.snapshot["subtracted_inactive_file_bytes"] == 100 * MiB
    assert err.value.snapshot["estimated_active_charge_bytes"] == 1600 * MiB


def test_pinned_budget_retains_actual_physical_and_oom_guards(tmp_path):
    fixture(tmp_path, current=1000, file=500, shmem=0, dirty=0, writeback=0, inactive=500)
    cls = runtime._memory_class(SOURCE)
    budget = cls(1500 * MiB, root=tmp_path)
    fixture(tmp_path, current=1000, file=500, shmem=0, dirty=0, writeback=0, inactive=500, oom=1)
    with pytest.raises(RuntimeError, match="OOM event"):
        budget.check()
    fixture(
        tmp_path, current=1000, maximum=2000, file=500, shmem=0, dirty=0, writeback=0, inactive=500
    )
    budget = cls(1500 * MiB, root=tmp_path)
    fixture(
        tmp_path,
        current=2001,
        maximum=2000,
        file=1900,
        shmem=0,
        dirty=0,
        writeback=0,
        inactive=1900,
    )
    with pytest.raises(RuntimeError, match="physical cgroup charge"):
        budget.check()


def test_recursive_artifacts_exact_boundary_and_memory_budget_reused(tmp_path, monkeypatch):
    artifacts = tmp_path / "stage"
    (artifacts / "native").mkdir(parents=True)
    (artifacts / "native/checkpoint.json").write_bytes(b"x" * 31)
    monkeypatch.setattr(runtime, "ARTIFACT_MAX_BYTES", 32)
    monkeypatch.setattr(runtime.time, "time", lambda: 1.0)
    monkeypatch.setattr(runtime.shutil, "disk_usage", lambda _: SimpleNamespace(free=512 * MiB))
    calls = []

    class Budget:
        def check(self):
            calls.append(1)
            return {"checked": True}

    monkeypatch.setitem(runtime._BUDGETS, tmp_path.resolve(), Budget())
    assert runtime.guard(2.0, tmp_path, artifacts) == {"checked": True}
    assert runtime.guard(2.0, tmp_path, artifacts) == {"checked": True}
    assert len(calls) == 2
    (artifacts / "native/checkpoint.json").write_bytes(b"x" * 32)
    with pytest.raises(RuntimeError, match="recursive artifact"):
        runtime.guard(2.0, tmp_path, artifacts)
    assert len(calls) == 2


def test_owned_wait_budget_failure_stops_only_created_child_group(tmp_path, monkeypatch):
    import qualify_actor_v3 as owner

    calls = []
    signals = []

    class Process:
        pid = 123456
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.returncode = -15
            return self.returncode

    monkeypatch.setattr(owner.time, "time", lambda: 1.0)
    monkeypatch.setattr(owner.subprocess, "Popen", lambda *a, **k: Process())
    monkeypatch.setattr(owner.os, "killpg", lambda pid, sig: signals.append((pid, sig)))

    def guard():
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("synthetic memory guard")

    with pytest.raises(RuntimeError, match="synthetic memory"):
        owner.run_owned(["never-run"], 2.0, tmp_path, "fixture", guard)
    assert len(calls) == 2 and len(signals) == 1 and signals[0][0] == 123456


def test_owned_expired_clock_never_launches(tmp_path, monkeypatch):
    import qualify_actor_v3 as owner

    monkeypatch.setattr(owner.time, "time", lambda: 2.0)
    monkeypatch.setattr(owner.subprocess, "Popen", lambda *a, **k: pytest.fail("expired launched"))
    with pytest.raises(TimeoutError, match="already exhausted"):
        owner.run_owned(["never-run"], 2.0, tmp_path, "expired")
