import importlib.util
from pathlib import Path

import pytest

P = Path(__file__).parent / "cpu_contingency_audit_support.py"
spec = importlib.util.spec_from_file_location("memory_support", P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_cached_file_and_shmem_are_distinct_charges():
    stats = (
        "file 15430000000\nshmem 186000000\nslab_reclaimable 120000000\n"
        "inactive_file 1170000000\nanon 658000000\n"
    )
    result = m.parse_cgroup_memory("16890000000", stats, "17179869184")
    assert result["estimated_nonreclaimable_bytes"] == 1526000000
    assert result["working_set_current_minus_inactive_file_bytes"] == 15720000000
    assert result["current_bytes"] == 16890000000
    assert result["total_charge_ceiling_bytes"] == 16 * 1024**3
    with pytest.raises(AssertionError):
        m.parse_cgroup_memory("1", stats.replace("shmem 186000000", "shmem 99999999999"), "max")


def test_real_cgroup_parser_diagnostic_only():
    result = m.memory_snapshot()
    assert result["metric"] == "estimated-nonreclaimable-cgroup-charge"
    assert result["configured_cgroup_max_bytes"] == 16 * 1024**3
    assert 0 <= result["estimated_nonreclaimable_bytes"] <= result["current_bytes"]


def test_guard_total_ceiling_not_relaxed_by_reclaimable_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(
        m,
        "memory_snapshot",
        lambda: dict(current_bytes=16 * 1024**3 + 1, estimated_nonreclaimable_bytes=1),
    )
    with pytest.raises(RuntimeError, match="total-cgroup-charge-above16GiB"):
        m.guard(m.time.time() + 100, tmp_path)


def test_nonreclaimable_and_deadline_guards_record_named_evidence(monkeypatch, tmp_path):
    monkeypatch.setattr(
        m,
        "memory_snapshot",
        lambda: dict(current_bytes=16 * 1024**3, estimated_nonreclaimable_bytes=15 * 1024**3 + 1),
    )
    with pytest.raises(RuntimeError, match="nonreclaimable-charge-above15GiB"):
        m.guard(m.time.time() + 100, tmp_path)
    with pytest.raises(RuntimeError, match="original-wholedeadline-exhausted"):
        m.guard(m.time.time() - 1, tmp_path)
