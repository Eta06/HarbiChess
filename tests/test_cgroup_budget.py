import pytest

from harbichess.training.cgroup_budget import (
    CgroupMemoryBudget,
    MemoryBudgetExceeded,
    memory_snapshot,
)

GIB = 1024**3


def cgroup(
    root,
    current=10 * GIB,
    maximum=16 * GIB,
    *,
    file=9 * GIB,
    inactive=8 * GIB,
    shmem=0,
    dirty=0,
    writeback=0,
    oom=0,
    oom_kill=0,
):
    (root / "memory.current").write_text(str(current))
    (root / "memory.max").write_text(str(maximum))
    (root / "memory.stat").write_text(
        f"file {file}\ninactive_file {inactive}\nshmem {shmem}\n"
        f"file_dirty {dirty}\nfile_writeback {writeback}\n"
    )
    (root / "memory.events").write_text(f"oom {oom}\noom_kill {oom_kill}\n")
    return root


def test_inactive_file_can_fit_budget_without_disabling_physical_limit(tmp_path):
    budget = CgroupMemoryBudget(4 * GIB, cgroup(tmp_path))
    s = budget.check()
    assert s["total_charge_bytes"] == 10 * GIB
    assert s["estimated_active_charge_bytes"] == 2 * GIB
    assert s["effective_active_budget_bytes"] == 4 * GIB


def test_active_file_shmem_dirty_writeback_are_not_discounted(tmp_path):
    cgroup(tmp_path, inactive=9 * GIB, shmem=3 * GIB, dirty=GIB, writeback=GIB)
    s = memory_snapshot(tmp_path)
    assert s["subtracted_inactive_file_bytes"] == 4 * GIB
    assert s["estimated_active_charge_bytes"] == 6 * GIB
    budget = CgroupMemoryBudget(5 * GIB, tmp_path)
    with pytest.raises(MemoryBudgetExceeded, match="active cgroup") as raised:
        budget.check()
    assert raised.value.snapshot["stat"]["shmem"] == 3 * GIB


def test_charge_over_physical_limit_is_rejected_even_if_cache_adjusted_fits(tmp_path):
    budget = CgroupMemoryBudget(
        15 * GIB, cgroup(tmp_path, current=17 * GIB, file=16 * GIB, inactive=15 * GIB)
    )
    with pytest.raises(MemoryBudgetExceeded, match="physical cgroup"):
        budget.check()


@pytest.mark.parametrize("event", ["oom", "oom_kill"])
def test_oom_event_increase_fails_and_preserves_event_receipt(tmp_path, event):
    budget = CgroupMemoryBudget(4 * GIB, cgroup(tmp_path))
    cgroup(tmp_path, **{event: 1})
    with pytest.raises(MemoryBudgetExceeded, match="OOM") as raised:
        budget.check()
    assert raised.value.snapshot["events"][event] == 1


def test_requested_budget_cannot_increase_physical_allocation(tmp_path):
    budget = CgroupMemoryBudget(64 * GIB, cgroup(tmp_path))
    assert budget.budget_bytes == 16 * GIB - 512 * 1024**2


def test_unknown_missing_cgroup_data_is_not_reported_safe(tmp_path):
    with pytest.raises(FileNotFoundError):
        CgroupMemoryBudget(GIB, tmp_path)


def test_unlimited_physical_cgroup_still_requires_finite_registered_budget(tmp_path):
    budget = CgroupMemoryBudget(4 * GIB, cgroup(tmp_path, maximum="max"))
    assert budget.check()["effective_active_budget_bytes"] == 4 * GIB
