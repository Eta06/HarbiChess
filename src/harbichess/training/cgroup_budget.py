"""Explicit Linux aggregate memory policy with failure-time evidence.

Only inactive eligible file cache is subtracted. This estimate is not RSS.
Physical charge and OOM events remain independent hard protections.
"""

from pathlib import Path


class MemoryBudgetExceeded(RuntimeError):
    def __init__(self, reason, snapshot):
        super().__init__(reason)
        self.snapshot = snapshot


def memory_snapshot(root=Path("/sys/fs/cgroup")):
    root = Path(root)
    current = int((root / "memory.current").read_text())
    maximum_text = (root / "memory.max").read_text().strip()
    maximum = None if maximum_text == "max" else int(maximum_text)
    stat = {
        k: int(v)
        for k, v in (line.split() for line in (root / "memory.stat").read_text().splitlines())
    }
    events = {
        k: int(v)
        for k, v in (line.split() for line in (root / "memory.events").read_text().splitlines())
    }
    eligible_file = max(
        0,
        stat.get("file", 0)
        - stat.get("shmem", 0)
        - stat.get("file_dirty", 0)
        - stat.get("file_writeback", 0),
    )
    inactive = min(eligible_file, max(0, stat.get("inactive_file", 0)), current)
    return {
        "schema": "cgroup-inactive-file-budget-v1",
        "total_charge_bytes": current,
        "physical_limit_bytes": maximum,
        "subtracted_inactive_file_bytes": inactive,
        "estimated_active_charge_bytes": current - inactive,
        "stat": stat,
        "events": events,
        "scope": "aggregate cgroup charge, not process RSS; active cache and shmem retained",
    }


class CgroupMemoryBudget:
    def __init__(self, budget_bytes, root=Path("/sys/fs/cgroup"), reserve_bytes=512 * 1024**2):
        if type(budget_bytes) is not int or budget_bytes <= 0 or reserve_bytes <= 0:
            raise ValueError("memory budget and physical reserve must be positive")
        self.root = root
        self.last_snapshot = memory_snapshot(root)
        maximum = self.last_snapshot["physical_limit_bytes"]
        if maximum is not None and maximum <= reserve_bytes:
            raise ValueError("physical memory limit cannot accommodate reserve")
        self.budget_bytes = (
            min(budget_bytes, maximum - reserve_bytes) if maximum is not None else budget_bytes
        )
        self.initial_events = self.last_snapshot["events"].copy()

    def check(self):
        s = self.last_snapshot = memory_snapshot(self.root)
        s["effective_active_budget_bytes"] = self.budget_bytes
        for key in ("oom", "oom_kill"):
            if s["events"].get(key, 0) > self.initial_events.get(key, 0):
                raise MemoryBudgetExceeded("registered cgroup OOM event occurred", s)
        maximum = s["physical_limit_bytes"]
        if maximum is not None and s["total_charge_bytes"] > maximum:
            raise MemoryBudgetExceeded("registered physical cgroup charge ceiling exceeded", s)
        if s["estimated_active_charge_bytes"] > self.budget_bytes:
            raise MemoryBudgetExceeded("registered active cgroup memory ceiling exceeded", s)
        return s
