"""One CPU, pinned-source aggregate memory/physical/OOM policy and artifact cap.

Prospective repair before actual V3 qualification. No change to numeric ceilings.
An external process owner/guardian remains required.
"""

import hashlib
import os
import shutil
import sys
import time
from pathlib import Path

HARD_END = 1791273600.0
ARTIFACT_MAX_BYTES = 16 * 1024**2
MEMORY_BUDGET_BYTES = 15 * 1024**3
CGROUP_HELPER_SHA256 = "a53079851ab9fb63b65dd5bef0d0e8fb01d6864ebfbf2389dcdbcab39b30289c"
_BUDGETS = {}


def _memory_class(workspace):
    """Import the actual verified clean source helper, not a copied approximation."""
    source = Path(workspace).resolve() / "src"
    sys.path.insert(0, str(source))
    from harbichess.training import cgroup_budget

    expected = source / "harbichess/training/cgroup_budget.py"
    if (
        Path(cgroup_budget.__file__).resolve() != expected
        or hashlib.sha256(expected.read_bytes()).hexdigest() != CGROUP_HELPER_SHA256
    ):
        raise ValueError("pinned clean-source cgroup helper differs")
    return cgroup_budget.CgroupMemoryBudget


def artifact_bytes(artifacts):
    return sum(p.stat().st_size for p in Path(artifacts).rglob("*") if p.is_file())


def guard(deadline, workspace, artifacts):
    if not time.time() < deadline <= HARD_END:
        raise TimeoutError("original deadline or hard08 exhausted")
    if shutil.disk_usage(workspace).free < 256 * 1024**2:
        raise RuntimeError("workspace256MiB floor")
    if artifact_bytes(artifacts) >= ARTIFACT_MAX_BYTES:
        raise RuntimeError("recursive artifact16MiB ceiling")
    key = Path(workspace).resolve()
    if key not in _BUDGETS:
        _BUDGETS[key] = _memory_class(workspace)(MEMORY_BUDGET_BYTES)
    return _BUDGETS[key].check()


def affinity(core):
    if core not in os.sched_getaffinity(0):
        raise ValueError("CPU outside allowed affinity")
    os.sched_setaffinity(0, {core})
