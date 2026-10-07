"""Prospective CPU owner resource guards; no experiment or model operations."""

import os
import shutil
import sys
import time
from pathlib import Path


def guard_factory(core, cpu, deadline, output=None, cap=None):
    os.sched_setaffinity(0, {cpu})
    sys.path.insert(0, str(Path(core) / 'src'))
    from harbichess.training.cgroup_budget import CgroupMemoryBudget

    budget = CgroupMemoryBudget(15 * 2**30)

    def guard():
        budget.check()
        if time.time() >= deadline:
            raise TimeoutError('original deadline includes validation/publication')
        if shutil.disk_usage('/workspace').free < 256 * 2**20:
            raise RuntimeError('workspace256MiB floor')
        if output is not None:
            usage = shutil.disk_usage(Path(output).parent)
            if usage.free < (cap or 0):
                raise RuntimeError('reserved output capacity unavailable')
            if Path(output).exists() and cap is not None:
                size = sum(p.stat().st_size for p in Path(output).rglob('*') if p.is_file())
                if size > cap:
                    raise RuntimeError('recursive registered artifact ceiling')
    guard()
    return guard
