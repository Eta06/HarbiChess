from pathlib import Path

import pytest

from harbichess.training.torch_online_run import _memory_metric, _memory_usage


def test_cgroup_metric_reads_current_and_does_not_use_host_usage(monkeypatch):
    monkeypatch.setattr(Path, "is_file", lambda self: str(self) == "/sys/fs/cgroup/memory.current")
    monkeypatch.setattr(Path, "read_text", lambda self: "123456\n")
    assert _memory_metric() == "cgroup-v2-current"
    assert _memory_usage(_memory_metric()) == 123456


def test_missing_cgroup_uses_explicit_system_used_including_cache(monkeypatch):
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    monkeypatch.setattr(Path, "read_text", lambda self:
                        "MemTotal: 1000000 kB\nMemFree: 200000 kB\nMemAvailable: 700000 kB\n")
    assert _memory_metric() == "system-used-including-cache"
    assert _memory_usage(_memory_metric()) == 800000 * 1024
    with pytest.raises(ValueError, match="unknown registered"):
        _memory_usage("invented")
