"""Prospective still-unregistered6 scheduling, with real failed45 terminal release."""

import json
from pathlib import Path

AUDIT_CUTOFF = 1791170700  # Actual2026-10-05 03:25UTC
LATEST_LATENCY_START = 1791172800  # Actual04:00UTC
END = 1791180000


def audit_deadline(first, whole_seconds):
    assert whole_seconds == 9000
    return min(first + whole_seconds, AUDIT_CUTOFF)


def validate_clock(registration, first, training_deadline, now):
    assert (
        registration["scheduling_version"]
        == "prospective-own6-terminal45-scheduling-v3"
    )
    assert registration["absolute_audit_cutoff_epoch"] == AUDIT_CUTOFF
    assert registration["latest_latency_start_epoch"] == LATEST_LATENCY_START
    assert registration["whole_training_seconds_per_seed"] == 6000
    assert registration["whole_audit_seconds_from_originalfirstclock"] == 9000
    assert (
        registration["earliest_training_epoch"]
        <= first
        <= now
        < training_deadline
        < END
    )
    assert training_deadline == first + 6000
    assert now < audit_deadline(first, 9000) < LATEST_LATENCY_START < END


def verify_previous(info, sha):
    from own6_terminal45_release import validate_barrier

    path = Path(info["terminal45_barrier_config"])
    assert sha(path) == info["terminal45_barrier_config_sha256"]
    config = json.loads(path.read_text())
    return validate_barrier(config)


def phase_deadline(started, ceiling):
    # Maximum phase ceiling unchanged; physical lease is an additional cutoff.
    assert ceiling > 0 and started < END - 1
    return min(started + ceiling, END - 1)


def bind_argument_deadline(arguments, deadline):
    values = list(map(str, arguments))
    if "--deadline-epoch" in values:
        assert values.count("--deadline-epoch") == 1
        index = values.index("--deadline-epoch") + 1
        values[index] = str(min(float(values[index]), deadline))
    return values
