"""No nonexistent5 latency/finals: verify actual incomplete45 owned termination."""

from own7_schedule_v3 import verify_previous


def wait_previous(config, wait_json, sha):
    del wait_json
    return verify_previous(config["previous45_completion_barrier"], sha)
