"""Prospective6 admission witness: older45 already terminal; never signals old/new owners."""

import argparse
import json
from pathlib import Path

from own6_audit_support import publish, sha
from own6_schedule_v3 import verify_previous


def main():
    if not __debug__:
        raise RuntimeError("Required assertions disabled")
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--config-sha256", required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    assert sha(a.config) == a.config_sha256
    c = json.loads(a.config.read_text())
    assert c["schema"] == "own6-terminal45-admission-witness-v3"
    proof = verify_previous(c["previous45_completion_barrier"], sha)
    receipt = {
        "schema": c["schema"],
        "status": "prior45-incomplete-owned-compute-ended-no-active-old-latency",
        "config_sha256": a.config_sha256,
        "terminal_proof": proof,
        "old_or_new_owned_processes_signalled": [],
        "qualification_or_latency_substituted": False,
    }
    if not a.execute:
        print(json.dumps(receipt))
        return
    a.root.mkdir(parents=True, exist_ok=False)
    publish(a.root / "result.json", receipt)


if __name__ == "__main__":
    main()
