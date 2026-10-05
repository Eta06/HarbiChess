"""Explicit control-only failed4 dependency repair around immutable original5 coordinator."""

import argparse
import importlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from failed4_dependency import (
    identity_live,
    proc_table,
    track_descendants,
    validate_terminal,
    wait_failed4,
)


def rewrite_argv(argv, supplement):
    argv = list(map(str, argv))
    if len(argv) < 2 or Path(argv[1]).name not in (
        "own5_strength_analysis.py",
        "own5_all_gates.py",
    ):
        return argv
    argv[1] = str(Path(supplement["analysis_v3_directory"]) / Path(argv[1]).name)
    argv += [
        "--analysis-supplement",
        supplement["analysis_v3_supplement"],
        "--analysis-supplement-sha256",
        supplement["analysis_v3_supplement_sha256"],
        "--original-helpers",
        supplement["original_helpers"],
        "--original-registration",
        supplement["original_registration"],
        "--previous-analysis-repair",
        supplement["analysis_v2_supplement"],
    ]
    return argv


def extra_compute_busy(process_commands, current_pid):
    patterns = tuple(
        f"own{slot}_{name}"
        for slot in (6, 7)
        for name in (
            "training_controller",
            "audit_controller",
            "baseline",
            "full_audit",
            "fresh_cli_replay",
            "parity",
            "final_arms",
            "latency",
        )
    )
    return sorted(
        pid
        for pid, command in process_commands.items()
        if pid != current_pid and any(word in command for word in patterns)
    )


def process_commands():
    commands = {}
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            commands[int(path.parent.name)] = (
                path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return commands


def main():
    if not __debug__:
        raise RuntimeError("Required assertions disabled")
    parser = argparse.ArgumentParser()
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--supplement-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    s = json.loads(args.supplement.read_text())
    helpers = Path(s["original_helpers"])
    sys.path.insert(0, str(helpers))
    original = importlib.import_module("own5_posttraining")
    cohort_module = importlib.import_module("own45_cohort")
    sha = original.sha
    assert Path(original.__file__).resolve() == (helpers / "own5_posttraining.py").resolve()
    assert Path(cohort_module.__file__).resolve() == (helpers / "own45_cohort.py").resolve()
    assert sha(args.supplement) == args.supplement_sha256
    assert s["schema"] == "own5-failed4-dependency-analysis-v3-control-supplement-v1"
    assert s["status"] == "registered-before-method5-final-outcomes"
    assert s["qualification_ledger_slot"] == 5
    for filename, digest in s["original_helper_sha256"].items():
        assert sha(helpers / filename) == digest
    for filename, digest in s["supplemental_helper_sha256"].items():
        assert sha(Path(__file__).parent / filename) == digest
    assert sha(s["original_config"]) == s["original_config_sha256"]
    old = json.loads(Path(s["original_config"]).read_text())
    assert sha(s["new_config"]) == s["new_config_sha256"]
    new = json.loads(Path(s["new_config"]).read_text())
    assert {k: v for k, v in old.items() if k != "root"} == {
        k: v for k, v in new.items() if k != "root"
    }
    assert old["root"] != new["root"] and not Path(new["root"]).exists()
    assert sha(new["qualification_config"]) == new["qualification_config_sha256"]
    assert new["qualification_config_sha256"] == s["original_qualification_config_sha256"]
    assert sha(s["original_registration"]) == s["original_registration_sha256"]
    for filename, digest in s["analysis_v3_helper_sha256"].items():
        assert sha(Path(s["analysis_v3_directory"]) / filename) == digest
    assert sha(s["analysis_v3_supplement"]) == s["analysis_v3_supplement_sha256"]
    assert sha(s["analysis_v2_supplement"]) == s["analysis_v2_supplement_sha256"]
    assert not identity_live(s["stopped_original5_post_identity"], proc_table())
    assert {p.name for p in Path(old["root"]).iterdir()} <= {"failure.json"}
    original_cohort = cohort_module.bind(old, 5, sha(helpers / "own5_posttraining.py"), sha)
    assert original_cohort is not None
    deadline = original_cohort["completion_deadline_epoch"]
    assert time.time() < deadline
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "validated-control-plan-only-no-jobs",
                    "original_qualification_config_sha256": new["qualification_config_sha256"],
                }
            )
        )
        return

    assert sha(s["failed4_terminal_receipt"]) == s["failed4_terminal_receipt_sha256"]
    terminal = json.loads(Path(s["failed4_terminal_receipt"]).read_text())
    validate_terminal(terminal, original_cohort, sha)
    tracked, monitoring_errors = {}, []
    monitoring_stop = threading.Event()

    def monitor_failed4():
        try:
            while not monitoring_stop.is_set():
                assert sha(s["failed4_terminal_receipt"]) == s["failed4_terminal_receipt_sha256"]
                track_descendants(terminal, proc_table(), tracked)
                if time.time() >= deadline:
                    raise TimeoutError("Original cohort deadline reached while tracking4")
                monitoring_stop.wait(0.5)
        except BaseException as error:
            monitoring_errors.append(error)

    monitor = threading.Thread(target=monitor_failed4, daemon=True)
    monitor.start()
    bind_before = cohort_module.bind
    verify_before = cohort_module.verify_receipt
    before_before = cohort_module.before_latency
    after_before = cohort_module.after_latency
    popen_before = subprocess.Popen
    publish_before = original.publish
    quiescent_before = original.quiescent

    def bind(config, slot, coordinator_sha, sha_fn):
        assert slot == 5 and config == new
        old_bound = bind_before(old, slot, coordinator_sha, sha_fn)
        result = json.loads(json.dumps(old_bound))
        cohort_module.member(result, 5)["root"] = new["root"]
        return result

    def before_latency(cohort, slot, wait_json, sha_fn):
        assert slot == 5
        monitoring_stop.set()
        monitor.join(1)
        assert not monitor.is_alive()
        if monitoring_errors:
            raise monitoring_errors[0]
        proof = wait_failed4(
            s["failed4_terminal_receipt"],
            s["failed4_terminal_receipt_sha256"],
            original_cohort,
            sha_fn,
            deadline,
            tracked_initial=tracked,
        )
        verify_before(cohort, 5, "ready", wait_json, sha_fn)
        publish_before(
            Path(new["root"]) / "failed4-dependency-release.json",
            {
                **proof,
                "control_supplement_sha256": args.supplement_sha256,
                "original_cohort_sha256": old["cohort"]["manifest_sha256"],
            },
        )

    def after_latency(cohort, slot, root, wait_json, sha_fn, publish):
        assert slot == 5
        result = wait_json(Path(root) / "latency-process-result.json", deadline)
        assert result["returncode"] == 0 and result["finished_epoch"] <= result["deadline_epoch"]
        owner = cohort_module.member(cohort, 5)
        publish(
            Path(root) / "cohort-latency-complete.json",
            {
                "schema": "own45-latency-owner-completion-receipt-v1",
                "slot": 5,
                "source_commit": cohort_module.SOURCES[5],
                "coordinator_sha256": owner["coordinator_sha256"],
                "process_receipt_sha256": {
                    "latency-process-result.json": sha_fn(
                        Path(root) / "latency-process-result.json"
                    )
                },
                "finished_epoch": time.time(),
            },
        )
        verify_before(cohort, 5, "latency", wait_json, sha_fn)

    def publish(path, value):
        if str(path).endswith("-command.json") and "argv" in value:
            value = {**value, "argv": rewrite_argv(value["argv"], s)}
        return publish_before(path, value)

    def popen(argv, *positional, **kwargs):
        return popen_before(rewrite_argv(argv, s), *positional, **kwargs)

    def quiescent(deadline):
        # Same original60s absolute ceiling. Additional registered/future compute
        # owners close gaps between child GPU/arena processes; waiting coordinators
        # and read-only guardians are excluded.
        while True:
            quiescent_before(deadline)
            if time.time() >= deadline:
                raise TimeoutError("Original60s quiescence ceiling exhausted")
            if not extra_compute_busy(process_commands(), os.getpid()):
                return
            time.sleep(0.5)

    original.quiescent = quiescent
    cohort_module.bind = bind
    cohort_module.before_latency = before_latency
    cohort_module.after_latency = after_latency
    original.publish = publish
    subprocess.Popen = popen
    sys.argv = [
        str(helpers / "own5_posttraining.py"),
        "--config",
        s["new_config"],
        "--config-sha256",
        s["new_config_sha256"],
        "--execute",
    ]
    publish_before(
        Path(s["control_activation_receipt"]),
        {
            "schema": "own5-supplemental-scheduling-control-activation-v1",
            "control_supplement_sha256": args.supplement_sha256,
            "actual_wrapper_sha256": sha(Path(__file__)),
            "original_coordinator_sha256": sha(helpers / "own5_posttraining.py"),
            "original_qualification_config_sha256": new["qualification_config_sha256"],
            "original_config_sha256": s["original_config_sha256"],
            "new_config_sha256": s["new_config_sha256"],
            "old_post_root_preserved": old["root"],
            "new_post_root": new["root"],
            "observed_epoch": time.time(),
            "no_original_deadline_or_gate_change": True,
        },
    )
    try:
        original.main()
    finally:
        monitoring_stop.set()
        monitor.join(1)
        cohort_module.bind = bind_before
        cohort_module.before_latency = before_before
        cohort_module.after_latency = after_before
        subprocess.Popen = popen_before
        original.publish = publish_before
        original.quiescent = quiescent_before


if __name__ == "__main__":
    main()
