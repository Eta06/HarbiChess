"""One real128x256 v2 development epoch plus chronological audit within ORIGINAL900."""

import argparse
import json
import os
import signal
import time
from pathlib import Path

from own6_adapter_controls import SOURCE
from own6_audit_support import (
    DISK,
    E8,
    MEMORY,
    check_source,
    guard,
    publish,
    run_owned,
    sha,
)
from own6_infrastructure_evidence import check_cli_receipt, check_unit34_receipt

AUDIT_STATUS = (
    "pass-actualCUDA-search-acting-v2-E1-all-data-FIRST8-LAST8-"
    "original128-masks-raw-packets-and-mutations"
)


def main():
    p = argparse.ArgumentParser()
    for name in (
        "checkout",
        "output",
        "python",
        "weights",
        "book",
        "config",
        "protocol",
        "input-manifest",
        "qualification-receipt",
        "qualification-run",
        "unit34-receipt",
        "audit-helper",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "source-commit",
        "input-manifest-sha256",
        "qualification-receipt-sha256",
        "unit34-receipt-sha256",
        "audit-helper-sha256",
        "audit-core-sha256",
        "common-helper-sha256",
        "adapter-controls-sha256",
        "infrastructure-evidence-sha256",
    ):
        p.add_argument("--" + name, required=True)
    p.add_argument("--deadline-epoch", type=float, required=True)
    a = p.parse_args()
    started = time.time()
    assert 0 < a.deadline_epoch - started <= 900
    assert a.source_commit == SOURCE
    check_source(a.checkout, a.source_commit)
    common = Path(__file__).with_name("own6_audit_support.py")
    core = Path(__file__).with_name("own6_audit_core.py")
    frozen = {
        a.input_manifest: a.input_manifest_sha256,
        a.qualification_receipt: a.qualification_receipt_sha256,
        a.unit34_receipt: a.unit34_receipt_sha256,
        Path(__file__).with_name(
            "own6_infrastructure_evidence.py"
        ): a.infrastructure_evidence_sha256,
        a.audit_helper: a.audit_helper_sha256,
        core: a.audit_core_sha256,
        common: a.common_helper_sha256,
        Path(__file__).with_name("own6_adapter_controls.py"): a.adapter_controls_sha256,
    }
    for path, digest in frozen.items():
        assert sha(path) == digest
    qualification = json.loads(a.qualification_receipt.read_text())
    check_cli_receipt(qualification, a.qualification_run, sha)
    check_unit34_receipt(json.loads(a.unit34_receipt.read_text()))
    inputs = {name: sha(getattr(a, name)) for name in ("weights", "book", "config", "protocol")}
    assert inputs == json.loads(a.input_manifest.read_text()) and inputs["weights"] == E8
    cfg = json.loads(a.config.read_text())
    assert cfg["actors"]["games"] == 128 and cfg["actors"]["temperature"] == 1
    assert cfg["actors"]["max_additional_plies"] == cfg["epoch_steps"] == 256
    assert cfg["device"] == "cuda:0" and cfg["learning_rate"] == 0.000025
    assert cfg["objective"]["behavior_kl_stop"] == 0.02
    assert cfg["search"]["block_plies"] == 8 and cfg["search"]["simulations"] == 16
    assert cfg["search"]["max_considered_actions"] == 4
    assert cfg["schedule"]["passes"] == 4 and cfg["schedule"]["minibatch_size"] == 1024
    a.output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "owned900-search-acting-v2-fullshape-development-v1",
        "status": "failed-preserved",
        "source_commit": a.source_commit,
        "started_epoch": started,
        "absolute_deadline_epoch": a.deadline_epoch,
        "controller_sha256": sha(__file__),
        "frozen_helper_sha256": {str(p): v for p, v in frozen.items()},
        "input_sha256": inputs,
        "neural_witness_K": 8,
        "phases": [],
        "scope": "E1 DEVELOPMENT ONLY; no formal6/candidate/strength eligibility",
    }
    env = {
        **os.environ,
        "PYTHONPATH": str(a.checkout.resolve() / "src"),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Owned v2 profile interruption {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        command = [
            str(a.python),
            "-m",
            "harbichess.training.torch_search_acting_run",
            str(a.output / "run"),
        ]
        for name in ("weights", "book", "config", "protocol"):
            command += ["--" + name, str(getattr(a, name).resolve())]
        command += [
            "--source-commit",
            a.source_commit,
            "--max-epochs",
            "1",
            "--checkpoint-interval",
            "1",
            "--deadline-epoch",
            str(a.deadline_epoch),
            "--memory-max-bytes",
            str(MEMORY),
            "--disk-min-free-bytes",
            str(DISK),
        ]
        result["phases"].append(
            run_owned(
                "one-epoch-CLI",
                command,
                cwd=a.checkout,
                env=env,
                output=a.output,
                deadline=a.deadline_epoch,
            )
        )
        invocations = sorted((a.output / "run").glob("invocation-*-result.json"))
        assert len(invocations) == 1
        cli = json.loads(invocations[0].read_text())
        assert cli["status"] == "completed" and cli["epoch"] == 1 and cli["closed_boundary"]
        result["CLI_receipt_sha256"] = sha(invocations[0])
        result["CLI_result"] = cli
        # This external manifest has its own SHA. Original four native inputs stay untouched.
        manifest = {
            "schema": "ufuk-search-acting-v2-E1-auditor-development-v1",
            "scope": "existing-search-acting-development-E1-not-formal6",
            "run": str((a.output / "run").resolve()),
            "checkout": str(a.checkout.resolve()),
            "source_commit": a.source_commit,
            "neural_witness_K": 8,
            "started_epoch": started,
            "whole_seconds": a.deadline_epoch - started,
            "original_profile_deadline_epoch": a.deadline_epoch,
            "frozen_config": cfg,
            "helper_sha256": {
                a.audit_helper.name: a.audit_helper_sha256,
                core.name: a.audit_core_sha256,
                common.name: a.common_helper_sha256,
                "own6_adapter_controls.py": frozen[
                    Path(__file__).with_name("own6_adapter_controls.py")
                ],
            },
            "inputs": {
                key: {"path": str(getattr(a, name).resolve()), "sha256": inputs[name]}
                for key, name in [
                    ("initial_weights", "weights"),
                    ("book", "book"),
                    ("experiment_config", "config"),
                    ("protocol", "protocol"),
                ]
            },
        }
        manifest_path = a.output / "independent-audit-manifest.json"
        publish(manifest_path, manifest)
        command = [
            str(a.python),
            str(a.audit_helper.resolve()),
            "--manifest",
            str(manifest_path.resolve()),
            "--manifest-sha256",
            sha(manifest_path),
            "--output",
            str(a.output / "independent-audit.json"),
            "--deadline-epoch",
            str(a.deadline_epoch),
        ]
        result["phases"].append(
            run_owned(
                "independent-chronological-audit",
                command,
                cwd=a.checkout,
                env=env,
                output=a.output,
                deadline=a.deadline_epoch,
            )
        )
        audit = json.loads((a.output / "independent-audit.json").read_text())
        assert audit["status"] == AUDIT_STATUS and audit["source_commit"] == a.source_commit
        assert audit["qualified_production_core_sha256"] == a.audit_core_sha256
        assert audit["audit_report"]["epoch"] == 1
        assert audit["audit_report"]["raw_actor_replayed"] == 32768
        assert audit["audit_report"]["raw_actor_packet_roots_verified"] == 18
        assert audit["audit_report"]["neural_witness_K"] == 8
        assert all(audit["targeted_actual_data_mutations_rejected"].values())
        assert audit["optimizer_updates_performed_by_qualification"] == 0
        result["audit_result"] = audit
        result["audit_result_sha256"] = sha(a.output / "independent-audit.json")
        for path, digest in frozen.items():
            assert sha(path) == digest
        assert inputs == {name: sha(getattr(a, name)) for name in inputs}
        check_source(a.checkout, a.source_commit)
        guard(a.deadline_epoch, a.output)
        result["status"] = "pass-one-search-acting-v2-development-epoch-and-fullchronological-audit"
    except BaseException as exc:
        result["error"] = repr(exc)
        raise
    finally:
        result["finished_epoch"] = time.time()
        result["whole_seconds"] = time.time() - started
        publish(a.output / "result.json", result)


if __name__ == "__main__":
    main()
