"""One quarter-LR128x256 cache-source epoch and audit within one absolute900s ceiling."""

import argparse
import json
import os
import signal
import time
import traceback
from pathlib import Path

from qualify_cuda_owned import (
    DISK,
    E8,
    MEMORY,
    check_source,
    guard,
    publish,
    run_owned,
    sha,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "checkout",
        "output",
        "python",
        "weights",
        "book",
        "config",
        "protocol",
        "audit-helper",
        "input-manifest",
        "qualification-receipt",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "source-commit",
        "audit-helper-sha256",
        "common-helper-sha256",
        "input-manifest-sha256",
        "qualification-receipt-sha256",
    ):
        p.add_argument("--" + name, required=True)
    p.add_argument("--deadline-epoch", type=float, required=True)
    a = p.parse_args()
    started = time.time()
    if not 0 < a.deadline_epoch - started <= 900:
        raise ValueError("requires one absolute900s whole profile/audit budget")
    if a.output.exists():
        raise FileExistsError(a.output)
    check_source(a.checkout, a.source_commit)
    common = Path(__file__).with_name("qualify_cuda_owned.py")
    bound = {
        a.audit_helper: a.audit_helper_sha256,
        common: a.common_helper_sha256,
        a.input_manifest: a.input_manifest_sha256,
        a.qualification_receipt: a.qualification_receipt_sha256,
    }
    if any(sha(path) != value for path, value in bound.items()):
        raise ValueError("frozen helper/qualification/inputmanifest hash mismatch")
    qualification = json.loads(a.qualification_receipt.read_text())
    if qualification["source_commit"] != a.source_commit or qualification["status"] != (
        "pass-all14-no-skip-actualCUDA-and-e8-cleanCLI-cache-equivalence"
    ):
        raise ValueError("actual CUDA14+cleanCLI cache qualification gate missing")
    inputs = {k: sha(getattr(a, k)) for k in ("weights", "book", "config", "protocol")}
    if inputs != json.loads(a.input_manifest.read_text()) or inputs["weights"] != E8:
        raise ValueError("ALLprofile inputs must match prospective hash manifest")
    cfg = json.loads(a.config.read_text())
    if not (
        cfg["device"] == "cuda:0"
        and cfg["learning_rate"] == 0.000025
        and cfg["objective"]["behavior_kl_stop"] == 0.02
        and cfg["epoch_steps"] == 256
        and cfg["actors"]["games"] == 128
        and cfg["actors"]["max_additional_plies"] == 256
        and cfg["actors"]["temperature"] == 1
        and cfg["search"]["simulations"] == 16
        and cfg["search"]["max_considered_actions"] == 4
        and cfg["search"]["block_plies"] == 8
        and cfg["schedule"]["passes"] == 4
        and cfg["schedule"]["minibatch_size"] == 1024
    ):
        raise ValueError("not the registered full production-shape profile")
    a.output.mkdir(parents=True)
    env = dict(
        os.environ,
        PYTHONPATH=str(a.checkout.resolve() / "src"),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )
    result = dict(
        schema="owned900-ownsearch-cache-quarterLR-development-profile-v2",
        status="failed-preserved",
        source_commit=a.source_commit,
        controller_sha256=sha(Path(__file__)),
        frozen_helpers_and_receipts={str(k): v for k, v in bound.items()},
        input_sha256=inputs,
        started_epoch=started,
        absolute_deadline_epoch=a.deadline_epoch,
        memory_max_bytes=MEMORY,
        disk_min_free_bytes=DISK,
        phases=[],
        scope="ONE development epoch, no formal/candidate/strength eligibility",
    )

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"owned controller signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        command = [
            str(a.python),
            "-m",
            "harbichess.training.torch_ownsearch_run",
            str((a.output / "run").resolve()),
        ]
        for k in ("weights", "book", "config", "protocol"):
            command += ["--" + k, str(getattr(a, k).resolve())]
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
                "one-epoch-cli",
                command,
                cwd=a.checkout,
                env=env,
                output=a.output,
                deadline=a.deadline_epoch,
            )
        )
        # Preserve the CLI receipt before independent audit can fail or time out.
        invocation = sorted((a.output / "run").glob("invocation-*-result.json"))
        if len(invocation) != 1:
            raise ValueError("one and only one CLI invocation required")
        cli = json.loads(invocation[0].read_text())
        result["cli_result"] = cli
        result["cli_result_sha256"] = sha(invocation[0])
        if (
            cli["status"] != "completed"
            or cli["epoch"] != 1
            or not cli["closed_boundary"]
        ):
            raise ValueError("CLI did not publish a complete one-epoch boundary")
        command = [
            str(a.python),
            str(a.audit_helper.resolve()),
            "--checkout",
            str(a.checkout.resolve()),
            "--run",
            str((a.output / "run").resolve()),
            "--output",
            str((a.output / "independent-audit.json").resolve()),
            "--source-commit",
            a.source_commit,
            "--deadline-epoch",
            str(a.deadline_epoch),
        ]
        for k in ("weights", "book", "config", "protocol"):
            command += ["--" + k, str(getattr(a, k).resolve())]
        result["phases"].append(
            run_owned(
                "independent-audit",
                command,
                cwd=a.checkout,
                env=env,
                output=a.output,
                deadline=a.deadline_epoch,
            )
        )
        audit = json.loads((a.output / "independent-audit.json").read_text())
        if audit["status"] != "pass" or audit["source_commit"] != a.source_commit:
            raise ValueError("independent one-epoch audit did not pass")
        result["audit_result"] = audit
        result["audit_result_sha256"] = sha(a.output / "independent-audit.json")
        check_source(a.checkout, a.source_commit)
        if inputs != {k: sha(getattr(a, k)) for k in inputs} or any(
            sha(path) != value for path, value in bound.items()
        ):
            raise ValueError("immutable inputs/helpers changed during profile")
        guard(a.deadline_epoch, a.output)
        result["status"] = "pass-one-development-epoch-and-readonly-audit"
    except BaseException as exc:
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        result["finished_epoch"] = time.time()
        result["whole_seconds"] = result["finished_epoch"] - started
        publish(a.output / "result.json", result)


if __name__ == "__main__":
    main()
