"""Owned CUDA infrastructure qualification: units180s, CLI180s; no strength eligibility."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

E8 = "e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03"
MEMORY = 64 * 1024**3
DISK = 8 * 1024**3


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def publish(path, data):
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("x") as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def memory_used():
    path = Path("/sys/fs/cgroup/memory.current")
    if path.exists():
        return int(path.read_text())
    fields = {
        line.split(":")[0]: int(line.split(":")[1].strip().split()[0]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if ":" in line
    }
    return fields["MemTotal"] - fields["MemAvailable"]


def guard(deadline, output):
    if time.time() >= deadline:
        raise TimeoutError("original wholedeadline exhausted")
    if memory_used() > MEMORY or shutil.disk_usage(output).free < DISK:
        raise RuntimeError("registered64GiB/8GiB resource ceiling")


def kill_group(child):
    with contextlib.suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGKILL)
    child.wait()


def run_owned(name, command, *, cwd, env, output, deadline):
    guard(deadline, output)
    record = dict(
        phase=name,
        command=command,
        started_epoch=time.time(),
        absolute_deadline_epoch=deadline,
        status="failed",
    )
    child = None
    try:
        with (
            (output / (name + ".stdout.log")).open("x") as out,
            (output / (name + ".stderr.log")).open("x") as err,
        ):
            child = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
            record["owned_pid"] = record["owned_process_group"] = child.pid
            publish(
                output / (name + ".owner.json"),
                dict(
                    pid=child.pid,
                    owned_process_group=child.pid,
                    deadline_epoch=deadline,
                ),
            )
            while child.poll() is None:
                guard(deadline, output)
                time.sleep(min(0.1, max(0.001, deadline - time.time())))
        record["returncode"] = child.returncode
        if child.returncode:
            raise RuntimeError(f"{name} returned {child.returncode}")
        guard(deadline, output)
        record["status"] = "pass"
    except BaseException as exc:
        record["error"] = repr(exc)
        raise
    finally:
        if child is not None:
            # Clean descendants even if their direct parent exited or failed.
            kill_group(child)
            record["returncode"] = child.returncode
        record["finished_epoch"] = time.time()
        record["whole_seconds"] = record["finished_epoch"] - record["started_epoch"]
        publish(output / (name + ".invocation.json"), record)
    return record


def check_source(checkout, source):
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
        != source
        or subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=checkout, text=True
        ).strip()
    ):
        raise ValueError("requires exact CLEANnew ownsearch producer")


def check_junit(path, expected=16):
    root = ET.parse(path).getroot()
    cases = root.findall(".//testcase")
    if len(cases) != expected or any(
        case.find(tag) is not None for case in cases for tag in ("skipped", "failure", "error")
    ):
        raise ValueError(
            "actual CUDA unit suite requires exactly16passes zero skips/failures/errors"
        )
    if not any("cuda:0" in case.attrib["name"] for case in cases):
        raise ValueError("CUDA new-process resume branch was not executed")
    if not any(
        "cuda:0" in case.attrib["name"] and "rejected_pass_restores" in case.attrib["name"]
        for case in cases
    ):
        raise ValueError("ActualCUDA rollback branch missing")
    return [case.attrib["name"] for case in cases]


RUNTIME = """
import json,torch
from harbichess.training.torch_online_learner import _prepare_device
from harbichess.training.torch_online_checkpoint import _runtime
assert torch.cuda.is_available() and 'A100' in torch.cuda.get_device_name(0)
_prepare_device('cuda:0')
print(json.dumps(_runtime('cuda:0'),sort_keys=True))
"""


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
        "helper",
        "input-manifest",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "source-commit",
        "helper-sha256",
        "test-sha256",
        "input-manifest-sha256",
    ):
        p.add_argument("--" + name, required=True)
    p.add_argument("--deadline-epoch", type=float, required=True)
    a = p.parse_args()
    started = time.time()
    if not 0 < a.deadline_epoch - started <= 360:
        raise ValueError("qualification must fit absolute360seconds, each phase capped180")
    if a.output.exists():
        raise FileExistsError(a.output)
    check_source(a.checkout, a.source_commit)
    test = a.checkout / "tests/test_search_acting_stage.py"
    if (
        sha(a.helper) != a.helper_sha256
        or sha(test) != a.test_sha256
        or sha(a.input_manifest) != a.input_manifest_sha256
    ):
        raise ValueError("frozen helper/test/input-manifest mismatch")
    expected = json.loads(a.input_manifest.read_text())
    before = {name: sha(getattr(a, name)) for name in ("weights", "book", "config", "protocol")}
    if expected != before or before["weights"] != E8:
        raise ValueError("requires ALLfrozen real-e8 qualification input hashes")
    cfg = json.loads(a.config.read_text())
    if (
        cfg["device"] != "cuda:0"
        or cfg["actors"]["temperature"] != 1
        or cfg["search"]["simulations"] != 16
        or cfg["search"]["max_considered_actions"] != 4
        or cfg["search"]["block_plies"] != 8
    ):
        raise ValueError("qualification requires actualCUDA/T1/random8block/16sim/max4")
    a.output.mkdir(parents=True)
    env = dict(
        os.environ,
        PYTHONPATH=str(a.checkout.resolve() / "src"),
        HARBICHESS_TEST_E8=str(a.weights.resolve()),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_PLUGINS", None)
    result = dict(
        schema="owned360-actual-CUDA-search-acting-qualification-v2",
        status="failed-preserved",
        source_commit=a.source_commit,
        controller_sha256=sha(Path(__file__)),
        helper_sha256=a.helper_sha256,
        test_sha256=a.test_sha256,
        input_manifest_sha256=a.input_manifest_sha256,
        input_sha256=before,
        started_epoch=started,
        absolute_deadline_epoch=a.deadline_epoch,
        memory_max_bytes=MEMORY,
        disk_min_free_bytes=DISK,
        scope="Infrastructure only; no production/candidate/strength eligibility",
        phases=[],
    )

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"owned controller signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        unit_deadline = min(a.deadline_epoch, started + 180)
        result["unit_absolute_deadline_epoch"] = unit_deadline
        result["phases"].append(
            run_owned(
                "runtime",
                [str(a.python), "-c", RUNTIME],
                cwd=a.checkout,
                env=env,
                output=a.output,
                deadline=unit_deadline,
            )
        )
        result["runtime"] = json.loads((a.output / "runtime.stdout.log").read_text())
        result["phases"].append(
            run_owned(
                "units",
                [
                    str(a.python),
                    "-m",
                    "pytest",
                    "-q",
                    "tests/test_search_acting_stage.py",
                    "--junitxml",
                    str((a.output / "units.xml").resolve()),
                ],
                cwd=a.checkout,
                env=env,
                output=a.output,
                deadline=unit_deadline,
            )
        )
        result["unit_cases"] = check_junit(a.output / "units.xml")
        cli_deadline = min(a.deadline_epoch, time.time() + 180)
        result["cli_absolute_deadline_epoch"] = cli_deadline
        result["phases"].append(
            run_owned(
                "cli",
                [
                    str(a.python),
                    str(a.helper.resolve()),
                    "--checkout",
                    str(a.checkout.resolve()),
                    "--output",
                    str((a.output / "cli").resolve()),
                    "--weights",
                    str(a.weights.resolve()),
                    "--book",
                    str(a.book.resolve()),
                    "--config",
                    str(a.config.resolve()),
                    "--protocol",
                    str(a.protocol.resolve()),
                    "--source-commit",
                    a.source_commit,
                    "--deadline-epoch",
                    str(cli_deadline),
                    "--memory-max-bytes",
                    str(MEMORY),
                    "--disk-min-free-bytes",
                    str(DISK),
                ],
                cwd=a.checkout,
                env=env,
                output=a.output,
                deadline=cli_deadline,
            )
        )
        cli = json.loads((a.output / "cli/result.json").read_text())
        if (
            cli["schema"] != "actual-CUDA-search-acting-CLI-qualification-v2"
            or cli["status"] != "pass"
            or cli["source_commit"] != a.source_commit
            or cli["input_sha256"] != before
            or not cli["both_final_full_native_freshprocess_strictload"]
            or len(cli["byte_exact_artifacts"]) != 8
        ):
            raise ValueError("cleanCLI exact native/journal/strictload receipt mismatch")
        result["cli_result_sha256"] = sha(a.output / "cli/result.json")
        result["cli_result"] = cli
        guard(a.deadline_epoch, a.output)
        check_source(a.checkout, a.source_commit)
        if (
            before != {n: sha(getattr(a, n)) for n in before}
            or sha(a.helper) != a.helper_sha256
            or sha(test) != a.test_sha256
            or sha(a.input_manifest) != a.input_manifest_sha256
        ):
            raise ValueError("source/input/helper changed during qualification")
        result["status"] = (
            "pass-all16-no-skip-actualCUDA-and-e8-cleanCLI-search-acting-v2-native-replay"
        )
        result["duplicate_CLI_fresh_presentations"] = cli[
            "duplicate_qualification_fresh_transitions"
        ]
        result["unit_duplicate_cost_scope"] = (
            "Unit rollouts/gradients separately charged infrastructure; not production fresh count"
        )
    except BaseException as exc:
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        result["finished_epoch"] = time.time()
        result["whole_seconds"] = result["finished_epoch"] - started
        publish(a.output / "result.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
