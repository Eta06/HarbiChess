"""Actual CUDA14 no-skip and real-e8 CLI qualification for the cache-only source."""

import argparse
import json
import os
import signal
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from qualify_cuda_owned import (
    DISK,
    E8,
    MEMORY,
    RUNTIME,
    check_source,
    guard,
    publish,
    run_owned,
    sha,
)


def cache_cases(path):
    cases = ET.parse(path).getroot().findall(".//testcase")
    if len(cases) != 14 or any(
        case.find(tag) is not None
        for case in cases for tag in ("skipped", "failure", "error")
    ):
        raise ValueError("Exactly14 actualCUDA unit passes, zero skips/failures/errors required")
    names = [case.attrib["name"] for case in cases]
    required = "test_old_and_cache_native_journals_all_rng_exact_and_fresh_resume[cuda:0]"
    if required not in names or sum("cuda:0" in name for name in names) != 2:
        raise ValueError("Both originalCUDAfreshresume and old/newCUDAcache proofs required")
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkout", "output", "python", "weights", "book", "config", "protocol",
                 "helper", "input-manifest", "baseline-checkout"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("source-commit", "helper-sha256", "test-sha256", "cache-test-sha256",
                 "input-manifest-sha256", "common-helper-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    a = parser.parse_args()
    started = time.time()
    if not __debug__ or not 0 < a.deadline_epoch - started <= 360:
        raise ValueError("One unoptimizedPython original360s ceiling; units180/CLI180")
    check_source(a.checkout, a.source_commit)
    check_source(a.baseline_checkout, "c93b79b8140b31fe1f4d0ca96adddd857991925b")
    bindings = {
        a.helper: a.helper_sha256,
        a.checkout / "tests/test_ownsearch_stage.py": a.test_sha256,
        a.checkout / "tests/test_ownsearch_cache8192.py": a.cache_test_sha256,
        a.input_manifest: a.input_manifest_sha256,
        Path(__file__).with_name("qualify_cuda_owned.py"): a.common_helper_sha256,
    }
    if any(sha(path) != value for path, value in bindings.items()):
        raise ValueError("Prospective helper/test/manifest bindings differ")
    inputs = {name: sha(getattr(a, name)) for name in ("weights", "book", "config", "protocol")}
    if inputs != json.loads(a.input_manifest.read_text()) or inputs["weights"] != E8:
        raise ValueError("All prospective qualification inputs must match real e8")
    cfg = json.loads(a.config.read_text())
    if (cfg["device"] != "cuda:0" or cfg["actors"]["temperature"] != 1
            or cfg["search"]["simulations"] != 16 or cfg["search"]["max_considered_actions"] != 4
            or cfg["search"]["block_plies"] != 8):
        raise ValueError("Requires original explicit actualCUDA/T1/16sim/max4/random8blocks")
    a.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(a.checkout / "src"), PYTHONOPTIMIZE="0",
               HARBICHESS_TEST_E8=str(a.weights),
               HARBICHESS_TEST_C93_SRC=str(a.baseline_checkout / "src"),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               CUBLAS_WORKSPACE_CONFIG=":4096:8")
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_PLUGINS", None)
    result = dict(
        schema="owned360-ownsearch-cache8192-CUDA-qualification-v2",
        status="failed-preserved", source_commit=a.source_commit,
        controller_sha256=sha(Path(__file__)), input_sha256=inputs,
        helper_test_manifest_sha256={str(k): v for k, v in bindings.items()},
        started_epoch=started, absolute_deadline_epoch=a.deadline_epoch,
        scope="Cache-only infrastructure; no formal/candidate/strength eligibility", phases=[],
    )

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Owned controller signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        units_deadline = min(a.deadline_epoch, started + 180)
        result["unit_absolute_deadline_epoch"] = units_deadline
        result["phases"].append(run_owned(
            "runtime", [str(a.python), "-c", RUNTIME], cwd=a.checkout,
            env=env, output=a.output, deadline=units_deadline,
        ))
        result["runtime"] = json.loads((a.output / "runtime.stdout.log").read_text())
        result["phases"].append(run_owned(
            "units", [str(a.python), "-m", "pytest", "-q", "tests/test_ownsearch_stage.py",
                      "tests/test_ownsearch_cache8192.py", "--junitxml",
                      str(a.output / "units.xml")],
            cwd=a.checkout, env=env, output=a.output, deadline=units_deadline,
        ))
        result["unit_cases"] = cache_cases(a.output / "units.xml")
        cli_deadline = min(a.deadline_epoch, time.time() + 180)
        result["cli_absolute_deadline_epoch"] = cli_deadline
        command = [str(a.python), str(a.helper), "--checkout", str(a.checkout),
                   "--output", str(a.output / "cli")]
        for name in ("weights", "book", "config", "protocol"):
            command += ["--" + name, str(getattr(a, name))]
        command += ["--source-commit", a.source_commit, "--deadline-epoch", str(cli_deadline)]
        command += ["--memory-max-bytes", str(MEMORY), "--disk-min-free-bytes", str(DISK)]
        result["phases"].append(run_owned(
            "cli", command, cwd=a.checkout, env=env, output=a.output, deadline=cli_deadline,
        ))
        receipt = a.output / "cli/result.json"
        result["cli_result"] = json.loads(receipt.read_text())
        result["cli_result_sha256"] = sha(receipt)
        cli = result["cli_result"]
        if (cli["status"] != "pass"
                or cli["schema"] != "actual-CUDA-sparse-ownsearch-CLI-qualification-v1"
                or cli["source_commit"] != a.source_commit or cli["input_sha256"] != inputs
                or not cli["both_final_full_native_freshprocess_strictload"]
                or len(cli["byte_exact_artifacts"]) != 8):
            raise ValueError("Real-e8 clean CLI/all-six-payloads replay did not pass")
        guard(a.deadline_epoch, a.output)
        check_source(a.checkout, a.source_commit)
        check_source(a.baseline_checkout, "c93b79b8140b31fe1f4d0ca96adddd857991925b")
        if inputs != {name: sha(getattr(a, name)) for name in inputs}:
            raise ValueError("Frozen inputs changed during qualification")
        if any(sha(path) != value for path, value in bindings.items()):
            raise ValueError("Frozen helper/test/manifest changed during qualification")
        result["status"] = "pass-all14-no-skip-actualCUDA-and-e8-cleanCLI-cache-equivalence"
    except BaseException as exc:
        result["error"] = repr(exc)
        raise
    finally:
        result["finished_epoch"] = time.time()
        result["whole_seconds"] = result["finished_epoch"] - started
        publish(a.output / "result.json", result)


if __name__ == "__main__":
    main()
