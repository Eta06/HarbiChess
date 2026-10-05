"""One original900-second CPU collector+independent E1 auditor; no clock resets."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from cpu_contingency_adapter_controls import HELPERS, SOURCE
from cpu_contingency_audit_support import check_source, publish, run_owned, sha


def main():
    p = argparse.ArgumentParser()
    for name in ("checkout", "output", "weights", "book", "config", "protocol", "cli-run", "helpers"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--started-epoch", type=float, required=True)
    p.add_argument("--deadline-epoch", type=float, required=True)
    a = p.parse_args()
    first, deadline = a.started_epoch, a.deadline_epoch
    assert first <= time.time() < deadline == first + 900 < 1791180000
    check_source(a.checkout, SOURCE)
    cli = json.loads((a.cli_run / "result.json").read_text())
    assert cli["schema"] == "actual-CPU-certificate-ledger-CLI-qualification-v1"
    assert cli["status"] == "pass" and cli["source_commit"] == SOURCE
    assert cli["finished_epoch"] < cli["absolute_deadline_epoch"]
    assert cli["absolute_deadline_epoch"] == cli["original_started_epoch"] + 600
    assert cli["both_all_native_epochs_freshprocess_strictload"] == [0, 1, 2]
    assert len(cli["byte_exact_artifacts"]) == 20
    for relative, digest in cli["byte_exact_artifacts"].items():
        assert sha(a.cli_run / "whole" / relative) == digest
        assert sha(a.cli_run / "split" / relative) == digest
    assert sha(a.weights) == "e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03"
    assert sha(a.book) == "1a5ca17664a828d669e58cf2bd5d9eeb7f980b83f20d3a4031d36e4ff0930ccb"
    config = json.loads(a.config.read_text())
    assert config["device"] == "cpu" and config["seed"] == 20261925
    assert config["actors"]["games"] == 64 and config["epoch_steps"] == 256
    protocol = json.loads(a.protocol.read_text())
    assert config == {**protocol["training"], "device": "cpu", "seed": 20261925}
    assert not a.output.exists()
    a.output.mkdir(parents=True)
    run = a.output / "run"
    inputs = {
        name: {"path": str(path.resolve()), "sha256": sha(path)}
        for name, path in {
            "initial_weights": a.weights,
            "book": a.book,
            "experiment_config": a.config,
            "protocol": a.protocol,
        }.items()
    }
    manifest = {
        "schema": "ufuk-cpu-contingency-E1-auditor-development-v1",
        "scope": "cpu-contingency-development-E1-not-strength",
        "run": str(run.resolve()),
        "checkout": str(a.checkout.resolve()),
        "source_commit": SOURCE,
        "original_profile_deadline_epoch": deadline,
        "started_epoch": first,
        "whole_seconds": 900,
        "frozen_config": config,
        "inputs": inputs,
        "helper_sha256": {name: sha(a.helpers / name) for name in sorted(HELPERS)},
        "neural_witness_K": 8,
        "cli_qualification_sha256": sha(a.cli_run / "result.json"),
        "owner_script_sha256": sha(__file__),
    }
    manifest_path = a.output / "E1-manifest.json"
    publish(manifest_path, manifest)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env["PYTHONPATH"] = str(a.checkout.resolve() / "src") + ":" + str(a.helpers.resolve())
    producer = [sys.executable, "-m", "harbichess.training.torch_search_acting_run"]
    for name, path in {"weights": a.weights, "book": a.book, "config": a.config, "protocol": a.protocol}.items():
        producer += ["--" + name, str(path.resolve())]
    producer += [str(run.resolve())]
    producer += ["--source-commit", SOURCE, "--max-epochs", "1", "--checkpoint-interval", "1",
                 "--deadline-epoch", str(deadline), "--memory-max-bytes", str(16 * 1024**3),
                 "--disk-min-free-bytes", str(256 * 1024**2)]
    result = {
        "schema": "actual-cpu-fullshape-original900-profile-v1",
        "status": "failed", "source_commit": SOURCE,
        "original_started_epoch": first, "absolute_deadline_epoch": deadline,
        "manifest_sha256": sha(manifest_path), "script_sha256": sha(__file__),
    }
    try:
        run_owned("collector", producer, cwd=a.checkout, env=env, output=a.output, deadline=deadline)
        audit = [sys.executable, str((a.helpers / "cpu_contingency_qualify_e1.py").resolve()),
                 "--manifest", str(manifest_path.resolve()), "--manifest-sha256", sha(manifest_path),
                 "--output", str((a.output / "independent-E1-audit.json").resolve()),
                 "--deadline-epoch", str(deadline)]
        run_owned("independent-auditor", audit, cwd=a.checkout, env=env, output=a.output, deadline=deadline)
        proof = json.loads((a.output / "independent-E1-audit.json").read_text())
        assert proof["source_commit"] == SOURCE and proof["finished_epoch"] < deadline
        assert proof["schema"] == "ufuk-cpu-contingency-E1-fullchronological-audit-result-v1"
        assert proof["status"] == "pass-actualCPU-search-acting-v4-E1-all-data-FIRST8-LAST8-original64-masks-raw-packets-and-mutations"
        assert len(proof["targeted_actual_data_mutations_rejected"]) == 6
        result.update(status="pass-actual-CPU-E1-qualified", audit_sha256=sha(a.output / "independent-E1-audit.json"))
    except BaseException as exc:
        result["error"] = repr(exc)
        raise
    finally:
        result["finished_epoch"] = time.time()
        result["whole_seconds"] = result["finished_epoch"] - first
        publish(a.output / "profile-result.json", result)


if __name__ == "__main__":
    main()
