"""Prospective formal6 search-acting-v2 epoch1->2 exact audit-only replay; no v1 native."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from harbichess.backends.torch_network import sha256
from own6_adapter_controls import validate_spec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--audit-run", type=Path, required=True)
    p.add_argument("--deadline-epoch", type=float, required=True)
    p.add_argument("--manifest-sha256", required=True)
    a = p.parse_args()
    started = time.time()
    assert 0 < a.deadline_epoch - started <= 600
    original = a.run.resolve()
    audit = a.audit_run.resolve()
    assert audit.parent == original.parent and audit != original and not audit.exists()
    assert sha256(a.manifest) == a.manifest_sha256
    spec = json.loads(a.manifest.read_text())
    validate_spec(spec)
    assert original == Path(spec["run"]).resolve()
    for name, digest in spec["helper_sha256"].items():
        assert sha256(Path(__file__).with_name(name)) == digest
    meta = json.loads((original / "metadata.json").read_text())
    assert (
        meta["source_commit"] == spec["source_commit"]
        and meta["absolute_deadline_epoch"] == spec["original_training_deadline_epoch"]
        and meta["schema"] == "search-acting-supervised-run-v2"
        and meta["max_epochs"] == spec["fixed_epochs"]
        and meta["config"]["seed"] in (20261625, 20261626)
        and meta["checkpoint_interval"] == 1
        and meta["config"] == spec["frozen_config"]
    )
    paths = {name: Path(row["path"]).resolve() for name, row in spec["inputs"].items()}
    for name, path in paths.items():
        assert sha256(path) == spec["inputs"][name]["sha256"]
        assert (
            (original / meta["inputs"][name]["relative_path"]).resolve()
            == path
            == (audit / meta["inputs"][name]["relative_path"]).resolve()
        )
    parenthashes = {
        str(path.relative_to(original)): sha256(path)
        for folder in ("checkpoints/epoch-00000001", "checkpoints/epoch-00000002")
        for path in (original / folder).iterdir()
        if path.is_file()
    }
    parenthashes["journal/epoch-00000001.json.gz"] = sha256(
        original / "journal/epoch-00000001.json.gz"
    )
    parenthashes["journal/epoch-00000002.json.gz"] = sha256(
        original / "journal/epoch-00000002.json.gz"
    )
    audit.mkdir()
    (audit / "checkpoints").mkdir()
    (audit / "journal").mkdir()
    shutil.copytree(
        original / "checkpoints/epoch-00000001", audit / "checkpoints/epoch-00000001"
    )
    shutil.copyfile(
        original / "journal/epoch-00000001.json.gz",
        audit / "journal/epoch-00000001.json.gz",
    )
    auditmeta = dict(meta)
    auditmeta["absolute_deadline_epoch"] = a.deadline_epoch
    (audit / "metadata.json").write_text(json.dumps(auditmeta, indent=2) + "\n")
    receipt = {
        "schema": "ufuk-search-acting-independent-freshCLI-replay-v2",
        "seed": meta["config"]["seed"],
        "scope": (
            "AUDIT ONLY: one duplicate epoch32768presentations; production inp"
            "ut/config/source unchanged; separate audit deadline, no productio"
            "n budget reset"
        ),
        "source_commit": spec["source_commit"],
        "manifest_sha256": sha256(a.manifest),
        "script_sha256": sha256(Path(__file__)),
        "original_run": str(original),
        "original_deadline_epoch": meta["absolute_deadline_epoch"],
        "audit_deadline_epoch": a.deadline_epoch,
        "parent_artifact_sha256": parenthashes,
        "started_epoch": started,
        "status": "running",
    }
    (audit / "audit-parent.json").write_text(json.dumps(receipt, indent=2) + "\n")
    args = [
        sys.executable,
        "-m",
        "harbichess.training.torch_search_acting_run",
        str(audit),
        "--weights",
        str(paths["initial_weights"]),
        "--book",
        str(paths["book"]),
        "--config",
        str(paths["experiment_config"]),
        "--protocol",
        str(paths["protocol"]),
        "--source-commit",
        spec["source_commit"],
        "--max-epochs",
        str(spec["fixed_epochs"]),
        "--checkpoint-interval",
        "1",
        "--deadline-epoch",
        str(a.deadline_epoch),
        "--memory-max-bytes",
        str(meta["memory_max_bytes"]),
        "--disk-min-free-bytes",
        str(meta["disk_min_free_bytes"]),
        "--stop-at",
        "2",
        "--resume",
        str(audit / "checkpoints/epoch-00000001"),
    ]
    receipt["command"] = args
    try:
        with (
            (audit / "fresh-cli.stdout.log").open("x") as out,
            (audit / "fresh-cli.stderr.log").open("x") as err,
        ):
            subprocess.run(
                args,
                stdout=out,
                stderr=err,
                check=True,
                timeout=max(0.01, a.deadline_epoch - time.time()),
            )
        assert time.time() < a.deadline_epoch
        payloads = (
            "model.safetensors",
            "base.safetensors",
            "behavior.safetensors",
            "training.pt",
            "actor.json",
            "last-frozen-epoch.json.gz",
        )
        compare = ["journal/epoch-00000002.json.gz"] + [
            f"checkpoints/epoch-00000002/{name}" for name in payloads
        ]
        for relative in compare:
            assert (original / relative).read_bytes() == (
                audit / relative
            ).read_bytes(), relative
        assert sha256(a.manifest) == a.manifest_sha256
        for name, digest in spec["helper_sha256"].items():
            assert sha256(Path(__file__).with_name(name)) == digest
        for name, path in paths.items():
            assert sha256(path) == spec["inputs"][name]["sha256"]
        for relative, digest in parenthashes.items():
            assert sha256(original / relative) == digest
        record = __import__("gzip").decompress(
            (audit / "journal/epoch-00000002.json.gz").read_bytes()
        )
        record = json.loads(record)
        receipt.update(
            status="pass-exact-search-acting-v2-native2-all-six-payloads-and-journal",
            compared_sha256={
                relative: sha256(audit / relative) for relative in compare
            },
            duplicate_fresh_presentations=32768,
            duplicate_optimizer_attempts=record["training"][
                "optimizer_steps_attempted"
            ],
            duplicate_optimizer_committed=record["training"][
                "optimizer_steps_committed"
            ],
        )
    except Exception as e:
        receipt.update(status="failed-preserved-audit", error=repr(e))
        raise
    finally:
        receipt["finished_epoch"] = time.time()
        (audit / "audit-result.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
