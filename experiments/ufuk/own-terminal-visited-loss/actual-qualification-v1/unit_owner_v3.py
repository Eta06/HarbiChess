"""Prospective source8 complete CUDA unit suite; exact frozen cases and no skip."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identities(root, expected):
    assert len(expected) == len(set(expected)) and len(expected) >= 59
    cases = list(root.iter("testcase"))
    actual = []
    for case in cases:
        module = case.attrib["classname"].rsplit(".", 1)[-1]
        suffix = "::" + case.attrib["name"]
        if module:
            identity = module + ".py" + suffix
        else:
            candidates = [identity for identity in expected if identity.endswith(suffix)]
            assert len(candidates) == 1, (
                "External JUnit identity must be unique in frozen inventory"
            )
            identity = candidates[0]
        assert not any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))
        actual.append(identity)
    assert actual == expected and len(actual) == len(set(actual))
    suites = list(root.iter("testsuite"))
    assert len(suites) == 1 and int(suites[0].attrib["tests"]) == len(expected)
    assert all(suites[0].attrib[name] == "0" for name in ("failures", "errors", "skipped"))
    return actual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    args = parser.parse_args()
    assert sha(args.config) == args.config_sha256
    cfg = json.loads(args.config.read_text())
    root = Path(cfg["output"])
    root.mkdir(exist_ok=False)
    repo = Path(cfg["checkout"])
    for path, digest in cfg["immutable_inputs"].items():
        assert sha(path) == digest, path
    assert (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        == cfg["source_commit"]
    )
    assert not subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True
    ).strip()
    start = time.time()
    deadline = min(start + cfg["whole_seconds"], 1791180000)
    assert deadline == start + 600
    receipt = {
        "schema": "ufuk-clean-producer-actualCUDA-unit-suite-v1",
        "status": "failed-preserved",
        "source_commit": cfg["source_commit"],
        "started_epoch": start,
        "absolute_deadline_epoch": deadline,
        "owner_config_sha256": args.config_sha256,
        "cases": cfg["cases"],
        "immutable_inputs": cfg["immutable_inputs"],
    }
    (root / "original-owner.json").write_text(json.dumps(receipt, indent=2) + "\n")
    child = None
    try:
        import torch

        receipt.update(
            torch_version=torch.__version__,
            device_name=torch.cuda.get_device_name(),
            actualCUDA=torch.cuda.is_available(),
        )
        assert (
            receipt["actualCUDA"]
            and "A100" in receipt["device_name"]
            and torch.__version__ == "2.11.0+cu130"
        )
        env = {**os.environ, **cfg["env"]}
        command = [
            "/usr/bin/python3",
            "-m",
            "pytest",
            "-q",
            "-ra",
            "--junitxml",
            str(root / "cases.xml"),
            *cfg["test_files"],
        ]
        receipt["command"] = command
        with (root / "stdout.log").open("x") as out, (root / "stderr.log").open("x") as err:
            child = subprocess.Popen(
                command,
                cwd=repo,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
            st = Path(f"/proc/{child.pid}/stat").read_text().rsplit(")", 1)[1].split()
            (root / "child-owner.json").write_text(
                json.dumps(
                    {
                        "pid": child.pid,
                        "startticks": int(st[19]),
                        "pgid": os.getpgid(child.pid),
                        "argv": command,
                        "started_epoch": start,
                        "deadline_epoch": deadline,
                    },
                    indent=2,
                )
                + "\n"
            )
            while child.poll() is None:
                if time.time() >= deadline:
                    raise TimeoutError("Original600 complete CUDA unit suite exhausted")
                time.sleep(0.2)
        xml = ET.parse(root / "cases.xml").getroot()
        identities(xml, cfg["cases"])
        cases = list(xml.iter("testcase"))
        expected_count = len(cfg["cases"])
        receipt["tests_skipped"] = sum(c.find("skipped") is not None for c in cases)
        receipt["tests_failed"] = sum(
            c.find("failure") is not None or c.find("error") is not None for c in cases
        )
        assert child.returncode == 0 and receipt["tests_skipped"] == receipt["tests_failed"] == 0
        assert time.time() < deadline
        for path, digest in cfg["immutable_inputs"].items():
            assert sha(path) == digest, path
        assert not subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True
        ).strip()
        receipt.update(
            status=f"pass-actualCUDA{expected_count}-no-skip",
            tests_passed=expected_count,
            source_clean=True,
        )
    except BaseException as exc:
        receipt["error"] = repr(exc)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
        receipt["finished_epoch"] = time.time()
        receipt["evidence_sha256"] = {p.name: sha(p) for p in root.iterdir() if p.is_file()}
        with (root / "result.json").open("x") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(receipt))
    return 0 if receipt["status"] == f"pass-actualCUDA{len(cfg['cases'])}-no-skip" else 1


if __name__ == "__main__":
    raise SystemExit(main())
