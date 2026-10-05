"""Read-only test identity repair, within the SAME original unit600 deadline."""

import argparse
import hashlib
import json
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identities(root, cfg):
    external = [Path(p).name for p in cfg["test_files"] if Path(p).is_absolute()]
    assert external == ["test_method7_certified_audit.py"]
    cases = list(root.iter("testcase"))
    rows = []
    for c in cases:
        module = c.attrib["classname"].rsplit(".", 1)[-1]
        name = module + ".py" if module else external[0]
        rows.append(name + "::" + c.attrib["name"])
        assert not any(c.find(tag) is not None for tag in ("failure", "error", "skipped"))
    assert len(rows) == len(set(rows)) == 59 and rows == cfg["cases"]
    suites = list(root.iter("testsuite"))
    assert len(suites) == 1
    assert suites[0].attrib["tests"] == "59"
    assert all(suites[0].attrib[key] == "0" for key in ("errors", "failures", "skipped"))
    return rows


def main():
    p = argparse.ArgumentParser()
    for name in ("original-root", "config", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("config-sha256", "original-result-sha256", "xml-sha256", "stdout-sha256"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    assert sha(a.config) == a.config_sha256
    cfg = json.loads(a.config.read_text())
    original_path = a.original_root / "result.json"
    assert sha(original_path) == a.original_result_sha256
    prior = json.loads(original_path.read_text())
    assert prior["status"] == "failed-preserved" and prior["error"].startswith("AssertionError(")
    assert prior["owner_config_sha256"] == a.config_sha256
    assert prior["started_epoch"] + 600 == prior["absolute_deadline_epoch"]
    assert prior["finished_epoch"] < time.time() < prior["absolute_deadline_epoch"]
    assert prior["actualCUDA"] and "A100" in prior["device_name"]
    assert prior["torch_version"] == "2.11.0+cu130"
    xml, out = a.original_root / "cases.xml", a.original_root / "stdout.log"
    assert sha(xml) == a.xml_sha256 == prior["evidence_sha256"]["cases.xml"]
    assert sha(out) == a.stdout_sha256 == prior["evidence_sha256"]["stdout.log"]
    cases = identities(ET.parse(xml).getroot(), cfg)
    assert "59 passed in 238.94s" in out.read_text()
    for path, digest in cfg["immutable_inputs"].items():
        assert sha(path) == digest, path
    repo = Path(cfg["checkout"])
    assert (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        == cfg["source_commit"]
    )
    assert not subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True
    ).strip()
    receipt = {
        **prior,
        "status": "pass-actualCUDA59-no-skip",
        "tests_passed": 59,
        "tests_skipped": 0,
        "tests_failed": 0,
        "source_clean": True,
        "cases": cases,
        "finished_epoch": time.time(),
        "original_driver_finished_epoch": prior["finished_epoch"],
        "metadata_correction": {
            "schema": "JUnit-empty-classname-one-explicit-external-testfile-v2",
            "original_failed_driver_result": str(original_path),
            "original_failed_driver_result_sha256": a.original_result_sha256,
            "helper_sha256": sha(__file__),
            "test_compute_repeated": False,
            "original600_budget_reset": False,
            "exact_raw59_XML_sha256": a.xml_sha256,
        },
    }
    receipt.pop("error")
    assert receipt["finished_epoch"] < prior["absolute_deadline_epoch"]
    with a.output.open("x") as f:
        json.dump(receipt, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"status": receipt["status"], "result_sha256": sha(a.output)}))


if __name__ == "__main__":
    main()
