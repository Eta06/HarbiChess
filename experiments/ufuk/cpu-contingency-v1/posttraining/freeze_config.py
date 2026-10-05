"""Fill only actual blind overlap evidence into an immutable CPU post config."""

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--template-sha256", required=True)
    p.add_argument("--blind-result", type=Path, required=True)
    p.add_argument("--blind-result-sha256", required=True)
    p.add_argument("--required-values", type=Path, required=True)
    p.add_argument("--required-values-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    for path, digest in [
        (a.template, a.template_sha256),
        (a.blind_result, a.blind_result_sha256),
        (a.required_values, a.required_values_sha256),
    ]:
        assert sha(path) == digest
    cfg = json.loads(a.template.read_text())
    proof = json.loads(a.blind_result.read_text())
    values = json.loads(a.required_values.read_text())
    assert cfg["schema"] == "cpu-posttraining-config-v1" and values["schema"] != "ROOT_UNKNOWN"
    assert values["status"] != "ROOT_UNKNOWN" and values
    for key, value in values.items():
        assert proof[key] == value
    cfg["blind_overlap"] = {
        "path": str(a.blind_result.resolve()),
        "sha256": sha(a.blind_result),
        "required_values": values,
        "prospective_required_values_sha256": sha(a.required_values),
    }
    with a.output.open("x") as stream:
        json.dump(cfg, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"config_path": str(a.output.resolve()), "config_sha256": sha(a.output)}))


if __name__ == "__main__":
    main()
