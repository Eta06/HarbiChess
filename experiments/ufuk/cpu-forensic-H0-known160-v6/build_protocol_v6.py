"""NEW inference-helper binding; original native/data/phase contracts remain unchanged."""

import argparse
import json
from pathlib import Path

from build_protocol_v5 import build, ref

HERE = Path(__file__).resolve().parent


def build_v6(base_protocol):
    q = build(base_protocol)
    q["runtime_revision"] = "forensic-h0-known160-runtime-v6"
    q["status"] = "DRAFT-new-runtime-v6-ROOT-new600-clock-pending"
    q["inference_only_conversion"] = (
        "v5 phase classifier failure preserved; v6 uses actual forensic-v4 reader and "
        "literalzero H0 role; model/native/source/data/training bytes untouched"
    )
    q["arena_helper_sha256"] = {p.name: ref(p)["sha256"] for p in HERE.glob("*.py")}
    q["inference_runtime_helpers"] = {name: ref(HERE / name) for name in ["runtime.py", "value.py"]}
    return q


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(build_v6(args.base_protocol), stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
