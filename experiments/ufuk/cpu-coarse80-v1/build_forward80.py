"""Build only the residual kernel; no board evaluation or training is run."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "forward80.c"
OUT = ROOT / "libcoarse80.so"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cmd = [
        "cc",
        "-O3",
        "-fno-fast-math",
        "-ffp-contract=off",
        "-fPIC",
        "-shared",
        "-std=c11",
        str(SRC),
        "-o",
        str(OUT),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    receipt = {
        "schema": "coarse80-compiled-kernel-build-v1",
        "source_sha256": sha(SRC),
        "binary_sha256": sha(OUT),
        "command": cmd,
        "python": sys.version,
        "platform": platform.platform(),
        "fast_math": False,
        "ffp_contract": "off",
        "evaluations_performed": 0,
        "training_performed": False,
    }
    (ROOT / "build-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
