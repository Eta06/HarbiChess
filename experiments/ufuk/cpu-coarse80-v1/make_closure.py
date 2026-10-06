"""Emit the hash inventory required in a later root-owned registration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILES = (
    "coarse80.py",
    "learner80.py",
    "dataset80.py",
    "compiled80.py",
    "forward80.c",
    "libcoarse80.so",
    "build-receipt.json",
    "registered_cli.py",
    "resume_worker.py",
    "proof_synthetic.py",
)


def main():
    closure = {}
    for name in FILES:
        path = ROOT / name
        closure[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    output = ROOT / "source-closure.json"
    output.write_text(json.dumps(closure, indent=2, sort_keys=True) + "\n")
    print(hashlib.sha256(output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
