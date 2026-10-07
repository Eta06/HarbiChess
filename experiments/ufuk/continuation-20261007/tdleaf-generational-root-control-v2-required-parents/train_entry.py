"""One original proof/fit clock across frozen contract builder and native proof."""

import argparse
import importlib
import json
import sys
from pathlib import Path

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["helper-directory", "seal", "contract", "registration"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.helper_directory.resolve()))
    factory = importlib.import_module("prepare_training")
    factory.prepare(json.loads(a.seal.read_bytes()), a.contract, a.registration)
    runner = importlib.import_module("prove")
    reg = json.loads(a.registration.read_bytes())
    reg["registration_sha256"] = runner.sha(a.registration)
    runner.execute(reg)
