"""One original10800 clock across frozen factory and collector; ROOT only."""

import argparse
import importlib
import json
import sys
from pathlib import Path

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["helper-directory", "seal", "registration"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.helper_directory.resolve()))
    factory = importlib.import_module("metadata_factory")
    factory.collection(json.loads(a.seal.read_bytes()), a.registration)
    runner = importlib.import_module("run_collection")
    runner.execute(a.registration)
