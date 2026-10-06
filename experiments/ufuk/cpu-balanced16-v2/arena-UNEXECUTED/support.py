"""Local pinned imports only, no jobs or registration side effects."""

import hashlib
import importlib.util
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, expected, name):
    path = Path(path).resolve()
    if sha(path) != expected:
        raise ValueError("dependency SHA differs")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path:
        raise ValueError("import origin differs")
    return module
