"""Explicit classical JSON or unchanged E8 NN adapter, no name/schema relabeling."""

import hashlib
import importlib.util
import sys
from pathlib import Path


def load(path, expected):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("value dependency SHA differs")
    name = "arena_value_" + expected
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class MixedValue:
    def __init__(self, path, protocol):
        if Path(path).suffix == ".json":
            module = load(
                protocol["classical_value_helper"]["path"],
                protocol["classical_value_helper"]["sha256"],
            )
            self.value = module.load_classical(path).nonterminal
            self.kind = "explicit-classical-own-JSON-v1"
        else:
            module = load(
                protocol["E8_value_helper"]["path"], protocol["E8_value_helper"]["sha256"]
            )
            self.value = module.NeuralValue(path)
            self.kind = "unchanged-E8-neural-core"

    def __call__(self, board):
        return self.value(board)
