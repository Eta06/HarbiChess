"""Quiet JSON means ordering weights ONLY: value remains pinned humanprior18."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, expected):
    path = Path(path)
    if sha(path) != expected:
        raise ValueError("evaluator dependency SHA differs")
    name = "quiet_arena_value_" + expected
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def quiet_model(protocol):
    directory = Path(protocol["ordering_helper_directory"])
    q = json.loads(Path(protocol["ordering_training_protocol"]["path"]).read_text())
    if (
        sha(protocol["ordering_training_protocol"]["path"])
        != protocol["ordering_training_protocol"]["sha256"]
    ):
        raise ValueError("qualified ordering protocol differs")
    for name, expected in q["ordering_helper_sha256"].items():
        if sha(directory / name) != expected:
            raise ValueError("qualified ordering helper differs")
    sys.path.insert(0, str(directory))
    import model

    if Path(model.__file__).resolve() != (directory / "model.py").resolve():
        raise ValueError("ordering model import origin differs")
    return model


class MixedValue:
    def __init__(self, path, protocol):
        path = Path(path)
        self.ordering_weights = [0.0] * 268
        if path.suffix != ".json":
            m = load(protocol["E8_value_helper"]["path"], protocol["E8_value_helper"]["sha256"])
            self.value = m.NeuralValue(path)
            self.kind = "unchanged-E8-zero-order-reference"
            return
        packet = json.loads(path.read_text())
        if packet.get("schema") == "classical-own-linear-value-v1":
            priorpath = path
            self.kind = "SAME-humanprior18-zero-order"
        elif packet.get("schema") == "classical-own-quiet-ordering-model-v1":
            self.ordering_weights = quiet_model(protocol).load_model(packet)
            priorpath = Path(protocol["fixed_prior_model"]["path"])
            self.kind = "learned-quiet268-ordering-SAME-humanprior18-value"
        else:
            raise ValueError("not explicit humanprior/quiet/E8 schema")
        if sha(priorpath) != protocol["fixed_prior_model"]["sha256"]:
            raise ValueError("fixed evaluator prior differs")
        m = load(
            protocol["classical_value_helper"]["path"], protocol["classical_value_helper"]["sha256"]
        )
        original = m.load_classical(priorpath)
        if original.theta != (0.0,) * 18:
            raise ValueError("trained value forbidden in ordering study")
        self.value = original.nonterminal

    def __call__(self, board):
        return self.value(board)
