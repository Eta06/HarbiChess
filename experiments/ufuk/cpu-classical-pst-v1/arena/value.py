"""Explicit 18-prior/PST242/E8 schema dispatch; no double-add or policy relabel."""

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, expected):
    path = Path(path)
    if sha(path) != expected:
        raise ValueError("value dependency SHA differs")
    name = "pst_arena_value_" + expected
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def pst_module(protocol):
    directory = Path(protocol["PST_helper_directory"])
    original = json.loads(Path(protocol["PST_training_protocol"]["path"]).read_text())
    if (
        sha(protocol["PST_training_protocol"]["path"])
        != protocol["PST_training_protocol"]["sha256"]
    ):
        raise ValueError("qualified PST training protocol differs")
    for name, expected in original["helper_sha256"].items():
        target = directory / name
        if name.startswith("reference_"):
            role = name.removeprefix("reference_").removesuffix(".py")
            filename = "journal_v3.py" if role == "journal" else role + ".py"
            target = directory / "frozen_reference" / filename
        if sha(target) != expected:
            raise ValueError("qualified11 PST helper closure differs")
    sys.path.insert(0, str(directory))
    import pst_value

    if Path(pst_value.__file__).resolve() != (directory / "pst_value.py").resolve():
        raise ValueError("PST import origin differs")
    return pst_value


def optimized_type(module):
    class ArenaPSTValue(module.PSTValue):
        def nonterminal(self, board):
            # Exact zero identity; omit ONLY unused base computation for nonzero PST.
            if not any(self.pst_theta):
                return self.linear.nonterminal(board)
            phi = module.piece_square_features(board)
            residual = sum(w * x for w, x in zip(self.pst_theta, phi, strict=True))
            raw = module.features(board)
            prior = sum(w * x for w, x in zip(module.PRIOR, raw, strict=True)) / module.SCALE
            linear = sum(
                t * x / scale
                for t, x, scale in zip(self.linear.theta, raw, module.FEATURE_SCALES, strict=True)
            )
            return math.tanh(prior + linear + residual)

    return ArenaPSTValue


class MixedValue:
    def __init__(self, path, protocol):
        path = Path(path)
        if path.suffix != ".json":
            module = load(
                protocol["E8_value_helper"]["path"], protocol["E8_value_helper"]["sha256"]
            )
            self.value = module.NeuralValue(path)
            self.kind = "unchanged-E8-neural-core"
            return
        packet = json.loads(path.read_text())
        if packet.get("schema") == "classical-own-linear-value-v1":
            module = load(
                protocol["classical_value_helper"]["path"],
                protocol["classical_value_helper"]["sha256"],
            )
            model = module.load_classical(path)
            if packet["theta"] != [0.0] * 18:
                raise ValueError(
                    "linear JSON is SAMEhumanprior baseline, not another trained control"
                )
            self.value = model.nonterminal
            self.kind = "explicit-untrained-humanprior18-JSON-v1"
        elif packet.get("schema") == "classical-own-pst-residual-v1":
            module = pst_module(protocol)
            original = module.load_pst(path)  # enforce entire actual learned schema
            optimized = optimized_type(module)(original.linear.theta, original.pst_theta)
            self.value = optimized.nonterminal
            self.kind = "explicit-trained-PST242-JSON-v1-arena-equivalent"
        else:
            raise ValueError("unknown evaluator JSON schema")

    def __call__(self, board):
        return self.value(board)
