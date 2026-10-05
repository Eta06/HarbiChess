"""Compiled qualified C18, with untouched fast classical and E8 controls."""

import gzip
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, expected):
    path = Path(path).resolve()
    if sha(path) != expected:
        raise ValueError("dependency SHA differs")
    name = "c18_arena_" + expected
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path:
        raise ValueError("dependency origin differs")
    return module


def read_gzip(path):
    path = Path(path)
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("compressed native size")
    with gzip.open(path, "rb") as stream:
        raw = stream.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("native decoded size")
    return json.loads(raw)


def model_from_candidate(candidate, seed, protocol):
    dep = protocol["critic18"]
    module = load(dep["path"], dep["sha256"])
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        return module.ResidualCritic18.from_candidate(
            candidate, seed=seed, contract=candidate["contract"]
        )
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


class MixedValue:
    def __init__(self, path, protocol):
        path = Path(path).resolve()
        roles = [(int(seed), row) for seed, models in protocol["models"].items()
                 for row in [models["learned"]] if Path(row["path"]).resolve() == path]
        if not roles:
            original = protocol["original_mixed_value"]
            module = load(original["path"], original["sha256"])
            self.value = module.MixedValue(path, protocol)
            self.kind = self.value.kind
            return
        if len(roles) != 1:
            raise ValueError("candidate must identify exactly one registered seed")
        seed, row = roles[0]
        if sha(path) != row["sha256"]:
            raise ValueError("candidate SHA differs")
        candidate = read_gzip(path)
        for source, expected in candidate["contract"]["inference_closure"].items():
            if sha(source) != expected:
                raise ValueError("compiled inference closure differs")
        dep = protocol["compiled18"]
        directory = Path(dep["path"]).resolve().parent
        sys.path.insert(0, str(directory))
        try:
            module = load(dep["path"], dep["sha256"])
            import _ownq_forward18_v2

            binary = Path(protocol["compiled_binary"]["path"]).resolve()
            if (Path(_ownq_forward18_v2.__file__).resolve() != binary
                    or sha(binary) != protocol["compiled_binary"]["sha256"]):
                raise ValueError("compiled binary origin/SHA differs")
        finally:
            sys.path.pop(0)
        classical = protocol["classical_value_helper"]
        features = load(classical["path"], classical["sha256"])
        model = model_from_candidate(candidate, seed, protocol)
        self.evaluator = module.CompiledEvaluator18(
            model, classical_features=features.features, prior_weights=features.PRIOR,
            feature_scales=features.FEATURE_SCALES, prior_score_scale=features.SCALE,
        )
        self.value = self.evaluator.nonterminal
        self.kind = "own-search-residual18-compiled-nonzero-final64"

    def __call__(self, board):
        return self.value(board)
