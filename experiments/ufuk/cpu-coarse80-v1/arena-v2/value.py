"""Qualified ROOT fast C80 reader; original prior/E8 controls remain explicit."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, expected):
    path = Path(path).resolve()
    if sha(path) != expected:
        raise ValueError("source SHA differs")
    name = "coarse80_arena_" + expected
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_sources(protocol):
    for path, expected in protocol["source_whitelist"].items():
        if sha(path) != expected:
            raise ValueError("source whitelist changed: " + path)


class MixedValue:
    def __init__(self, path, protocol):
        check_sources(protocol)
        path = Path(path).resolve()
        roles = [
            (int(seed), row)
            for seed, models in protocol["models"].items()
            for row in [models["learned"]]
            if Path(row["path"]).resolve() == path
        ]
        if not roles:
            original = protocol["original_mixed_value"]
            self.value = load(original["path"], original["sha256"]).MixedValue(
                path, protocol
            )
            self.kind = self.value.kind
            return
        if len(roles) != 1 or sha(path) != roles[0][1]["sha256"]:
            raise ValueError("exact one registered learned candidate required")
        wrapper = json.loads(path.read_bytes())
        if (
            set(wrapper)
            != {
                "schema",
                "model",
                "contract",
                "registration_sha256",
                "native_sha256",
                "seed",
            }
            or wrapper["schema"] != "coarse80-real-ownq-candidate-v1"
            or wrapper["seed"] != roles[0][0]
        ):
            raise ValueError("qualified full coarse80 candidate envelope required")
        candidate = wrapper["model"]
        directory = Path(protocol["coarse_directory"])
        sys.path.insert(0, str(directory))
        try:
            coarse = load(
                directory / "coarse80.py",
                protocol["source_whitelist"][str(directory / "coarse80.py")],
            )
            if (
                candidate["schema"] != coarse.ARCH
                or candidate["bins"] != [4, 4]
                or candidate["zero_mean_per_piece"] is not True
                or not coarse.is_zero_mean(candidate["parameters"])
            ):
                raise ValueError("exact coarse80 constrained model schema")
            classical = protocol["classical_value_helper"]
            prior_module = load(classical["path"], classical["sha256"])
            prior = prior_module.ClassicalValue()
            fast = protocol["fast_reader"]
            module = load(fast["path"], fast["sha256"])
            import coarse80
            import compiled80

            for dependency in [coarse80, compiled80]:
                origin = Path(dependency.__file__).resolve()
                if (
                    origin != directory / origin.name
                    or str(origin) not in protocol["source_whitelist"]
                ):
                    raise ValueError(
                        "actual pure/C coarse dependency import origin differs"
                    )
            self.evaluator = module.FastEvaluator80(
                prior,
                candidate["parameters"],
                classical_features=prior_module.features,
                prior_scale=prior_module.SCALE,
                binary=protocol["compiled_binary"]["path"],
            )
        finally:
            sys.path.pop(0)
        self.value = self.evaluator.nonterminal
        self.kind = "own-Q-coarse80-final64-frozen-human18-prior"

    def __call__(self, board):
        return self.value(board)
