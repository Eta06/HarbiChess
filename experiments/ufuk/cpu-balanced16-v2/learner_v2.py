"""Balanced risk BCE; freshv2 native."""

import math
import random

from transform import DIM, load_parent, model_dict, probability

parent = load_parent(
    "learner.py", "3d77eea89bf3817f8de096b7de9bcded5bb1f7d017a60aac89c38aba826022b4"
)

SCHEMA = "own-selective-quiescence-balanced-risk-native-v2"
MATH = dict(
    lr=0.01,
    batch=256,
    slots=4,
    max_updates=1024,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    clipnorm=5.0,
    l2=0.001,
    objective="nonchecked-only-game-equal-balanced-BCE-risk-v2",
)


def tuple_tree(obj):
    return tuple(tuple_tree(x) for x in obj) if isinstance(obj, list) else obj


def row_gradient(weights, row, prevalence):
    x = row["features"]
    if len(x) != DIM or row["target"] not in [0, 1]:
        raise ValueError("binary16 target")
    if x[1] != 0 or not 0 < prevalence < 1:
        raise ValueError("nonchecked TRAIN required")
    weight = (1 - prevalence) / prevalence if row["target"] == 1 else 1.0
    error = weight * (probability(weights, x) - row["target"]) / (2 * (1 - prevalence))
    return [error * v for v in x]


class Learner(parent.Learner):
    def __init__(self, seed, contract, state=None):
        required = {
            "schema",
            "seed",
            "updates",
            "math",
            "transform",
            "inputs",
            "source_commit",
            "helper_sha256",
            "first_epoch",
            "deadline_epoch",
        }
        if (
            set(contract) != required
            or contract["schema"] != "own-balanced-risk-learning-contract-v2"
            or contract["seed"] != seed
            or contract["math"] != MATH
            or contract["updates"] != 48
        ):
            raise ValueError("v2 immutable contract")
        self.prevalence = contract["transform"]["game_equal_positive_prevalence"]
        if not 0 < self.prevalence < 1:
            raise ValueError("nondegenerate TRAIN")
        super().__init__(seed, contract)  # fresh ZERO Adam only; no v1 native imported
        if state is not None:
            if (
                set(state) != set(self.native())
                or state["schema"] != SCHEMA
                or state["contract"] != contract
                or state["model"] != model_dict(state["weights"])
            ):
                raise ValueError("v2 full native")
            for k in ["weights", "m", "v"]:
                x = state[k]
                if len(x) != DIM or any(not math.isfinite(v) for v in x):
                    raise ValueError("finite16 parameter/Adam")
                setattr(self, k, list(x))
            if (
                any(v < 0 for v in self.v)
                or type(state["step"]) is not int
                or not 0 <= state["step"] <= 48
            ):
                raise ValueError("native counter/Adam")
            self.step = state["step"]
            self.sampler.setstate(tuple_tree(state["sampler_rng"]))
            random.setstate(tuple_tree(state["global_rng"]))

    def advance(self, groups, stop, guard=lambda: None):
        from transform import group_digest, summary

        measured = summary(groups)
        if (
            measured["game_equal_positive_prevalence"] != self.prevalence
            or group_digest(groups) != self.contract["transform"]["training_group_sha256"]
        ):
            raise ValueError("fixed transformed TRAIN groups")
        old_gradient = parent.row_gradient
        try:
            parent.row_gradient = lambda weights, row: row_gradient(weights, row, self.prevalence)
            return super().advance(groups, stop, guard)
        finally:
            parent.row_gradient = old_gradient

    def native(self):
        packet = super().native()
        packet.update(schema=SCHEMA, model=model_dict(self.weights))
        return packet
