"""Sparse CE Adam; fixed final budget, exact full optimizer/two-RNG native."""

import math
import random

from model import DIM, model_dict, score

SCHEMA = "own-quiet-ordering-native-v1"
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
)


def tuple_tree(obj):
    return tuple(tuple_tree(x) for x in obj) if isinstance(obj, list) else obj


def row_gradient(weights, row):
    scores = [score(weights, packet) for packet in row["features"]]
    maximum = max(scores)
    exp = [math.exp(s - maximum) for s in scores]
    total = sum(exp)
    gradient = [0.0] * DIM
    for i, packet in enumerate(row["features"]):
        error = exp[i] / total - int(i == row["target"])
        for feature in packet:
            gradient[feature] += error
    return gradient


class Learner:
    def __init__(self, seed, contract, state=None):
        self.contract = contract
        self.weights, self.m, self.v = [[0.0] * DIM for _ in range(3)]
        self.step = 0
        self.sampler = random.Random(seed)
        random.seed(seed ^ 0x0A11)
        if state is not None:
            expected = {
                "schema",
                "contract",
                "weights",
                "m",
                "v",
                "step",
                "model",
                "global_rng",
                "sampler_rng",
            }
            if set(state) != expected or state["schema"] != SCHEMA or state["contract"] != contract:
                raise ValueError("exact native/schema/immutable contract required")
            if state["model"] != model_dict(state["weights"]):
                raise ValueError("model storage differs")
            for key in ("weights", "m", "v"):
                values = state[key]
                if len(values) != DIM or not all(math.isfinite(x) for x in values):
                    raise ValueError("all268 finite parameter/Adam storage required")
                setattr(self, key, list(values))
            if any(v < 0 for v in self.v):
                raise ValueError("negative Adam second moment")
            if type(state["step"]) is not int or not 0 <= state["step"] <= contract["updates"]:
                raise ValueError("native counter differs")
            self.step = state["step"]
            self.sampler.setstate(tuple_tree(state["sampler_rng"]))
            random.setstate(tuple_tree(state["global_rng"]))

    def advance(self, groups, stop, guard=lambda: None):
        if not self.step <= stop <= self.contract["updates"]:
            raise ValueError("no budget extension or rewind")
        keys = sorted(groups)
        while self.step < stop:
            guard()
            grad = [2 * MATH["l2"] * w / DIM for w in self.weights]
            for _ in range(MATH["batch"]):
                group = groups[keys[self.sampler.randrange(len(keys))]]
                row = group[self.sampler.randrange(len(group))]
                error = row_gradient(self.weights, row)
                for j in range(DIM):
                    grad[j] += error[j] / MATH["batch"]
            norm = math.sqrt(sum(x * x for x in grad))
            scale = min(1.0, MATH["clipnorm"] / max(norm, 1e-30))
            self.step += 1
            for j in range(DIM):
                g = grad[j] * scale
                self.m[j] = 0.9 * self.m[j] + 0.1 * g
                self.v[j] = 0.999 * self.v[j] + 0.001 * g * g
                first = self.m[j] / (1 - 0.9**self.step)
                second = self.v[j] / (1 - 0.999**self.step)
                self.weights[j] -= 0.01 * first / (math.sqrt(second) + 1e-8)
            guard()

    def native(self):
        return dict(
            schema=SCHEMA,
            contract=self.contract,
            weights=self.weights,
            m=self.m,
            v=self.v,
            step=self.step,
            model=model_dict(self.weights),
            global_rng=random.getstate(),
            sampler_rng=self.sampler.getstate(),
        )
