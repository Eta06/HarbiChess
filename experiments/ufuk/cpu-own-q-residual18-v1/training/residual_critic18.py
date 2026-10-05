"""Tiny NumPy residual value head for an unexecuted own-search proposal.

The caller supplies mover-relative 18 classical features and the immutable prior logit.
This module performs no chess parsing, search, I/O, or model selection.
"""

from __future__ import annotations

import copy
import math
import random

import numpy as np

INPUTS = 18
HIDDEN = 32
SCHEMA = "own-search-residual-critic18-native-v1"


def _finite_array(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} has wrong shape or nonfinite values")
    return array


class ResidualCritic18:
    """`tanh(frozen_prior_logit + zero-initialized nonlinear residual)` for 18 features."""

    def __init__(self, *, seed: int, contract: dict, state: dict | None = None):
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("integer seed required")
        self.contract = copy.deepcopy(contract)
        if (
            not isinstance(self.contract, dict)
            or not isinstance(self.contract.get("updates"), int)
            or isinstance(self.contract.get("updates"), bool)
            or self.contract["updates"] <= 0
        ):
            raise ValueError("fixed positive update budget required in native contract")
        self.step = 0
        self.sampler_rng = random.Random(seed ^ 0x5EED)
        random.seed(seed ^ 0xA11CE)
        np.random.seed((seed ^ 0xC0FFEE) & 0xFFFFFFFF)
        self.init_rng = np.random.default_rng(seed)
        self.m = {
            "w1": np.zeros((INPUTS, HIDDEN)),
            "b1": np.zeros(HIDDEN),
            "w2": np.zeros((HIDDEN, 1)),
            "b2": np.zeros(1),
        }
        self.v = {key: value.copy() for key, value in self.m.items()}
        self.params = {
            "w1": self.init_rng.normal(
                0.0, math.sqrt(2.0 / (INPUTS + HIDDEN)), (INPUTS, HIDDEN)
            ),
            "b1": np.zeros(HIDDEN),
            "w2": np.zeros((HIDDEN, 1)),
            "b2": np.zeros(1),
        }
        # W2 and b2 are zero, so the initialized model exactly equals the prior.
        if state is not None:
            self._restore(state)

    def _restore(self, state):
        required = {
            "schema",
            "contract",
            "step",
            "params",
            "adam_m",
            "adam_v",
            "sampler_rng",
            "global_python_rng",
            "global_numpy_rng",
            "private_init_rng",
        }
        if set(state) != required or state["schema"] != SCHEMA:
            raise ValueError("native schema/keyset differs")
        if state["contract"] != self.contract:
            raise ValueError("native contract differs")
        shapes = {"w1": (INPUTS, HIDDEN), "b1": (HIDDEN,), "w2": (HIDDEN, 1), "b2": (1,)}
        restored = {}
        for field in ("params", "adam_m", "adam_v"):
            if set(state[field]) != set(shapes):
                raise ValueError("native tensor inventory differs")
            restored[field] = {
                key: _finite_array(state[field][key], shape, f"{field}.{key}")
                for key, shape in shapes.items()
            }
        step = state["step"]
        if (
            not isinstance(step, int)
            or isinstance(step, bool)
            or not 0 <= step <= self.contract["updates"]
        ):
            raise ValueError("invalid Adam step")
        self.params = restored["params"]
        self.m = restored["adam_m"]
        self.v = restored["adam_v"]
        if any((value < 0).any() for value in self.v.values()):
            raise ValueError("negative Adam second moment")
        self.sampler_rng.setstate(_tuples(state["sampler_rng"]))
        random.setstate(_tuples(state["global_python_rng"]))
        np.random.set_state(_numpy_state(state["global_numpy_rng"]))
        self.init_rng.bit_generator.state = state["private_init_rng"]
        self.step = step

    def native(self):
        def encode(values):
            return {key: value.tolist() for key, value in values.items()}

        return {
            "schema": SCHEMA,
            "contract": copy.deepcopy(self.contract),
            "step": self.step,
            "params": encode(self.params),
            "adam_m": encode(self.m),
            "adam_v": encode(self.v),
            "sampler_rng": _lists(self.sampler_rng.getstate()),
            "global_python_rng": _lists(random.getstate()),
            "global_numpy_rng": _lists(np.random.get_state()),
            "private_init_rng": copy.deepcopy(self.init_rng.bit_generator.state),
        }

    def candidate(self):
        if self.step != self.contract.get("updates"):
            raise ValueError("only the fixed final update is an inference candidate")
        return {
            "schema": "own-search-residual-critic18-candidate-v1",
            "contract": copy.deepcopy(self.contract),
            "params": {key: value.tolist() for key, value in self.params.items()},
        }

    @classmethod
    def from_candidate(cls, candidate, *, seed, contract):
        if (
            set(candidate) != {"schema", "contract", "params"}
            or candidate["schema"] != "own-search-residual-critic18-candidate-v1"
            or candidate["contract"] != contract
            or set(candidate["params"]) != {"w1", "b1", "w2", "b2"}
        ):
            raise ValueError("inference candidate contract/schema differs")
        model = cls(seed=seed, contract=contract)
        shapes = {"w1": (INPUTS, HIDDEN), "b1": (HIDDEN,), "w2": (HIDDEN, 1), "b2": (1,)}
        model.params = {
            key: _finite_array(candidate["params"][key], shape, f"candidate.{key}")
            for key, shape in shapes.items()
        }
        return model

    def predict(self, x, prior_logits):
        x = _finite_array(x, (len(x), INPUTS), "features")
        base = _finite_array(prior_logits, (len(x),), "prior logits")
        hidden = np.tanh(x @ self.params["w1"] + self.params["b1"])
        residual = (hidden @ self.params["w2"] + self.params["b2"]).reshape(-1)
        logits = np.where(residual == 0.0, base, base + residual)
        # Match the frozen scalar evaluator's Python-math tanh operation order.
        prediction = np.fromiter((math.tanh(float(value)) for value in logits), dtype=np.float64)
        return prediction, residual

    def train_batch(
        self,
        x,
        prior_logits,
        targets,
        *,
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        clip_norm=5.0,
        anchor_weight=0.1,
        weight_l2=1e-4,
    ):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != INPUTS or x.shape[0] < 1:
            raise ValueError("nonempty [batch,18] features required")
        base = _finite_array(prior_logits, (len(x),), "prior logits")
        y = _finite_array(targets, (len(x),), "targets")
        if (np.abs(y) > 1).any():
            raise ValueError("scalar search targets must be clipped to [-1,1]")
        if not np.isfinite(x).all() or learning_rate <= 0 or clip_norm <= 0:
            raise ValueError("invalid batch or optimizer settings")
        w1, b1, w2, b2 = (self.params[k] for k in ("w1", "b1", "w2", "b2"))
        hidden = np.tanh(x @ w1 + b1)
        residual = (hidden @ w2 + b2).reshape(-1)
        prediction = np.tanh(base + residual)
        n = len(x)
        dlogit = (
            2.0 * (prediction - y) * (1.0 - prediction**2)
            + 2.0 * anchor_weight * residual
        ) / n
        grads = {
            "w2": hidden.T @ dlogit[:, None] + 2.0 * weight_l2 * w2,
            "b2": np.asarray([dlogit.sum()]),
        }
        dhidden = (dlogit[:, None] @ w2.T) * (1.0 - hidden**2)
        grads["w1"] = x.T @ dhidden + 2.0 * weight_l2 * w1
        grads["b1"] = dhidden.sum(axis=0)
        norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
        scale = min(1.0, clip_norm / max(norm, 1e-30))
        loss = float(
            np.mean((prediction - y) ** 2)
            + anchor_weight * np.mean(residual**2)
            + weight_l2 * (np.sum(w1**2) + np.sum(w2**2))
        )
        self.step += 1
        for key, gradient in grads.items():
            g = gradient * scale
            self.m[key] = beta1 * self.m[key] + (1.0 - beta1) * g
            self.v[key] = beta2 * self.v[key] + (1.0 - beta2) * g * g
            mhat = self.m[key] / (1.0 - beta1**self.step)
            vhat = self.v[key] / (1.0 - beta2**self.step)
            self.params[key] -= learning_rate * mhat / (np.sqrt(vhat) + epsilon)
        state_arrays = (*self.params.values(), *self.m.values(), *self.v.values())
        if not all(np.isfinite(array).all() for array in state_arrays):
            raise ValueError("nonfinite optimizer update")
        return {"loss": loss, "grad_norm": norm, "step": self.step}


def _lists(value):
    if isinstance(value, tuple):
        return [_lists(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _tuples(value):
    return tuple(_tuples(item) for item in value) if isinstance(value, list) else value


def _numpy_state(value):
    if not isinstance(value, list) or len(value) != 5:
        raise ValueError("invalid NumPy RNG state")
    return (
        value[0],
        np.asarray(value[1], dtype=np.uint32),
        int(value[2]),
        int(value[3]),
        float(value[4]),
    )
