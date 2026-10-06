"""Full-state Adam learner for fixed own-Q labels; synthetic tests only here."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import random

from coarse80 import ARCH, PARAMS, is_zero_mean, project_zero_mean, residual_python

SCHEMA = "ownq-coarse80-full-native-v1"
UPDATES = 64
BATCH = 256
LR = 1e-3
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8
CLIP = 5.0
ANCHOR = 0.1
RIDGE = 1e-4
SMOOTH = 1e-4
OPTIMIZER = {
    "architecture": ARCH,
    "parameters": PARAMS,
    "updates": UPDATES,
    "batch_size": BATCH,
    "learning_rate": LR,
    "beta1": BETA1,
    "beta2": BETA2,
    "epsilon": EPS,
    "clip_norm": CLIP,
    "anchor_weight": ANCHOR,
    "ridge": RIDGE,
    "neighbor_smoothness": SMOOTH,
    "loss": "mover_q_mse_only",
}


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(obj):
    return hashlib.sha256(canonical(obj)).hexdigest()


def tuple_tree(value):
    return tuple(tuple_tree(x) for x in value) if isinstance(value, list) else value


def model_dict(theta):
    if not is_zero_mean(theta):
        raise ValueError("piece-type table constraint differs")
    return {"schema": ARCH, "parameters": list(theta), "bins": [4, 4], "zero_mean_per_piece": True}


def encode_native(state):
    envelope = {"state": state, "state_sha256": digest(state)}
    return gzip.compress(canonical(envelope), mtime=0)


def decode_native(blob, contract):
    envelope = json.loads(gzip.decompress(blob))
    if set(envelope) != {"state", "state_sha256"}:
        raise ValueError("native envelope keyset differs")
    state = envelope["state"]
    if digest(state) != envelope.get("state_sha256"):
        raise ValueError("native digest mismatch")
    Learner(contract["seed"], contract, state=state)
    if encode_native(state) != blob:
        raise ValueError("native is not canonical deterministic gzip")
    return state


class Learner:
    def __init__(self, seed, contract, state=None):
        self.contract = json.loads(json.dumps(contract))
        required_contract = {
            "schema",
            "seed",
            "optimizer",
            "labels_sha256",
            "prior_sha256",
            "search_producer_sha256",
            "source_commit",
            "closure_sha256",
        }
        if set(contract) != required_contract or contract["schema"] != "ownq-coarse80-contract-v1":
            raise ValueError("complete own-Q/source contract required")
        if type(contract["seed"]) is not int or contract["seed"] != seed:
            raise ValueError("native/learner seed differs")
        if contract["optimizer"] != OPTIMIZER:
            raise ValueError("fixed coarse80 optimizer contract differs")
        for key in ("labels_sha256", "prior_sha256", "search_producer_sha256", "closure_sha256"):
            value = contract[key]
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"missing immutable contract pin: {key}")
            try:
                int(value, 16)
            except ValueError as exc:
                raise ValueError(f"invalid contract digest: {key}") from exc
        source_commit = contract["source_commit"]
        if not isinstance(source_commit, str) or len(source_commit) != 40:
            raise ValueError("clean core source commit pin required")
        self.sampler = random.Random(seed)
        self.theta = [0.0] * PARAMS
        self.m = [0.0] * PARAMS
        self.v = [0.0] * PARAMS
        self.step = 0
        random.seed(seed ^ 0xC080)
        if state is not None:
            expected = {
                "schema",
                "contract",
                "theta",
                "m",
                "v",
                "step",
                "candidate",
                "sampler_rng",
                "global_rng",
            }
            if (
                set(state) != expected
                or state["schema"] != SCHEMA
                or state["contract"] != self.contract
            ):
                raise ValueError("complete compatible coarse80 native required")
            for name in ("theta", "m", "v"):
                values = state[name]
                if len(values) != PARAMS or not all(math.isfinite(float(v)) for v in values):
                    raise ValueError("invalid native parameter/Adam shape")
                setattr(self, name, [float(v) for v in values])
            if any(v < 0.0 for v in self.v):
                raise ValueError("negative Adam second moment")
            if not is_zero_mean(self.theta) or not is_zero_mean(self.m):
                raise ValueError("native piece tables or Adam first moment violate zero mean")
            if type(state["step"]) is not int or not 0 <= state["step"] <= UPDATES:
                raise ValueError("invalid native step")
            if state["candidate"] != model_dict(state["theta"]):
                raise ValueError("candidate model/native theta differ")
            self.step = state["step"]
            self.sampler.setstate(tuple_tree(state["sampler_rng"]))
            random.setstate(tuple_tree(state["global_rng"]))

    def native(self):
        return {
            "schema": SCHEMA,
            "contract": self.contract,
            "theta": self.theta,
            "m": self.m,
            "v": self.v,
            "step": self.step,
            "candidate": model_dict(self.theta),
            "sampler_rng": self.sampler.getstate(),
            "global_rng": random.getstate(),
        }

    def _gradient(self, groups):
        names = sorted(groups)
        if not names or any(not groups[name] for name in names):
            raise ValueError("nonempty game-balanced training groups required")
        grad = [0.0] * PARAMS
        for _ in range(BATCH):
            rows = groups[names[self.sampler.randrange(len(names))]]
            row = rows[self.sampler.randrange(len(rows))]
            if not isinstance(row, list | tuple) or len(row) != 3:
                raise ValueError("invalid fixed own-Q row")
            x, prior_logit, target = row
            if (
                len(x) != PARAMS
                or not all(math.isfinite(float(value)) for value in x)
                or not isinstance(prior_logit, int | float)
                or not math.isfinite(prior_logit)
                or not isinstance(target, int | float)
                or not math.isfinite(target)
                or not -1 <= target <= 1
            ):
                raise ValueError("invalid fixed own-Q row")
            z = prior_logit + residual_python(self.theta, x)
            score = math.tanh(z)
            factor = 2.0 * (score - target) * (1.0 - score * score) / BATCH
            for j, value in enumerate(x):
                grad[j] += factor * value
        # Fixed prior anchor and small ridge in the zero-centered subspace.
        for j in range(PARAMS):
            grad[j] += 2.0 * (ANCHOR + RIDGE) * self.theta[j]
        # 4-neighbor smoothness within each 4-by-4 piece table.
        for piece in range(5):
            start = piece * 16
            for rank in range(4):
                for file in range(4):
                    i = start + rank * 4 + file
                    if file < 3:
                        j = i + 1
                        d = self.theta[i] - self.theta[j]
                        grad[i] += 2.0 * SMOOTH * d
                        grad[j] -= 2.0 * SMOOTH * d
                    if rank < 3:
                        j = i + 4
                        d = self.theta[i] - self.theta[j]
                        grad[i] += 2.0 * SMOOTH * d
                        grad[j] -= 2.0 * SMOOTH * d
        # Projection makes the gradient an admissible update for zero-mean tables.
        grad = project_zero_mean(grad)
        return grad

    def advance(self, groups, target_step, guard=lambda: None):
        if type(target_step) is not int or not self.step <= target_step <= UPDATES:
            raise ValueError("backward or excessive update step")
        while self.step < target_step:
            guard()
            grad = self._gradient(groups)
            norm = math.sqrt(sum(g * g for g in grad))
            scale = min(1.0, CLIP / max(norm, 1e-30))
            self.step += 1
            for j in range(PARAMS):
                g = grad[j] * scale
                self.m[j] = BETA1 * self.m[j] + (1.0 - BETA1) * g
                self.v[j] = BETA2 * self.v[j] + (1.0 - BETA2) * g * g
                mh = self.m[j] / (1.0 - BETA1**self.step)
                vh = self.v[j] / (1.0 - BETA2**self.step)
                self.theta[j] -= LR * mh / (math.sqrt(vh) + EPS)
            # Adam's coordinate preconditioner can leave the constraint plane.
            self.theta = project_zero_mean(self.theta)
            self.m = project_zero_mean(self.m)
            if not all(math.isfinite(x) for x in self.theta + self.m + self.v):
                raise ValueError("nonfinite optimizer state")
        return self.native()
