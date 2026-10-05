"""Fresh Adam over 18 frozen-prior residuals plus a 224-coefficient PST."""

import copy
import hashlib
import json
import math
import random

from pst_features import FEATURE_COUNT, TABLE_WIDTH
from pst_value import PST_L2, PST_SMOOTHNESS, model_dict

LINEAR_COUNT = 18
PARAM_COUNT = LINEAR_COUNT + FEATURE_COUNT
BATCH_SIZE = 256
LR = 0.01
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8
MAX_GRAD = 5.0


def smooth_edges():
    edges = []
    tables = FEATURE_COUNT // TABLE_WIDTH
    for table in range(tables):
        start = table * TABLE_WIDTH
        for rank in range(8):
            for file in range(4):
                i = start + rank * 4 + file
                if rank < 7:
                    edges.append((i, i + 4))
                if file < 3:
                    edges.append((i, i + 1))
    return tuple(edges)


SMOOTH_EDGES = smooth_edges()


def make_contract(
    protocol, protocol_path, config, journal_path, seed, receipt, data_sha
):
    """Bind fresh PST training to the exact frozen final classical18 rows."""
    from pathlib import Path

    def file_sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    if protocol.get("schema") != "classical-own-pst-offline-protocol-v1":
        raise ValueError("versioned PST protocol required")
    if protocol.get("status") != "registered-pst-proposal-fit-not-strength":
        raise ValueError("registered PST protocol required before a fit")
    pair = protocol["inputs"][str(seed)]
    config_path = protocol["config_paths"][str(seed)]
    fit_clock = protocol.get("phase_clocks", {}).get("fit", {})
    if (
        config.get("seed") != seed
        or config.get("max_actions") != 16384
        or config.get("source_commit") != protocol["source_commit"]
        or config.get("model_sha256") != protocol["prior_model_sha256"]
        or config.get("excluded_training_position_keys")
        != protocol["protected_position_keys"]
        or Path(config_path) != Path(config_path).resolve()
        or pair["journal_sha256"] != file_sha(journal_path)
        or pair["config_sha256"] != file_sha(config_path)
        or receipt["actions"] != protocol["final_actions"]
        or receipt["linear_dataset_sha256"]
        != pair["classical18_dataset_sha256"]
        or data_sha != pair["PST_dataset_sha256"]
        or type(fit_clock.get("first_epoch")) not in (float, int)
        or type(fit_clock.get("deadline_epoch")) not in (float, int)
        or abs(fit_clock["deadline_epoch"] - fit_clock["first_epoch"] - 1800.0)
        > 1e-6
        or fit_clock.get("deadline_epoch")
        != protocol.get("deadline_by_seed", {}).get(str(seed))
    ):
        raise ValueError("seed/final16384 journal/config binding differs")
    if protocol["optimizer"] != {
        "lr": 0.01,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "batch": BATCH_SIZE,
        "slots": 4,
        "max_updates": 1024,
    }:
        raise ValueError("frozen classical18 optimizer/update contract differs")
    if protocol["objective"] != {
        "terminal_weight": 0.75,
        "search_weight": 0.25,
        "linear_prior_l2": 0.01,
        "pst_l2": PST_L2,
        "pst_neighbor_smoothness": PST_SMOOTHNESS,
    }:
        raise ValueError("PST objective/regularization contract differs")
    updates = min(1024, 4 * receipt["training_rows"] // BATCH_SIZE)
    if updates <= 0:
        raise ValueError("empty PST optimizer schedule")
    return {
        "schema": "classical-own-pst-fit-contract-v1",
        "protocol_sha256": file_sha(protocol_path),
        "journal_sha256": pair["journal_sha256"],
        "config_sha256": pair["config_sha256"],
        "linear_dataset_sha256": receipt["linear_dataset_sha256"],
        "dataset_sha256": data_sha,
        "updates": updates,
        "receipt": receipt,
        "source_commit": protocol["source_commit"],
        "seed": seed,
        "original_deadline_epoch": protocol["deadline_by_seed"][str(seed)],
        "objective": protocol["objective"],
        "optimizer": protocol["optimizer"],
        "initialization": "18-human-prior-residuals-zero-and-224-PST-residuals-zero-fresh-Adam",
    }


class PSTLearner:
    def __init__(self, seed, contract, state=None):
        self.contract = copy.deepcopy(contract)
        self._contract_bytes = json.dumps(
            self.contract, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        self.sampler = random.Random(seed)
        self.theta = [0.0] * PARAM_COUNT
        self.m = [0.0] * PARAM_COUNT
        self.v = [0.0] * PARAM_COUNT
        self.step = 0
        random.seed(seed ^ 0x505354)
        if state is not None:
            required = {
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
                set(state) != required
                or state["schema"] != "classical-own-pst-offline-native-v1"
            ):
                raise ValueError("complete versioned PST native required")
            if state["contract"] != contract:
                raise ValueError("PST native source/data/config differs")
            for name in ("theta", "m", "v"):
                values = state[name]
                if len(values) != PARAM_COUNT or not all(
                    math.isfinite(float(x)) for x in values
                ):
                    raise ValueError("PST native parameter/moment vector invalid")
                setattr(self, name, [float(x) for x in values])
            if state["candidate"] != self.candidate():
                raise ValueError("PST native candidate/theta differs")
            if any(x < 0 for x in self.v):
                raise ValueError("negative PST second moment")
            self.step = state["step"]
            if type(self.step) is not int or not 0 <= self.step <= contract["updates"]:
                raise ValueError("PST native update counter invalid")
            self.sampler.setstate(_tuple_tree(state["sampler_rng"]))
            random.setstate(_tuple_tree(state["global_rng"]))

    def candidate(self):
        return model_dict(self.theta[:LINEAR_COUNT], self.theta[LINEAR_COUNT:])

    def native(self):
        return {
            "schema": "classical-own-pst-offline-native-v1",
            "contract": self.contract,
            "theta": self.theta,
            "m": self.m,
            "v": self.v,
            "step": self.step,
            "candidate": self.candidate(),
            "sampler_rng": self.sampler.getstate(),
            "global_rng": random.getstate(),
        }

    def advance(self, groups, target, guard=lambda: None):
        if (
            json.dumps(
                self.contract, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            != self._contract_bytes
        ):
            raise ValueError("immutable PST fit contract changed")
        if (
            type(target) is not int
            or not self.step <= target <= self.contract["updates"]
        ):
            raise ValueError("backward/excess PST update")
        groups = [groups[key] for key in sorted(groups)]
        if not groups or any(not group for group in groups):
            raise ValueError("empty PST training groups")
        while self.step < target:
            guard()
            grad = [0.0] * PARAM_COUNT
            for _ in range(BATCH_SIZE):
                rows = groups[self.sampler.randrange(len(groups))]
                phi18, _prior, z, q, psq, prior_logit = rows[
                    self.sampler.randrange(len(rows))
                ]
                if len(phi18) != LINEAR_COUNT or len(psq) != FEATURE_COUNT:
                    raise ValueError("PST dataset shape differs")
                residual = sum(
                    t * x for t, x in zip(self.theta[:LINEAR_COUNT], phi18, strict=True)
                ) + sum(
                    t * x for t, x in zip(self.theta[LINEAR_COUNT:], psq, strict=True)
                )
                score = math.tanh(prior_logit + residual)
                dloss = (
                    2.0
                    * (0.75 * (score - z) + 0.25 * (score - q))
                    * (1.0 - score * score)
                    / BATCH_SIZE
                )
                for j in range(LINEAR_COUNT):
                    grad[j] += dloss * phi18[j]
                for j in range(FEATURE_COUNT):
                    grad[LINEAR_COUNT + j] += dloss * psq[j]
            # Keep the classical18 regularizer unchanged; predeclare two spatial terms.
            for j in range(LINEAR_COUNT):
                grad[j] += 0.02 * self.theta[j] / LINEAR_COUNT
            for j in range(FEATURE_COUNT):
                grad[LINEAR_COUNT + j] += (
                    2.0 * PST_L2 * self.theta[LINEAR_COUNT + j] / FEATURE_COUNT
                )
            smooth_scale = 2.0 * PST_SMOOTHNESS / len(SMOOTH_EDGES)
            for left, right in SMOOTH_EDGES:
                delta = (
                    self.theta[LINEAR_COUNT + left] - self.theta[LINEAR_COUNT + right]
                )
                grad[LINEAR_COUNT + left] += smooth_scale * delta
                grad[LINEAR_COUNT + right] -= smooth_scale * delta
            norm = math.sqrt(sum(g * g for g in grad))
            scale = min(1.0, MAX_GRAD / max(norm, 1e-30))
            self.step += 1
            for j, raw in enumerate(grad):
                g = raw * scale
                self.m[j] = BETA1 * self.m[j] + (1.0 - BETA1) * g
                self.v[j] = BETA2 * self.v[j] + (1.0 - BETA2) * g * g
                mhat = self.m[j] / (1.0 - BETA1**self.step)
                vhat = self.v[j] / (1.0 - BETA2**self.step)
                self.theta[j] -= LR * mhat / (math.sqrt(vhat) + EPS)
            if not all(math.isfinite(x) for x in self.theta):
                raise ValueError("nonfinite PST update")
        return self.native()


def _tuple_tree(value):
    return tuple(_tuple_tree(x) for x in value) if isinstance(value, list) else value
