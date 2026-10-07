"""Versioned full CPU state and honest weights-only own-phase initialization."""

import copy
import hashlib
import random
from pathlib import Path

import torch
from model import FEATURE_SCHEMA, MODEL_SCHEMA, NNUE16, tensor_batch

SCHEMA = "own-kingbucket-nnue16-full-native-cpu-v1"
MATH = dict(
    lr=0.001,
    batch=256,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    weight_decay=0,
    clip_norm=5,
    objective="MSE-tanh-authoritative-prior-logit-plus-residual",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bits_equal(a, b):
    if torch.is_tensor(a):
        return (
            torch.is_tensor(b)
            and a.dtype == b.dtype
            and a.shape == b.shape
            and torch.equal(
                a.detach().reshape(-1).view(torch.uint8), b.detach().reshape(-1).view(torch.uint8)
            )
        )
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(bits_equal(a[k], b[k]) for k in a)
    if isinstance(a, list | tuple):
        return len(a) == len(b) and all(bits_equal(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def validate_weights(state):
    expected = {"embedding.weight": (12288, 16), "head.weight": (1, 16), "head.bias": (1,)}
    if state.keys() != expected.keys() or any(
        not torch.is_tensor(state[k])
        or state[k].device.type != "cpu"
        or state[k].dtype != torch.float64
        or tuple(state[k].shape) != shape
        or not torch.isfinite(state[k]).all()
        for k, shape in expected.items()
    ):
        raise ValueError("exact finite196625 float64 model storage required")


class Learner:
    def __init__(self, contract, state=None, weights_initializer=None):
        if (
            contract["phase"] not in ["teacher-bootstrap", "own-learning", "synthetic-test"]
            or contract["updates"]
            != {"teacher-bootstrap": 256, "own-learning": 64, "synthetic-test": 8}[
                contract["phase"]
            ]
            or contract["math"] != MATH
            or contract["feature_schema"] != FEATURE_SCHEMA
            or (state is not None and weights_initializer is not None)
        ):
            raise ValueError("exact versioned phase/math; weights bridge is not full resume")
        self.contract = copy.deepcopy(contract)
        random.seed(contract["seed"])
        torch.manual_seed(contract["seed"])
        self.model = NNUE16()
        if contract["phase"] == "own-learning" and state is not None:
            weights_initializer = bootstrap_weights(contract)
        if weights_initializer is not None:
            if contract["phase"] != "own-learning" or not contract.get(
                "bootstrap_candidate_sha256"
            ):
                raise ValueError("named bootstrap weights-only initialization required")
            validate_weights(weights_initializer)
            self.model.load_state_dict(weights_initializer)
        elif contract["phase"] == "own-learning" and state is None:
            raise ValueError(
                "own-phase requires frozen-bootstrap weights, not anonymous initializer"
            )
        self.baseline = copy.deepcopy(self.model.state_dict())
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
        )
        self.sampler = random.Random(contract["seed"] ^ 0xB007)
        self.step = 0
        if state is not None:
            self.restore(state)

    def restore(self, state):
        keys = {
            "schema",
            "contract",
            "step",
            "model",
            "baseline",
            "optimizer",
            "torch_cpu_rng",
            "global_python_rng",
            "sampler_rng",
        }
        if (
            state.keys() != keys
            or state["schema"] != SCHEMA
            or state["contract"] != self.contract
            or type(state["step"]) is not int
            or not 0 <= state["step"] <= self.contract["updates"]
        ):
            raise ValueError("exact native schema/contract/counter")
        validate_weights(state["model"])
        validate_weights(state["baseline"])
        if not bits_equal(state["baseline"], self.baseline):
            raise ValueError("immutable initialized/frozen-bootstrap baseline bits differ")
        self.model.load_state_dict(state["model"])
        self.baseline = copy.deepcopy(state["baseline"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.step = state["step"]
        groups = self.optimizer.param_groups
        if (
            len(groups) != 1
            or len(groups[0]["params"]) != 3
            or groups[0]["lr"] != 0.001
            or groups[0]["betas"] != (0.9, 0.999)
            or groups[0]["eps"] != 1e-8
            or groups[0]["weight_decay"] != 0
        ):
            raise ValueError("fixed Adam math")
        if len(self.optimizer.state) != (3 if self.step else 0):
            raise ValueError("complete Adam state inventory")
        for param, value in self.optimizer.state.items():
            if (
                set(value) != {"step", "exp_avg", "exp_avg_sq"}
                or value["step"].numel() != 1
                or float(value["step"]) != self.step
            ):
                raise ValueError("Adam internal counter differs")
            for key in ["exp_avg", "exp_avg_sq"]:
                tensor = value[key]
                if (
                    tensor.dtype != torch.float64
                    or tensor.shape != param.shape
                    or tensor.device.type != "cpu"
                    or not torch.isfinite(tensor).all()
                    or (key == "exp_avg_sq" and (tensor < 0).any())
                ):
                    raise ValueError("exact Adam moment storage")
        torch.set_rng_state(state["torch_cpu_rng"])
        random.setstate(state["global_python_rng"])
        self.sampler.setstate(state["sampler_rng"])

    def advance(self, rows, stop, guard=lambda: None):
        if not self.step <= stop <= self.contract["updates"]:
            raise ValueError("no rewind/budget extension")
        while self.step < stop:
            guard()
            batch = [rows[self.sampler.randrange(len(rows))] for _ in range(256)]
            ids, offsets, prior, target = tensor_batch(batch)
            self.optimizer.zero_grad(set_to_none=True)
            prediction = torch.tanh(prior + self.model.residual(ids, offsets))
            loss = (prediction - target).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5, error_if_nonfinite=True)
            self.optimizer.step()
            self.step += 1
            guard()

    def native(self):
        return dict(
            schema=SCHEMA,
            contract=copy.deepcopy(self.contract),
            step=self.step,
            model=copy.deepcopy(self.model.state_dict()),
            baseline=copy.deepcopy(self.baseline),
            optimizer=copy.deepcopy(self.optimizer.state_dict()),
            torch_cpu_rng=torch.get_rng_state().clone(),
            global_python_rng=random.getstate(),
            sampler_rng=self.sampler.getstate(),
        )

    def candidate(self):
        if self.step != self.contract["updates"]:
            raise ValueError("fixed final only")
        return dict(
            schema=MODEL_SCHEMA,
            contract=copy.deepcopy(self.contract),
            model=copy.deepcopy(self.model.state_dict()),
        )


def load_native(path, contract):
    if Path(path).stat().st_size > 8 * 2**20:
        raise ValueError("full native8MiB cap")
    # Use only SHA-sealed local artifacts; torch pickle is not an untrusted public parser.
    state = torch.load(path, map_location="cpu", weights_only=False)
    learner = Learner(contract, state=state)
    if not bits_equal(learner.native(), state):
        raise ValueError("full native storage/RNG roundtrip differs")
    return learner


def bootstrap_weights(contract):
    path = Path(contract["bootstrap_candidate_path"])
    if path.stat().st_size > 8 * 2**20 or sha(path) != contract["bootstrap_candidate_sha256"]:
        raise ValueError("exact named bootstrap candidate bytes")
    packet = torch.load(path, map_location="cpu", weights_only=False)
    if (
        packet.keys() != {"schema", "contract", "model"}
        or packet["schema"] != MODEL_SCHEMA
        or packet["contract"]["phase"] != "teacher-bootstrap"
        or packet["contract"]["updates"] != 256
        or packet["contract"]["feature_schema"] != contract["feature_schema"]
        or packet["contract"]["prior_helper_sha256"] != contract["prior_helper_sha256"]
    ):
        raise ValueError("same architecture and frozen prior teacher-bootstrap required")
    validate_weights(packet["model"])
    return packet["model"]
