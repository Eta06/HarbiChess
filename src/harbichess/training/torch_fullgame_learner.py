"""Experimental closed-epoch own-game PPO with terminal labels and frozen e8 anchors."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch

from harbichess.backends.torch_network import load_weights, sha256
from harbichess.chess.encoding import ENCODER_CHANNELS, BoardEncoder
from harbichess.selfplay.online_actor import OnlineActorConfig, OnlineActors
from harbichess.selfplay.online_epoch import collect_policy_epoch, serialize_policy_epoch
from harbichess.training.fullgame_own_targets import build_fullgame_targets
from harbichess.training.torch_fullgame_ppo import (
    FullGamePPOConfig,
    FullGamePPOTrainConfig,
    make_torch_epoch_inference,
    torch_model_digest,
    train_fullgame_policy_epoch,
)
from harbichess.training.torch_online_learner import _prepare_device, read_online_train_book

FULLGAME_LEARNER_SCHEMA = "torch-fresh-fullgame-ppo-v1"
EMPTY_CHAIN = hashlib.sha256(b"").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@dataclass(frozen=True, slots=True)
class TorchFullGameConfig:
    seed: int
    actors: OnlineActorConfig
    objective: FullGamePPOConfig
    schedule: FullGamePPOTrainConfig
    epoch_steps: int
    learning_rate: float
    weight_decay: float
    device: str = "cpu"

    def __post_init__(self):
        if (
            type(self.seed) is not int
            or not 0 <= self.seed < 2**63
            or self.device not in ("cpu", "cuda:0")
            or type(self.epoch_steps) is not int
            or self.epoch_steps <= 0
            or self.actors.temperature != 1.0
            or self.actors.max_additional_plies > self.epoch_steps
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise ValueError("invalid explicit fullgame configuration")


def tensor_bits_equal(left, right):
    """Exact FP storage values, including signed zero; reshape supports scalars."""
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and torch.equal(
            left.detach().contiguous().reshape(-1).view(torch.uint8),
            right.detach().contiguous().reshape(-1).view(torch.uint8),
        )
    )


class TorchFullGameLearner:
    @classmethod
    def fresh(cls, *, config, input_paths, source_commit):
        if (
            not {"initial_weights", "book"} <= input_paths.keys()
            or len(source_commit) != 40
            or any(c not in "0123456789abcdef" for c in source_commit)
        ):
            raise ValueError("fullgame requires immutable inputs and exact source commit")
        _prepare_device(config.device)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed % 2**32)
        random.seed(config.seed)
        self = cls()
        self.config, self.input_paths, self.source_commit = config, dict(input_paths), source_commit
        self.online = load_weights(input_paths["initial_weights"]).to(config.device).train()
        if (
            self.online.config.input_channels != ENCODER_CHANNELS
            or self.online.config.policy_size != 4672
        ):
            raise ValueError("fullgame requires shared104history/4672action schemas")
        for name, p in self.online.named_parameters():
            p.requires_grad_(not name.startswith("material_value_linear."))
        self.base = copy.deepcopy(self.online).eval().requires_grad_(False)
        self.behavior = copy.deepcopy(self.online).eval().requires_grad_(False)
        self.base_model_sha256 = torch_model_digest(self.base)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.online.parameters() if p.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
            foreach=False if config.device == "cuda:0" else None,
        )
        self.actors = OnlineActors(
            read_online_train_book(input_paths["book"]),
            config=config.actors,
            rng=random.Random(config.seed),
        )
        self.encoder = BoardEncoder(self.actors.rules)
        self.sampler_seed_rng = random.Random(config.seed ^ 0x51A9)
        self.epoch = self.optimizer_accepted_updates = self.optimizer_attempted_updates = 0
        self.optimizer_rejected_updates = 0
        self.sample_chain_sha256 = EMPTY_CHAIN
        self.last_epoch_gzip = b""
        self.last_sampler_rng_state = None
        self.closed = True
        self.run_config = {
            "schema": FULLGAME_LEARNER_SCHEMA,
            "config": asdict(config),
            "input_sha256": {k: sha256(v) for k, v in input_paths.items()},
            "actor_book_sha256": self.actors.book_sha256,
            "optimizer": {"class": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8},
            "targets": "only-complete-own-games-terminal-mover-WDL",
            "sampling": "frozen-behavior-temperature-one-game-balanced-PPO-surrogate",
            "transfer": "weights-only-warm-start-optimizer-reset",
            "buffer_boundary": "closed-empty-after-immutable-epoch-archive",
        }
        self.validate()
        return self

    def validate(self):
        from harbichess.training.torch_online_checkpoint import (
            _validate_models,
            _validate_optimizer,
        )

        if not self.closed:
            raise ValueError("partial fullgame epoch/pass; restore complete native boundary")
        if len(self.sample_chain_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.sample_chain_sha256
        ):
            raise ValueError("invalid fullgame chain cursor")
        if self.actors.steps != self.epoch * self.config.epoch_steps:
            raise ValueError("independent collection/epoch counters differ")
        if (
            self.optimizer_attempted_updates
            != self.optimizer_accepted_updates + self.optimizer_rejected_updates
        ):
            raise ValueError("attempted/accepted/rejected optimizer counters differ")
        if any(
            game.state != self.actors.openings[game.opening_index].state
            for game in self.actors.games
        ):
            raise ValueError("closed epoch must leave fresh complete opening histories")
        _validate_models(self.online, self.base, self.behavior)
        _validate_optimizer(self.optimizer, self.online, self.optimizer_accepted_updates)
        expected = {
            "lr": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "amsgrad": False,
            "maximize": False,
            "capturable": False,
            "differentiable": False,
            "foreach": False if self.config.device == "cuda:0" else None,
            "fused": None,
        }
        if any(self.optimizer.param_groups[0].get(k) != v for k, v in expected.items()):
            raise ValueError("fullgame optimizer settings changed")
        if any(
            p.requires_grad != (not n.startswith("material_value_linear."))
            for n, p in self.online.named_parameters()
        ):
            raise ValueError("fullgame trainable forward parameters changed")
        if self.run_config["input_sha256"] != {k: sha256(v) for k, v in self.input_paths.items()}:
            raise ValueError("immutable fullgame inputs changed")
        if torch_model_digest(self.base) != self.base_model_sha256:
            raise ValueError("immutable e8 base changed")
        for name, p in self.online.named_parameters():
            if not p.requires_grad and (
                not tensor_bits_equal(p, self.base.state_dict()[name])
                or not tensor_bits_equal(p, self.behavior.state_dict()[name])
            ):
                raise ValueError("unused inherited material changed")

    def train_epoch(self, *, guard=None):
        self.validate()
        self.closed = False  # Any exception requires native restoration, never partial save.
        if guard:
            guard()
        self.behavior.load_state_dict(self.online.state_dict(), strict=True)
        self.behavior.eval().requires_grad_(False)
        infer = make_torch_epoch_inference(
            self.behavior, self.base, self.encoder, device=self.config.device
        )

        def guarded_infer(states, legal):
            if guard:
                guard()
            return infer(states, legal)

        epoch = collect_policy_epoch(
            self.actors,
            steps=self.config.epoch_steps,
            infer=guarded_infer,
            model_digest=lambda: torch_model_digest(self.behavior),
        )
        targets = build_fullgame_targets(
            self.actors.rules, epoch, claim_draw=self.config.actors.claim_draw
        )
        sampler_seed = self.sampler_seed_rng.randrange(2**63)
        if targets.targets:
            training = train_fullgame_policy_epoch(
                self.online,
                targets=targets,
                optimizer=self.optimizer,
                rules=self.actors.rules,
                encoder=self.encoder,
                seed=sampler_seed,
                device=self.config.device,
                objective=self.config.objective,
                schedule=self.config.schedule,
                guard=guard,
            )
            self.last_sampler_rng_state = training["sampler_rng_state"]
        else:
            training = {
                "schema": "fullgame-terminal-ppo-train-v1",
                "optimizer_steps_attempted": 0,
                "optimizer_steps_committed": 0,
                "accepted_passes": 0,
                "rejected_passes": 0,
                "trained_transitions": 0,
                "sampler_rng_state": None,
                "reason": "no-complete-games-UNKNOWN-caps-and-tails-excluded",
            }
            self.last_sampler_rng_state = None
        self.epoch += 1
        self.optimizer_attempted_updates += training["optimizer_steps_attempted"]
        self.optimizer_accepted_updates += training["optimizer_steps_committed"]
        self.optimizer_rejected_updates += (
            training["optimizer_steps_attempted"] - training["optimizer_steps_committed"]
        )
        record = {
            "schema": FULLGAME_LEARNER_SCHEMA,
            "epoch": self.epoch,
            "actor_steps": self.actors.steps,
            "fresh_transitions": len(epoch.actions),
            "total_fresh_transitions": self.actors.steps * self.config.actors.games,
            "optimizer_attempted_updates": self.optimizer_attempted_updates,
            "optimizer_accepted_updates": self.optimizer_accepted_updates,
            "optimizer_rejected_updates": self.optimizer_rejected_updates,
            "collection": json.loads(serialize_policy_epoch(epoch)),
            "training": training,
            "target_counts": {
                f.name: getattr(targets, f.name) for f in fields(targets) if f.name != "targets"
            },
            "sampler_seed": sampler_seed,
            "previous_sample_chain_sha256": self.sample_chain_sha256,
        }
        self.sample_chain_sha256 = hashlib.sha256(
            bytes.fromhex(self.sample_chain_sha256) + canonical(record)
        ).hexdigest()
        record["sample_chain_sha256"] = self.sample_chain_sha256
        self.last_epoch_gzip = gzip.compress(canonical(record), mtime=0)
        self.closed = True
        self.validate()
        if guard:
            guard()
        return record

    def checkpoint(self, directory):
        from harbichess.training.torch_fullgame_checkpoint import save_fullgame_checkpoint

        return save_fullgame_checkpoint(Path(directory), self)

    @classmethod
    def resume(cls, directory, *, config, input_paths, source_commit):
        from harbichess.training.torch_fullgame_checkpoint import load_fullgame_checkpoint

        self = cls.fresh(config=config, input_paths=input_paths, source_commit=source_commit)
        load_fullgame_checkpoint(Path(directory), self)
        return self
