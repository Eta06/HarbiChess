"""Search-acting v2: sparse pre-action improved behavior, supervised CE and own terminal WDL."""

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
from harbichess.chess.encoding import ENCODER_CHANNELS
from harbichess.selfplay.online_actor import OnlineActorConfig, OnlineActors
from harbichess.selfplay.online_epoch import (
    serialize_policy_epoch,
)
from harbichess.training.fullgame_own_targets import build_fullgame_targets
from harbichess.training.ownsearch_targets import (
    OwnSearchConfig,
)
from harbichess.training.search_acting_epoch import collect_search_acting_epoch
from harbichess.training.torch_array_encoder import TorchArrayBoardEncoder
from harbichess.training.torch_fullgame_ppo import (
    FullGamePPOTrainConfig,
    make_torch_epoch_inference,
    torch_model_digest,
)
from harbichess.training.torch_online_learner import (
    _prepare_device,
    read_online_train_book,
)
from harbichess.training.torch_ownsearch_core import (
    OwnSearchObjective,
    SearchTrainRow,
    train_search_epoch,
)

SEARCH_ACTING_LEARNER_SCHEMA = "torch-fresh-sparse-search-acting-v2"

OWNSEARCH_BOARD_CACHE_SIZE = (
    8192  # Runtime-only; pinned source/protocol, not native state.
)
EMPTY_CHAIN = hashlib.sha256(b"").hexdigest()


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


@dataclass(frozen=True, slots=True)
class TorchSearchActingConfig:
    seed: int
    actors: OnlineActorConfig
    objective: OwnSearchObjective
    search: OwnSearchConfig
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


class TorchSearchActingLearner:
    @classmethod
    def fresh(cls, *, config, input_paths, source_commit):
        if (
            not {"initial_weights", "book"} <= input_paths.keys()
            or len(source_commit) != 40
            or any(c not in "0123456789abcdef" for c in source_commit)
        ):
            raise ValueError(
                "fullgame requires immutable inputs and exact source commit"
            )
        if (
            sha256(input_paths["initial_weights"])
            != "e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03"
        ):
            raise ValueError(
                "search-acting-v2 fresh requires original immutable e8; new Adam, not full resume"
            )
        _prepare_device(config.device)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed % 2**32)
        random.seed(config.seed)
        self = cls()
        self.config, self.input_paths, self.source_commit = (
            config,
            dict(input_paths),
            source_commit,
        )
        self.online = (
            load_weights(input_paths["initial_weights"]).to(config.device).train()
        )
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
        self.actors.rules.board_cache_size = OWNSEARCH_BOARD_CACHE_SIZE
        self.encoder = TorchArrayBoardEncoder(self.actors.rules)
        self.sampler_seed_rng = random.Random(config.seed ^ 0x51A9)
        self.schedule_rng = random.Random(config.seed ^ 0x831A)
        self.search_rngs = [
            random.Random((config.seed ^ 0xFA671) + slot)
            for slot in range(config.actors.games)
        ]
        self.pending_search_schedule = {}
        self.epoch = self.optimizer_accepted_updates = (
            self.optimizer_attempted_updates
        ) = 0
        self.optimizer_rejected_updates = 0
        self.sample_chain_sha256 = EMPTY_CHAIN
        self.last_epoch_gzip = b""
        self.last_sampler_rng_state = None
        self.closed = True
        self.run_config = {
            "schema": SEARCH_ACTING_LEARNER_SCHEMA,
            "config": asdict(config),
            "input_sha256": {k: sha256(v) for k, v in input_paths.items()},
            "actor_book_sha256": self.actors.book_sha256,
            "optimizer": {"class": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8},
            "targets": "sparse-own-Gumbel-policy-CE-plus-complete-own-game-mover-WDL",
            "sampling": (
                "raw-pi-archived;preselected-search-T1-mu;supervised-CE-no-PPO;"
                "game-balanced-known-WDL"
            ),
            "search_schedule": "independent-before-outcome-uniform-offset-per-game-block-v1",
            "policy_sampling": "uniform-exogenously-searched-roots;UNKNOWN-policy-allowed",
            "transfer": "original-e8-weights-only-new-Adam-not-v1-resume",
            "buffer_boundary": "closed-empty-after-immutable-epoch-archive",
        }
        self.validate()
        return self

    def validate(self):
        from harbichess.training.torch_online_checkpoint import (
            _validate_models,
            _validate_optimizer,
        )

        if (
            self.pending_search_schedule
            or len(self.search_rngs) != self.config.actors.games
        ):
            raise ValueError(
                "ownsearch boundary must have empty pending schedule and all actor search RNGs"
            )
        if not self.closed:
            raise ValueError(
                "partial fullgame epoch/pass; restore complete native boundary"
            )
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
        _validate_optimizer(
            self.optimizer, self.online, self.optimizer_accepted_updates
        )
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
        if self.run_config["input_sha256"] != {
            k: sha256(v) for k, v in self.input_paths.items()
        }:
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
        guard = guard or (lambda: None)
        self.validate()
        self.closed = False
        guard()
        self.behavior.load_state_dict(self.online.state_dict(), strict=True)
        self.behavior.eval().requires_grad_(False)
        infer = make_torch_epoch_inference(
            self.behavior, self.base, self.encoder, device=self.config.device
        )
        epoch, ledger = collect_search_acting_epoch(
            self.actors,
            steps=self.config.epoch_steps,
            infer=infer,
            model_digest=lambda: torch_model_digest(self.behavior),
            behavior=self.behavior,
            config=self.config.search,
            schedule_rng=self.schedule_rng,
            search_rngs=self.search_rngs,
            device=self.config.device,
            guard=guard,
        )
        targets = build_fullgame_targets(
            self.actors.rules, epoch, claim_draw=self.config.actors.claim_draw
        )
        sampler_seed = self.sampler_seed_rng.randrange(2**63)
        training = train_search_epoch(
            self.online,
            policy_rows=tuple(
                SearchTrainRow(
                    row.transition,
                    row.legal_actions,
                    row.policy,
                    row.base_policy,
                    row.base_wdl,
                    tuple(receipt["search_policy"]),
                    row.transition.source_id,
                    row.transition.game_index,
                )
                for receipt in ledger["roots"]
                for row in [epoch.actions[receipt["collection_index"]]]
            ),
            value_targets=targets,
            optimizer=self.optimizer,
            rules=self.actors.rules,
            encoder=self.encoder,
            seed=sampler_seed,
            device=self.config.device,
            objective=self.config.objective,
            schedule=self.config.schedule,
            guard=guard,
        )
        training["schema"] = "search-acting-supervised-train-v2"
        training["behavior_kl_reference"] = (
            "frozen-raw-network-policy-on-searched-roots"
        )
        self.last_sampler_rng_state = training["sampler_rng_state"]
        self.epoch += 1
        self.optimizer_attempted_updates += training["optimizer_steps_attempted"]
        self.optimizer_accepted_updates += training["optimizer_steps_committed"]
        self.optimizer_rejected_updates += (
            training["optimizer_steps_attempted"]
            - training["optimizer_steps_committed"]
        )
        record = dict(
            schema=SEARCH_ACTING_LEARNER_SCHEMA,
            epoch=self.epoch,
            actor_steps=self.actors.steps,
            fresh_transitions=len(epoch.actions),
            total_fresh_transitions=self.actors.steps * self.config.actors.games,
            optimizer_attempted_updates=self.optimizer_attempted_updates,
            optimizer_accepted_updates=self.optimizer_accepted_updates,
            optimizer_rejected_updates=self.optimizer_rejected_updates,
            collection=json.loads(serialize_policy_epoch(epoch)),
            own_search=ledger,
            training=training,
            target_counts={
                f.name: getattr(targets, f.name)
                for f in fields(targets)
                if f.name != "targets"
            },
            sampler_seed=sampler_seed,
            previous_sample_chain_sha256=self.sample_chain_sha256,
        )
        self.sample_chain_sha256 = hashlib.sha256(
            bytes.fromhex(self.sample_chain_sha256) + canonical(record)
        ).hexdigest()
        record["sample_chain_sha256"] = self.sample_chain_sha256
        self.last_epoch_gzip = gzip.compress(canonical(record), mtime=0)
        self.pending_search_schedule = {}
        self.closed = True
        self.validate()
        guard()
        return record

    def checkpoint(self, directory):
        from harbichess.training.torch_search_acting_checkpoint import (
            save_search_acting_checkpoint,
        )

        return save_search_acting_checkpoint(Path(directory), self)

    @classmethod
    def resume(cls, directory, *, config, input_paths, source_commit):
        from harbichess.training.torch_search_acting_checkpoint import (
            load_search_acting_checkpoint,
        )

        self = cls.fresh(
            config=config, input_paths=input_paths, source_commit=source_commit
        )
        load_search_acting_checkpoint(Path(directory), self)
        return self
