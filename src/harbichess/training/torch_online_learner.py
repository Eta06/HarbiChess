"""Experimental fresh one-ply CPU self-play learning with a frozen prior and EMA.

Each batch advances every actor once, consumes those new transitions once, makes
one AdamW update, then updates EMA. No Stockfish or search labels are generated.
This component has no automatic production training or model-promotion policy.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import chess
import numpy as np
import torch
from harbichess.backends.torch_network import load_weights, sha256
from harbichess.chess.actions import legal_action_indices, move_to_action
from harbichess.chess.encoding import ENCODER_CHANNELS, BoardEncoder
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove, ChessState
from harbichess.selfplay.online_actor import ActorOpening, OnlineActorConfig, OnlineActors
from harbichess.training.config import NonFiniteTrainingError
from harbichess.training.online_objective import OnlineObjectiveConfig, OnlineObjectiveTargets
from harbichess.training.online_targets import build_one_ply_target
from harbichess.training.torch_online_checkpoint import (
    load_online_checkpoint,
    save_online_checkpoint,
)
from harbichess.training.torch_online_objective import online_loss

ONLINE_LEARNER_SCHEMA = "torch-fresh-one-ply-v2"


def read_online_train_book(path: Path) -> tuple[ActorOpening, ...]:
    """Use the immutable book's train split only; replay every original prefix."""
    book = json.loads(path.read_text())
    if book.get("schema") != 1 or not book.get("splits", {}).get("train"):
        raise ValueError("online training requires a nonempty version1 train opening split")
    rules = PythonChessRules()
    openings = []
    replayed_fens = {}
    for row in book["splits"]["train"]:
        opening = row["opening"]
        state = ChessState(
            opening.get("root_fen", chess.STARTING_FEN),
            tuple(ChessMove(move) for move in opening["moves"]),
        )
        if row.get("root_ply", state.ply) != state.ply:
            raise ValueError("online source root ply differs from complete history")
        if "fen" in opening:
            if state not in replayed_fens:
                replayed_fens[state] = rules.view(state).fen
            if replayed_fens[state] != opening["fen"]:
                raise ValueError("online source FEN differs from replayed history")
        openings.append(ActorOpening(row["source_game"], state))
    ids = [row.source_id for row in openings]
    if len(set(ids)) != len(ids):
        raise ValueError("online training source-game IDs must be distinct")
    heldout_ids = {
        row["source_game"]
        for name, rows in book["splits"].items()
        if name != "train"
        for row in rows
    }
    if set(ids) & heldout_ids:
        raise ValueError("online train and heldout source-game IDs overlap")
    return tuple(openings)


@dataclass(frozen=True, slots=True)
class TorchOnlineConfig:
    seed: int
    actors: OnlineActorConfig
    objective: OnlineObjectiveConfig
    learning_rate: float
    weight_decay: float
    max_gradient_norm: float
    ema_decay: float
    importance_maximum: float
    device: str = "cpu"

    def __post_init__(self):
        if (
            self.device not in ("cpu", "cuda:0")
            or type(self.seed) is not int
            or self.seed < 0
            or any(
                not math.isfinite(value) or value <= 0
                for value in (self.learning_rate, self.max_gradient_norm, self.importance_maximum)
            )
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
            or not math.isfinite(self.ema_decay)
            or not 0 <= self.ema_decay < 1
        ):
            raise ValueError("invalid explicit online optimizer/EMA configuration")


def _run_config(config, initial_sha256, actors, input_paths):
    configuration = asdict(config)
    if config.device == "cpu":
        configuration.pop("device")  # Preserve existing CPU run-config bytes.
    return {
        "schema": ONLINE_LEARNER_SCHEMA,
        "config": configuration,
        "initial_weights_sha256": initial_sha256,
        "actor_book_sha256": actors.book_sha256,
        "input_sha256": {name: sha256(path) for name, path in input_paths.items()},
        "optimizer": {"class": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8},
        "transfer": "weights-only-warm-start-optimizer-reset",
        "sampling": "normalized-current-policy-fixed-temperature-float64-v1",
    }


def _prepare_device(device):
    if device == "cuda:0":
        # Must precede the first CUDA context/matmul in a fresh process.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in (":4096:8", ":16:8"):
            raise ValueError("deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG")
        if not torch.cuda.is_available():
            raise ValueError("configured cuda:0 is unavailable")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


class TorchOnlineLearner:
    @classmethod
    def fresh(
        cls,
        *,
        config: TorchOnlineConfig,
        input_paths: dict[str, Path],
        source_commit: str,
    ):
        if "initial_weights" not in input_paths or "book" not in input_paths:
            raise ValueError("online learning requires immutable initial_weights and book inputs")
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
            raise ValueError("online learning requires the shared104history/4672action schemas")
        # The inherited material readout is absent from this forward path.
        for name, parameter in self.online.named_parameters():
            parameter.requires_grad_(not name.startswith("material_value_linear."))
        self.base, self.ema = copy.deepcopy(self.online).eval(), copy.deepcopy(self.online).eval()
        self.base.requires_grad_(False)
        self.ema.requires_grad_(False)
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
        self.update = 0
        self.sample_chain_sha256 = hashlib.sha256(b"").hexdigest()
        self.run_config = _run_config(
            config, sha256(input_paths["initial_weights"]), self.actors, input_paths
        )
        return self

    @classmethod
    def resume(
        cls,
        directory: Path,
        *,
        config: TorchOnlineConfig,
        input_paths: dict[str, Path],
        source_commit: str,
    ):
        _prepare_device(config.device)
        openings = read_online_train_book(input_paths["book"])
        provisional = OnlineActors(openings, config=config.actors, rng=random.Random(0))
        expected = _run_config(
            config, sha256(input_paths["initial_weights"]), provisional, input_paths
        )
        loaded = load_online_checkpoint(
            directory,
            expected_run_config=expected,
            expected_input_paths=input_paths,
            expected_source_commit=source_commit,
            device=config.device,
        )
        state = loaded.run_state
        if (
            state["schema"] != ONLINE_LEARNER_SCHEMA
            or state["actors"]["steps"] != loaded.manifest["update"]
            or state["transitions"] != loaded.manifest["update"] * config.actors.games
        ):
            raise ValueError("online training/actor/transition cursors differ")
        self = cls()
        self.config, self.input_paths, self.source_commit = config, dict(input_paths), source_commit
        self.online, self.base, self.ema = loaded.online, loaded.base, loaded.ema
        self.optimizer = loaded.optimizer
        self.actors = OnlineActors(
            openings, config=config.actors, rng=loaded.actor_rng, cursor=state["actors"]
        )
        self.encoder = BoardEncoder(self.actors.rules)
        self.update, self.run_config = state["update"], expected
        self.sample_chain_sha256 = state["sample_chain_sha256"]
        if len(bytes.fromhex(self.sample_chain_sha256)) != 32:
            raise ValueError("invalid online sample-chain cursor")
        self._validate_run_settings()
        return self

    def checkpoint(self, directory: Path):
        self._validate_run_settings()
        if self.actors.steps != self.update:
            raise ValueError(
                "cannot checkpoint an actor batch without its successful optimizer update"
            )
        if self.run_config["input_sha256"] != {
            name: sha256(path) for name, path in self.input_paths.items()
        }:
            raise ValueError("online immutable inputs changed during training")
        return save_online_checkpoint(
            directory,
            online=self.online,
            base=self.base,
            ema=self.ema,
            optimizer=self.optimizer,
            actor_rng=self.actors.rng,
            run_config=self.run_config,
            run_state={
                "schema": ONLINE_LEARNER_SCHEMA,
                "update": self.update,
                "transitions": self.update * self.config.actors.games,
                "sample_chain_sha256": self.sample_chain_sha256,
                "actors": self.actors.cursor(),
            },
            input_paths=self.input_paths,
            source_commit=self.source_commit,
            update=self.update,
        )

    def _validate_run_settings(self):
        group = self.optimizer.param_groups[0]
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
        if len(self.optimizer.param_groups) != 1 or any(
            group.get(key) != value for key, value in expected.items()
        ):
            raise ValueError("online optimizer settings changed from immutable run config")
        if any(
            parameter.requires_grad != (not name.startswith("material_value_linear."))
            for name, parameter in self.online.named_parameters()
        ):
            raise ValueError("online trainable parameters changed from declared used-forward path")

    def _inputs(self, states):
        return torch.tensor(
            np.array(
                [self.encoder.encode(state).values for state in states], dtype=np.float32
            ).reshape(-1, 8, 8, self.online.config.input_channels),
            device=self.config.device,
        )

    def train_update(self) -> dict:
        self._validate_run_settings()
        if self.actors.steps != self.update:
            raise ValueError("partial online step; restore the last complete checkpoint")
        pre = self.actors.states
        legal = [legal_action_indices(self.actors.rules.inspect(state)) for state in pre]
        width = max(map(len, legal))
        masks = np.array([[index < len(row) for index in range(width)] for row in legal])
        indices = torch.tensor(
            [list(row) + [row[0]] * (width - len(row)) for row in legal],
            device=self.config.device,
        )
        inputs = self._inputs(pre)
        policy, value = self.online.masked_policy_value(inputs, indices)
        mask_tensor = torch.tensor(masks, device=self.config.device)
        with torch.no_grad():
            probabilities = policy.masked_fill(~mask_tensor, -torch.inf).softmax(1).cpu().numpy()
            online_wdl = value.softmax(1).cpu().numpy()
            base_policy, base_value = self.base.masked_policy_value(inputs, indices)
            base_probabilities = (
                base_policy.masked_fill(~mask_tensor, -torch.inf).softmax(1).cpu().numpy()
            )
            base_wdl = base_value.softmax(1).cpu().numpy()
        transitions = self.actors.step(
            tuple(
                tuple(float(p) for p in row[: len(actions)])
                for row, actions in zip(probabilities, legal, strict=True)
            )
        )
        nonterminal = [row.slot for row in transitions if row.terminal_result is None]
        ema_post = {}
        if nonterminal:
            states = tuple(transitions[index].post for index in nonterminal)
            first_actions = torch.tensor(
                [[legal_action_indices(self.actors.rules.inspect(state))[0]] for state in states],
                device=self.config.device,
            )
            with torch.no_grad():
                _, ema_value = self.ema.masked_policy_value(self._inputs(states), first_actions)
                ema_wdl = ema_value.softmax(1).cpu().numpy()
            ema_post = dict(zip(nonterminal, ema_wdl, strict=True))
        target_rows = [
            build_one_ply_target(
                self.actors.rules,
                row.pre,
                row.action,
                online_pre_wdl=tuple(float(x) for x in online_wdl[row.slot]),
                ema_post_wdl=tuple(float(x) for x in ema_post[row.slot])
                if row.slot in ema_post
                else None,
                policy_probability=row.policy_probability,
                behavior_probability=row.behavior_probability,
                importance_maximum=self.config.importance_maximum,
                claim_draw=self.config.actors.claim_draw,
                rollout_cutoff=row.rollout_cutoff,
            )
            for row in transitions
        ]
        chosen = np.array(
            [
                legal[row.slot].index(
                    move_to_action(
                        self.actors.rules.inspect(row.pre), chess.Move.from_uci(row.action.uci)
                    )
                )
                for row in transitions
            ]
        )
        targets = OnlineObjectiveTargets(
            legal_masks=masks,
            actions=chosen,
            advantages=np.array([row.advantage for row in target_rows], dtype=np.float32),
            importance=np.array([row.importance for row in target_rows], dtype=np.float32),
            target_wdl=np.array([row.mover_wdl for row in target_rows], dtype=np.float32),
            base_policy=base_probabilities,
            base_wdl=base_wdl,
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss = online_loss(policy, value, targets, self.config.objective)
        loss.total.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            self.online.parameters(), self.config.max_gradient_norm
        )
        if (
            not all(math.isfinite(float(x.detach())) for x in loss)
            or not torch.isfinite(norm)
            or any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in self.online.parameters()
            )
        ):
            self.optimizer.zero_grad(set_to_none=True)
            raise NonFiniteTrainingError(
                "online nonfinite loss/gradient; restore last full checkpoint"
            )
        self.optimizer.step()
        if any(not torch.isfinite(p).all() for p in self.online.parameters()):
            raise NonFiniteTrainingError(
                "online nonfinite optimizer update; restore full checkpoint"
            )
        with torch.no_grad():
            for target, current in zip(
                self.ema.parameters(), self.online.parameters(), strict=True
            ):
                if current.requires_grad:
                    target.mul_(self.config.ema_decay).add_(
                        current, alpha=1 - self.config.ema_decay
                    )
        self.update += 1
        samples = []
        for row, label in zip(transitions, target_rows, strict=True):
            slot = row.slot
            samples.append(
                {
                    "slot": slot,
                    "game_index": row.game_index,
                    "source_id": row.source_id,
                    "root_fen": row.pre.root_fen,
                    "pre_moves": [move.uci for move in row.pre.moves],
                    "action": row.action.uci,
                    "legal_actions": list(legal[slot]),
                    "online_policy": [float(x) for x in probabilities[slot, : len(legal[slot])]],
                    "base_policy": [float(x) for x in base_probabilities[slot, : len(legal[slot])]],
                    "online_pre_wdl": [float(x) for x in online_wdl[slot]],
                    "base_wdl": [float(x) for x in base_wdl[slot]],
                    "ema_post_wdl": [float(x) for x in ema_post[slot]]
                    if slot in ema_post
                    else None,
                    "mover_target_wdl": list(label.mover_wdl),
                    "target_schema": label.schema,
                    "target_source": label.target_source,
                    "advantage": label.advantage,
                    "importance": label.importance,
                    "policy_probability": row.policy_probability,
                    "behavior_probability": row.behavior_probability,
                    "rollout_cutoff": row.rollout_cutoff,
                    "terminal_result": row.terminal_result,
                    "terminal_termination": row.terminal_termination,
                }
            )
        record = {
            "schema": ONLINE_LEARNER_SCHEMA,
            "update": self.update,
            "policy_model_update": self.update - 1,
            "loss": {
                name: float(value.detach()) for name, value in zip(loss._fields, loss, strict=True)
            },
            "unclipped_gradient_norm": float(norm),
            "gradient_norm": min(float(norm), self.config.max_gradient_norm),
            "transitions": len(transitions),
            "terminated_games": dict(self.actors.terminations),
            "samples": samples,
        }
        previous = self.sample_chain_sha256
        encoded = json.dumps(
            record, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        self.sample_chain_sha256 = hashlib.sha256(bytes.fromhex(previous) + encoded).hexdigest()
        return record | {
            "previous_sample_chain_sha256": previous,
            "sample_chain_sha256": self.sample_chain_sha256,
        }
