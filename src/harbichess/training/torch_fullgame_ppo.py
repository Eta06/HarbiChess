"""Clipped own-game policy updates with behavior/base KL and terminal WDL."""

from __future__ import annotations

import copy
import hashlib
import math
import random
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
from torch.nn import functional as F

from harbichess.chess.encoding import BoardEncoder
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessState
from harbichess.selfplay.online_epoch import EpochPolicyOutput
from harbichess.training.config import NonFiniteTrainingError
from harbichess.training.fullgame_own_targets import (
    FullGameTargetSet,
    GameBalancedSampler,
)


def torch_model_digest(network) -> str:
    """Hash an inference snapshot's exact named state for epoch provenance."""
    digest = hashlib.sha256()
    for name, tensor in sorted(network.state_dict().items()):
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(bytes(value.numpy().tobytes()))
    return digest.hexdigest()


def make_torch_epoch_inference(behavior, base, encoder: BoardEncoder, *, device: str):
    """Build CUDA/CPU batched inference from immutable behavior and e8 base nets."""
    if behavior.specification != base.specification:
        raise ValueError("behavior and e8 base networks must share the native architecture")
    if str(next(behavior.parameters()).device) != device or str(
        next(base.parameters()).device
    ) != device:
        raise ValueError("behavior/base networks must already be on the configured device")
    behavior.eval()
    base.eval()
    behavior.requires_grad_(False)
    base.requires_grad_(False)

    def infer(states: tuple[ChessState, ...], legal: tuple[tuple[int, ...], ...]):
        width = max(len(row) for row in legal)
        actions = torch.tensor(
            [list(row) + [row[0]] * (width - len(row)) for row in legal],
            dtype=torch.long,
            device=device,
        )
        mask = torch.tensor(
            [[index < len(row) for index in range(width)] for row in legal],
            dtype=torch.bool,
            device=device,
        )
        inputs = torch.tensor(
            np.array([encoder.encode(state).values for state in states], dtype=np.float32)
            .reshape(-1, 8, 8, behavior.config.input_channels),
            device=device,
        )
        with torch.inference_mode():
            policy_logits, value_logits = behavior.masked_policy_value(inputs, actions)
            base_logits, base_value_logits = base.masked_policy_value(inputs, actions)
            policy = policy_logits.masked_fill(~mask, -torch.inf).softmax(1).cpu().numpy()
            wdl = value_logits.softmax(1).cpu().numpy()
            base_policy = base_logits.masked_fill(~mask, -torch.inf).softmax(1).cpu().numpy()
            base_wdl = base_value_logits.softmax(1).cpu().numpy()
        return EpochPolicyOutput(
            tuple(tuple(float(x) for x in row[: len(support)])
                  for row, support in zip(policy, legal, strict=True)),
            tuple(tuple(float(x) for x in row) for row in wdl),
            tuple(tuple(float(x) for x in row[: len(support)])
                  for row, support in zip(base_policy, legal, strict=True)),
            tuple(tuple(float(x) for x in row) for row in base_wdl),
        )

    return infer


@dataclass(frozen=True, slots=True)
class FullGamePPOConfig:
    policy_clip: float
    behavior_kl_stop: float
    policy_anchor_weight: float
    value_weight: float
    value_anchor_weight: float

    def __post_init__(self):
        values = (
            self.policy_clip,
            self.behavior_kl_stop,
            self.policy_anchor_weight,
            self.value_weight,
            self.value_anchor_weight,
        )
        if any(not math.isfinite(x) or x < 0 for x in values) or (
            self.policy_clip <= 0 or self.behavior_kl_stop <= 0
        ):
            raise ValueError("invalid clipped full-game PPO/KL configuration")


class FullGamePPOLoss(NamedTuple):
    total: torch.Tensor
    policy: torch.Tensor
    terminal_wdl: torch.Tensor
    base_policy_kl: torch.Tensor
    behavior_policy_kl: torch.Tensor
    base_value_kl: torch.Tensor


def behavior_kl_exceeded(loss: FullGamePPOLoss, config: FullGamePPOConfig) -> bool:
    """The optimizer must skip/stop before an update that breaches this KL gate."""
    return float(loss.behavior_policy_kl.detach()) > config.behavior_kl_stop


def fullgame_ppo_loss(
    policy_logits: torch.Tensor,
    value_logits: torch.Tensor,
    *,
    legal_masks: torch.Tensor,
    actions: torch.Tensor,
    old_action_probabilities: torch.Tensor,
    advantages: torch.Tensor,
    behavior_policy: torch.Tensor,
    base_policy: torch.Tensor,
    terminal_wdl: torch.Tensor,
    base_wdl: torch.Tensor,
    config: FullGamePPOConfig,
) -> FullGamePPOLoss:
    """PPO-style per-decision surrogate; not an unbiased trajectory ratio.

    `behavior_policy` and `base_policy` share exactly the current legal action
    support. `behavior_policy` supplies a KL stop diagnostic, while the e8
    `base_policy` and `base_wdl` are the persistent run anchors.
    """
    size, action_count = policy_logits.shape
    if (
        legal_masks.shape != (size, action_count)
        or behavior_policy.shape != (size, action_count)
        or base_policy.shape != (size, action_count)
        or value_logits.shape != (size, 3)
        or terminal_wdl.shape != (size, 3)
        or base_wdl.shape != (size, 3)
        or actions.shape != (size,)
        or old_action_probabilities.shape != (size,)
        or advantages.shape != (size,)
        or size == 0
        or not (policy_logits.device == value_logits.device == legal_masks.device
                == actions.device == old_action_probabilities.device == advantages.device
                == behavior_policy.device == base_policy.device == terminal_wdl.device
                == base_wdl.device)
        or not (policy_logits.dtype == value_logits.dtype == behavior_policy.dtype
                == base_policy.dtype == terminal_wdl.dtype == base_wdl.dtype)
        or policy_logits.dtype not in (torch.float32, torch.float64)
    ):
        raise ValueError("full-game PPO tensors have incompatible shapes")
    if any(
        value.requires_grad
        for value in (
            old_action_probabilities,
            advantages,
            behavior_policy,
            base_policy,
            terminal_wdl,
            base_wdl,
        )
    ):
        raise ValueError("full-game PPO behavior/target tensors must be detached constants")
    if not legal_masks.dtype == torch.bool or not legal_masks.any(dim=1).all():
        raise ValueError("every full-game PPO row requires boolean legal support")
    if (
        (actions < 0).any()
        or (actions >= action_count).any()
        or not legal_masks.gather(1, actions[:, None]).all()
        or (old_action_probabilities <= 0).any()
        or not torch.isfinite(old_action_probabilities).all()
        or not torch.isfinite(advantages).all()
        or not torch.equal(behavior_policy.masked_select(~legal_masks), torch.zeros_like(
            behavior_policy.masked_select(~legal_masks)
        ))
        or not torch.equal(base_policy.masked_select(~legal_masks), torch.zeros_like(
            base_policy.masked_select(~legal_masks)
        ))
        or not torch.allclose(
            old_action_probabilities,
            behavior_policy.gather(1, actions[:, None])[:, 0],
            atol=1e-6,
            rtol=0,
        )
    ):
        raise ValueError("full-game PPO chosen action or old behavior mass is invalid")
    for name, probabilities in (
        ("behavior", behavior_policy),
        ("base", base_policy),
        ("terminal WDL", terminal_wdl),
        ("base WDL", base_wdl),
    ):
        if (
            not torch.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or not torch.allclose(
                probabilities.sum(dim=1),
                torch.ones(size, device=probabilities.device, dtype=probabilities.dtype),
                atol=1e-5,
                rtol=0,
            )
        ):
            raise ValueError(f"full-game PPO {name} targets must be normalized probabilities")
    if not torch.isfinite(policy_logits).all() or not torch.isfinite(value_logits).all():
        raise ValueError("full-game PPO logits must be finite")
    log_policy = F.log_softmax(policy_logits.masked_fill(~legal_masks, -torch.inf), dim=1)
    log_behavior = torch.where(behavior_policy > 0, behavior_policy.clamp_min(1e-30).log(), 0.0)
    log_base = torch.where(base_policy > 0, base_policy.clamp_min(1e-30).log(), 0.0)

    def kl(target, log_target, log_model):
        support = target > 0
        safe_model = torch.where(support, log_model, 0.0)
        safe_target = torch.where(support, log_target, 0.0)
        return (target * (safe_target - safe_model)).sum(dim=1).mean()

    action_probability = log_policy.gather(1, actions[:, None]).exp()[:, 0]
    ratio = action_probability / old_action_probabilities
    clipped = ratio.clamp(1.0 - config.policy_clip, 1.0 + config.policy_clip)
    policy = -torch.minimum(ratio * advantages, clipped * advantages).mean()
    critic = -(terminal_wdl * F.log_softmax(value_logits, dim=1)).sum(dim=1).mean()
    base_policy_kl = kl(base_policy, log_base, log_policy)
    behavior_kl = kl(behavior_policy, log_behavior, log_policy)
    base_value_kl = kl(
        base_wdl,
        torch.where(base_wdl > 0, base_wdl.clamp_min(1e-30).log(), 0.0),
        F.log_softmax(value_logits, dim=1),
    )
    total = (
        policy
        + config.value_weight * critic
        + config.policy_anchor_weight * base_policy_kl
        + config.value_anchor_weight * base_value_kl
    )
    return FullGamePPOLoss(total, policy, critic, base_policy_kl, behavior_kl, base_value_kl)


@dataclass(frozen=True, slots=True)
class FullGamePPOTrainConfig:
    minibatch_size: int
    passes: int
    max_gradient_norm: float

    def __post_init__(self):
        if (
            type(self.minibatch_size) is not int
            or self.minibatch_size <= 0
            or type(self.passes) is not int
            or self.passes <= 0
            or not math.isfinite(self.max_gradient_norm)
            or self.max_gradient_norm <= 0
        ):
            raise ValueError("invalid full-game PPO training schedule")


def _tensor_batch(network, encoder, rules, rows, device):
    size = len(rows)
    width = max(len(row.legal_actions) for row in rows)
    input_array = np.array(
        [encoder.encode(row.transition.pre).values for row in rows], dtype=np.float32
    ).reshape(size, 8, 8, network.config.input_channels)
    indices = np.zeros((size, width), dtype=np.int64)
    mask = np.zeros((size, width), dtype=np.bool_)
    behavior = np.zeros((size, width), dtype=np.float32)
    base = np.zeros((size, width), dtype=np.float32)
    actions = np.zeros(size, dtype=np.int64)
    old_probabilities = np.zeros(size, dtype=np.float32)
    advantages = np.zeros(size, dtype=np.float32)
    targets = np.zeros((size, 3), dtype=np.float32)
    base_wdl = np.zeros((size, 3), dtype=np.float32)
    for index, row in enumerate(rows):
        count = len(row.legal_actions)
        indices[index, :count] = row.legal_actions
        indices[index, count:] = row.legal_actions[0]
        mask[index, :count] = True
        behavior[index, :count] = row.behavior_policy
        base[index, :count] = row.base_policy
        actions[index] = row.action_index
        old_probabilities[index] = row.behavior_policy[row.action_index]
        advantages[index] = row.advantage
        targets[index] = row.target_wdl
        base_wdl[index] = row.base_value_wdl
    return (
        torch.tensor(input_array, device=device),
        torch.tensor(indices, device=device),
        torch.tensor(mask, device=device),
        torch.tensor(actions, device=device),
        torch.tensor(old_probabilities, device=device),
        torch.tensor(advantages, device=device),
        torch.tensor(behavior, device=device),
        torch.tensor(base, device=device),
        torch.tensor(targets, device=device),
        torch.tensor(base_wdl, device=device),
    )


def _whole_dataset_kls(network, encoder, rules, targets, *, device, guard=None):
    """Mean over games, then positions, so long episodes do not dominate KL."""
    grouped = {}
    for row in targets:
        grouped.setdefault((row.source_id, row.game_index), []).append(row)
    behavior_game_kl, base_game_kl = [], []
    was_training = network.training
    network.eval()
    with torch.no_grad():
        for rows in grouped.values():
            values = []
            for start in range(0, len(rows), 128):
                if guard is not None:
                    guard()
                tensors = _tensor_batch(network, encoder, rules, rows[start:start + 128], device)
                inputs, indices, mask, _, _, _, behavior, base, _, _ = tensors
                logits, _ = network.masked_policy_value(inputs, indices)
                log_policy = F.log_softmax(logits.masked_fill(~mask, -torch.inf), dim=1)
                safe_behavior = torch.where(
                    behavior > 0, behavior.clamp_min(1e-30).log(), 0.0
                )
                safe_base = torch.where(base > 0, base.clamp_min(1e-30).log(), 0.0)
                behavior_model = torch.where(behavior > 0, log_policy, 0.0)
                base_model = torch.where(base > 0, log_policy, 0.0)
                behavior_kl = (behavior * (safe_behavior - behavior_model)).sum(dim=1)
                base_kl = (base * (safe_base - base_model)).sum(dim=1)
                values.extend(zip(behavior_kl.cpu().tolist(), base_kl.cpu().tolist(), strict=True))
            behavior_game_kl.append(math.fsum(row[0] for row in values) / len(values))
            base_game_kl.append(math.fsum(row[1] for row in values) / len(values))
    network.train(was_training)
    return math.fsum(behavior_game_kl) / len(behavior_game_kl), math.fsum(base_game_kl) / len(
        base_game_kl
    )


def train_fullgame_policy_epoch(
    network,
    *,
    targets: FullGameTargetSet,
    optimizer: torch.optim.Optimizer,
    rules: PythonChessRules,
    encoder: BoardEncoder,
    seed: int,
    device: str,
    objective: FullGamePPOConfig,
    schedule: FullGamePPOTrainConfig,
    guard=None,
) -> dict:
    """Train one closed replay epoch with a hard whole-buffer behavior-KL gate.

    Each accepted pass uses game-balanced position minibatches. If a full-buffer
    post-pass behavior KL exceeds the registered threshold, that entire pass's
    model and Adam state are rolled back before stopping. This limits drift on
    collected paths; the separate tactical/game strength gates remain necessary.
    """
    if not targets.targets or type(seed) is not int or seed < 0:
        raise ValueError("PPO requires completed-game targets and an explicit seed")
    parameters = [parameter for parameter in network.parameters() if parameter.requires_grad]
    if not parameters or any(
        parameter.device.type != device.split(":")[0] for parameter in parameters
    ):
        raise ValueError("PPO network must have trainable parameters on the configured device")
    sampler = GameBalancedSampler(targets.targets, seed=seed)
    sampler_rng_initial = sampler.rng.getstate()
    minibatches_per_pass = max(1, math.ceil(len(targets.targets) / schedule.minibatch_size))
    if guard is not None:
        guard()
    before_behavior_kl, before_base_kl = _whole_dataset_kls(
        network, encoder, rules, targets.targets, device=device, guard=guard
    )
    if before_behavior_kl > objective.behavior_kl_stop:
        raise ValueError("collection behavior snapshot differs beyond the full-buffer KL gate")
    accepted, rejected, attempted_steps, committed_steps = 0, 0, 0, 0
    max_gradient_norm = 0.0
    last_losses = None
    network.train()
    for _ in range(schedule.passes):
        if guard is not None:
            guard()
        model_before = copy.deepcopy(network.state_dict())
        optimizer_before = copy.deepcopy(optimizer.state_dict())
        python_rng_before = random.getstate()
        numpy_rng_before = copy.deepcopy(np.random.get_state())
        torch_rng_before = torch.get_rng_state().clone()
        cuda_rng_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        sampler_rng_before = sampler.rng.getstate()
        for _ in range(minibatches_per_pass):
            if guard is not None:
                guard()
            rows = sampler.sample(schedule.minibatch_size)
            inputs, indices, mask, actions, old_p, advantages, behavior, base, wdl, base_wdl = (
                _tensor_batch(network, encoder, rules, rows, device)
            )
            logits, value_logits = network.masked_policy_value(inputs, indices)
            loss = fullgame_ppo_loss(
                logits,
                value_logits,
                legal_masks=mask,
                actions=actions,
                old_action_probabilities=old_p,
                advantages=advantages,
                behavior_policy=behavior,
                base_policy=base,
                terminal_wdl=wdl,
                base_wdl=base_wdl,
                config=objective,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.total.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, schedule.max_gradient_norm)
            if (
                not all(torch.isfinite(value.detach()) for value in loss)
                or not torch.isfinite(norm)
                or any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                       for parameter in parameters)
            ):
                optimizer.zero_grad(set_to_none=True)
                raise NonFiniteTrainingError("nonfinite full-game PPO loss/gradient")
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in parameters):
                raise NonFiniteTrainingError("nonfinite full-game PPO parameter update")
            attempted_steps += 1
            max_gradient_norm = max(max_gradient_norm, float(norm))
            last_losses = {name: float(value.detach()) for name, value in zip(
                loss._fields, loss, strict=True
            )}
        if guard is not None:
            guard()
        behavior_kl, base_kl = _whole_dataset_kls(
            network, encoder, rules, targets.targets, device=device, guard=guard
        )
        if behavior_kl > objective.behavior_kl_stop:
            network.load_state_dict(model_before, strict=True)
            optimizer.load_state_dict(optimizer_before)
            optimizer.zero_grad(set_to_none=True)
            random.setstate(python_rng_before)
            np.random.set_state(numpy_rng_before)
            torch.set_rng_state(torch_rng_before)
            if cuda_rng_before is not None:
                torch.cuda.set_rng_state_all(cuda_rng_before)
            sampler.rng.setstate(sampler_rng_before)
            rejected += 1
            break
        accepted += 1
        committed_steps += minibatches_per_pass
    if guard is not None:
        guard()
    final_behavior_kl, final_base_kl = _whole_dataset_kls(
        network, encoder, rules, targets.targets, device=device, guard=guard
    )
    if rejected:
        # The final audit is observational too: if its guard callback advances
        # RNG state, leave the caller at the exact rejected-pass boundary.
        random.setstate(python_rng_before)
        np.random.set_state(numpy_rng_before)
        torch.set_rng_state(torch_rng_before)
        if cuda_rng_before is not None:
            torch.cuda.set_rng_state_all(cuda_rng_before)
        sampler.rng.setstate(sampler_rng_before)
    return {
        "schema": "fullgame-terminal-ppo-train-v1",
        "completed_games": targets.complete_games,
        "trained_transitions": len(targets.targets),
        "unknown_cap_games": targets.normal_cap_games,
        "policy_epoch_truncations": targets.epoch_truncated_games,
        "accepted_passes": accepted,
        "rejected_passes": rejected,
        "optimizer_steps_attempted": attempted_steps,
        "optimizer_steps_committed": committed_steps,
        "optimizer_steps_discarded": attempted_steps - committed_steps,
        "minibatches_per_pass": minibatches_per_pass,
        "initial_behavior_kl": before_behavior_kl,
        "final_behavior_kl": final_behavior_kl,
        "initial_base_policy_kl": before_base_kl,
        "final_base_policy_kl": final_base_kl,
        "max_unclipped_gradient_norm": max_gradient_norm,
        "last_minibatch_losses": last_losses,
        "sampler_rng_state": sampler.rng.getstate(),
        "sampler_rng_state_before_epoch": sampler_rng_initial,
    }
