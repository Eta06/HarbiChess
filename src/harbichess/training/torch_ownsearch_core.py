"""Detached own-search CE, fresh complete-game WDL and frozen-base KL; no PPO ratios."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn.functional as F

from harbichess.training.fullgame_own_targets import GameBalancedSampler
from harbichess.training.torch_fullgame_ppo import (
    _tensor_batch,
    _whole_dataset_kls,
    compile_epoch_features,
)


@dataclass(frozen=True, slots=True)
class OwnSearchObjective:
    policy_weight: float
    value_weight: float
    policy_anchor_weight: float
    value_anchor_weight: float
    behavior_kl_stop: float

    def __post_init__(self):
        if (
            any(
                not math.isfinite(v) or v < 0
                for v in (
                    self.policy_weight,
                    self.value_weight,
                    self.policy_anchor_weight,
                    self.value_anchor_weight,
                )
            )
            or not math.isfinite(self.behavior_kl_stop)
            or self.behavior_kl_stop <= 0
        ):
            raise ValueError(
                "own-search loss and KL limits must be explicit finite nonnegative values"
            )


@dataclass(frozen=True, slots=True)
class SearchTrainRow:
    transition: object
    legal_actions: tuple
    behavior_policy: tuple
    base_policy: tuple
    base_value_wdl: tuple
    search_policy: tuple
    source_id: str
    game_index: int
    action_index: int = 0
    advantage: float = 0.0
    target_wdl: tuple = (0.0, 0.0, 0.0)


def search_train_rows(epoch, ledger):
    return tuple(
        SearchTrainRow(
            row.transition,
            row.legal_actions,
            row.behavior_policy,
            row.base_policy,
            row.base_wdl,
            tuple(receipt["search_policy"]),
            row.transition.source_id,
            row.transition.game_index,
        )
        for receipt in ledger["roots"]
        for row in [epoch.actions[receipt["collection_index"]]]
    )


def ownsearch_loss(
    logits,
    wdl_logits,
    mask,
    targets,
    base_policy,
    base_wdl,
    *,
    terminal_logits=None,
    terminal_targets=None,
    config,
):
    if targets.requires_grad or base_policy.requires_grad or base_wdl.requires_grad:
        raise ValueError("search and immutable base targets must be detached")
    if (
        targets.shape != logits.shape
        or (targets[~mask] != 0).any()
        or (targets < 0).any()
        or not torch.allclose(
            targets.sum(1), torch.ones_like(targets.sum(1)), atol=1e-6, rtol=0
        )
    ):
        raise ValueError("search targets require normalized exact legal support")
    log_policy = F.log_softmax(logits.masked_fill(~mask, -torch.inf), dim=1)
    safe_log = torch.where(mask, log_policy, 0.0)
    policy = -(targets * safe_log).sum(1).mean()
    anchor = (
        (base_policy * (base_policy.clamp_min(1e-30).log() - safe_log)).sum(1).mean()
    )
    log_value = F.log_softmax(wdl_logits, dim=1)
    value_anchor = (
        (base_wdl * (base_wdl.clamp_min(1e-30).log() - log_value)).sum(1).mean()
    )
    value = (
        wdl_logits.sum() * 0
        if terminal_logits is None
        else -(terminal_targets.detach() * F.log_softmax(terminal_logits, dim=1))
        .sum(1)
        .mean()
    )
    total = (
        config.policy_weight * policy
        + config.value_weight * value
        + config.policy_anchor_weight * anchor
        + config.value_anchor_weight * value_anchor
    )
    return total, dict(
        policy=policy, value=value, policy_anchor=anchor, value_anchor=value_anchor
    )


def train_search_epoch(
    network,
    *,
    policy_rows,
    value_targets,
    optimizer,
    rules,
    encoder,
    seed,
    device,
    objective,
    schedule,
    guard,
):
    actual_policy_count = len(policy_rows)
    if not policy_rows and not value_targets.targets:
        return dict(
            schema="ownsearch-supervised-train-v1",
            optimizer_steps_attempted=0,
            optimizer_steps_committed=0,
            accepted_passes=0,
            rejected_passes=0,
            trained_transitions=len(value_targets.targets),
            policy_target_rows=0,
            sampler_rng_state=None,
            reason="no-exogenously-selected-root-no-update",
        )
    if not policy_rows:
        # Fresh known outcomes still train when a tiny epoch selects no search root.
        policy_rows = tuple(
            SearchTrainRow(
                row.transition,
                row.legal_actions,
                row.behavior_policy,
                row.base_policy,
                row.base_value_wdl,
                row.behavior_policy,
                row.source_id,
                row.game_index,
            )
            for row in value_targets.targets
        )
        objective = replace(objective, policy_weight=0.0)
    parameters = [p for p in network.parameters() if p.requires_grad]
    features = compile_epoch_features(
        (*policy_rows, *value_targets.targets), encoder, guard=guard, compact=True
    )
    rng = random.Random(seed)
    sampler = (
        GameBalancedSampler(value_targets.targets, seed=seed ^ 0xBA71)
        if value_targets.targets
        else None
    )
    initial = dict(
        policy=rng.getstate(), value=sampler.rng.getstate() if sampler else None
    )
    minibatches = max(1, math.ceil(len(policy_rows) / schedule.minibatch_size))
    attempted = committed = accepted = rejected = 0
    maxnorm = 0.0
    last_losses = None
    before_kl, _ = _whole_dataset_kls(
        network,
        encoder,
        rules,
        policy_rows,
        device=device,
        guard=guard,
        features=features,
    )
    if before_kl > objective.behavior_kl_stop:
        raise ValueError("own-search frozen behavior differs before supervised update")
    for _ in range(schedule.passes):
        guard()
        snapshot = dict(
            model=copy.deepcopy(network.state_dict()),
            optimizer=copy.deepcopy(optimizer.state_dict()),
            python=random.getstate(),
            numpy=copy.deepcopy(np.random.get_state()),
            torch=torch.get_rng_state().clone(),
            cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            policy=rng.getstate(),
            value=sampler.rng.getstate() if sampler else None,
        )
        for _ in range(minibatches):
            guard()
            rows = rng.choices(policy_rows, k=schedule.minibatch_size)
            tensors = _tensor_batch(network, encoder, rules, rows, device, features)
            inputs, indices, mask, _, _, _, _, base, _, base_wdl = tensors
            logits, wdl = network.masked_policy_value(inputs, indices)
            targets = torch.zeros_like(logits)
            for i, row in enumerate(rows):
                targets[i, : len(row.search_policy)] = torch.tensor(
                    row.search_policy, dtype=targets.dtype, device=device
                )
            terminal_logits = terminal_targets = None
            if sampler:
                value_rows = sampler.sample(schedule.minibatch_size)
                value_tensors = _tensor_batch(
                    network, encoder, rules, value_rows, device, features
                )
                _, terminal_logits = network.masked_policy_value(
                    value_tensors[0], value_tensors[1]
                )
                terminal_targets = value_tensors[8]
            loss, components = ownsearch_loss(
                logits,
                wdl,
                mask,
                targets,
                base,
                base_wdl,
                terminal_logits=terminal_logits,
                terminal_targets=terminal_targets,
                config=objective,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, schedule.max_gradient_norm
            )
            if (
                not torch.isfinite(loss)
                or not torch.isfinite(norm)
                or any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in parameters
                )
            ):
                raise RuntimeError("nonfinite own-search loss/gradient")
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in parameters):
                raise RuntimeError("nonfinite own-search parameter")
            attempted += 1
            maxnorm = max(maxnorm, float(norm))
            last_losses = {k: float(v.detach()) for k, v in components.items()}
        behavior_kl, _ = _whole_dataset_kls(
            network,
            encoder,
            rules,
            policy_rows,
            device=device,
            guard=guard,
            features=features,
        )
        if behavior_kl > objective.behavior_kl_stop:
            network.load_state_dict(snapshot["model"], strict=True)
            optimizer.load_state_dict(snapshot["optimizer"])
            optimizer.zero_grad(set_to_none=True)
            random.setstate(snapshot["python"])
            np.random.set_state(snapshot["numpy"])
            torch.set_rng_state(snapshot["torch"])
            if snapshot["cuda"] is not None:
                torch.cuda.set_rng_state_all(snapshot["cuda"])
            rng.setstate(snapshot["policy"])
            if sampler:
                sampler.rng.setstate(snapshot["value"])
            rejected += 1
            break
        committed += minibatches
        accepted += 1
    # Avoid a final RNG-consuming callback after rejected-pass restoration.
    return dict(
        schema="ownsearch-supervised-train-v1",
        optimizer_steps_attempted=attempted,
        optimizer_steps_committed=committed,
        optimizer_steps_discarded=attempted - committed,
        accepted_passes=accepted,
        rejected_passes=rejected,
        trained_transitions=len(value_targets.targets),
        policy_target_rows=actual_policy_count,
        minibatches_per_pass=minibatches,
        sampler_rng_state_before_epoch=initial,
        sampler_rng_state=dict(
            policy=rng.getstate(), value=sampler.rng.getstate() if sampler else None
        ),
        max_gradient_norm=maxnorm,
        last_minibatch_losses=last_losses,
        behavior_kl_after_last_attempt=behavior_kl,
        behavior_kl_scope=(
            "game-equal among exogenously searched root states; "
            "not all32768actor rows"
        ),
    )
