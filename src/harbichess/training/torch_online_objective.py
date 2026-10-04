"""Differentiable Torch implementation of the shared experimental online loss."""

import torch

from harbichess.training.online_objective import (
    OnlineLoss,
    OnlineObjectiveConfig,
    OnlineObjectiveTargets,
)


def online_loss(
    policy_logits: torch.Tensor,
    value_logits: torch.Tensor,
    targets: OnlineObjectiveTargets,
    config: OnlineObjectiveConfig,
) -> OnlineLoss:
    if (
        tuple(policy_logits.shape) != targets.legal_masks.shape
        or tuple(value_logits.shape) != targets.target_wdl.shape
        or policy_logits.device != value_logits.device
        or policy_logits.dtype != value_logits.dtype
        or policy_logits.dtype not in (torch.float32, torch.float64)
    ):
        raise ValueError("online logits require matching float32/64 shape, dtype and device")
    if not torch.isfinite(policy_logits).all() or not torch.isfinite(value_logits).all():
        raise ValueError("online logits must be finite")

    def constant(array, dtype=policy_logits.dtype):
        # Copy the read-only owned constants; none enters an autograd graph.
        return torch.tensor(array, dtype=dtype, device=policy_logits.device)

    mask = constant(targets.legal_masks, torch.bool)
    chosen = constant(targets.actions, torch.long)
    importance = constant(targets.importance)
    log_policy = torch.log_softmax(policy_logits.masked_fill(~mask, -torch.inf), 1)
    log_value = torch.log_softmax(value_logits, 1)

    def kl(probabilities, log_model):
        support = probabilities > 0
        safe = torch.where(support, probabilities, torch.ones_like(probabilities))
        return (probabilities * (safe.log() - torch.where(support, log_model, 0.0))).sum(1)

    actor = -(
        importance * constant(targets.advantages) * log_policy.gather(1, chosen[:, None])[:, 0]
    ).mean()
    anchor = kl(constant(targets.base_policy), log_policy).mean()
    target = (importance * kl(constant(targets.target_wdl), log_value)).mean()
    base_value = kl(constant(targets.base_wdl), log_value).mean()
    total = (
        actor
        + config.policy_anchor_weight * anchor
        + config.value_target_weight * target
        + config.value_anchor_weight * base_value
    )
    return OnlineLoss(total, actor, anchor, target, base_value)
