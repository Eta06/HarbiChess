"""Small semi-gradient loss for captured static PV leaves."""

from __future__ import annotations

import torch


def loss(prediction_leaf_mover, target_leaf_mover, gradient_mask):
    if prediction_leaf_mover.ndim != 1 or target_leaf_mover.shape != prediction_leaf_mover.shape:
        raise ValueError("one prediction and TDLeaf target per root row required")
    mask = torch.as_tensor(gradient_mask, dtype=torch.bool, device=prediction_leaf_mover.device)
    if mask.shape != prediction_leaf_mover.shape or not bool(mask.any()):
        raise ValueError("at least one static PV leaf with a gradient is required")
    if (
        not torch.isfinite(prediction_leaf_mover[mask]).all()
        or not torch.isfinite(target_leaf_mover[mask]).all()
    ):
        raise ValueError("finite static-leaf predictions and targets required")
    # Detach targets to retain the semi-gradient TDLeaf update.
    return torch.mean((prediction_leaf_mover[mask] - target_leaf_mover[mask].detach()) ** 2)
