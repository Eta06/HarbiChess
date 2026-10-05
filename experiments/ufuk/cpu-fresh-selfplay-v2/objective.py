"""Own-terminal WDL objective matching deployed additive-e8 composition."""

import torch
import torch.nn.functional as F


def additive_own_mc_loss(base_wdl, residual_logits, labels, anchor_kl_beta=1.0):
    if base_wdl.ndim != 2 or base_wdl.shape != (len(labels), 3):
        raise ValueError("one aligned base WDL triple is required per completed own row")
    if residual_logits.shape != base_wdl.shape:
        raise ValueError("residual WDL logits must align with fixed e8 anchors")
    if not torch.isfinite(base_wdl).all() or not torch.isfinite(residual_logits).all():
        raise ValueError("nonfinite additive value input")
    if torch.any(base_wdl < 0) or not torch.allclose(
        base_wdl.sum(1), torch.ones(len(labels), device=base_wdl.device), atol=2e-6, rtol=0
    ):
        raise ValueError("fixed e8 anchors must be normalized probabilities")
    if not torch.isfinite(torch.tensor(anchor_kl_beta)) or anchor_kl_beta < 0:
        raise ValueError("anchor KL weight must be finite and nonnegative")
    base_log = base_wdl.clamp_min(1e-30).log()
    logits = base_log + residual_logits
    learned_log = F.log_softmax(logits, dim=1)
    own_mc = F.nll_loss(learned_log, labels)
    fixed_base_kl = (base_wdl * (base_log - learned_log)).sum(1).mean()
    return own_mc + anchor_kl_beta * fixed_base_kl, own_mc, fixed_base_kl, logits
