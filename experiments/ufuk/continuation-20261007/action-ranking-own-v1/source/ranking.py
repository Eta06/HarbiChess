"""Legal-child search-policy imitation with a scalar selected-Q anchor."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

TEMPERATURE = 0.25
CE_WEIGHT = 0.1


def objective(model, roots):
    """Compute one root-balanced CE plus selected-child Q loss.

    Each root supplies every legal move as a candidate class. Only the
    selected move has a Q target. Exact terminal child values are constants;
    they are never forwarded through the model.
    """
    if not roots:
        raise ValueError("nonempty root batch required")
    flat = []
    layout = []
    for root in roots:
        candidates = root["candidates"]
        selected = root["selected_index"]
        if not candidates or type(selected) is not int or not 0 <= selected < len(candidates):
            raise ValueError("legal candidate list and selected index required")
        local = []
        for candidate in candidates:
            if candidate["terminal"]:
                local.append((None, float(candidate["child_wdl"])))
            else:
                local.append((len(flat), None))
                flat.append(candidate)
        layout.append(local)

    if flat:
        ids, offsets, prior, _ = tensor_features(flat)
        predicted = torch.tanh(prior + model.residual(ids, offsets))
    else:
        predicted = None
    zero = next(model.parameters()).sum() * 0.0

    root_losses = []
    for root, local in zip(roots, layout, strict=True):
        values = []
        for flat_index, terminal_value in local:
            child_value = zero + terminal_value if flat_index is None else predicted[flat_index]
            values.append(-child_value / TEMPERATURE)
        logits = torch.stack(values).reshape(1, -1)
        selected = root["selected_index"]
        action_ce = F.cross_entropy(
            logits,
            torch.tensor([selected], dtype=torch.long),
        )
        chosen = local[selected][0]
        if chosen is None:
            q_anchor = logits.sum() * 0.0
        else:
            q_anchor = (predicted[chosen] - float(root["q_target"])) ** 2
        root_losses.append(q_anchor + CE_WEIGHT * action_ce)
    return torch.stack(root_losses).mean()


def tensor_features(rows):
    ids, offsets = [], []
    for row in rows:
        raw = row["indices"]
        if (
            not 1 <= len(raw) <= 32
            or len(set(raw)) != len(raw)
            or any(type(i) is not int or not 0 <= i < VOCAB for i in raw)
            or raw != sorted(raw)
            or not math.isfinite(row["prior_logit"])
        ):
            raise ValueError("finite legal child sparse feature packet required")
        offsets.append(len(ids))
        ids.extend(raw)
    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(offsets, dtype=torch.long),
        torch.tensor([r["prior_logit"] for r in rows], dtype=torch.float64),
        None,
    )


from model import VOCAB  # noqa: E402
