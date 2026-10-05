"""Deterministic grouped-row learner and exact fresh-object resume check."""

from __future__ import annotations

import hashlib
import json

import numpy as np
from residual_critic18 import INPUTS, ResidualCritic18

TRAINING_SETTINGS = {
    "updates": 64,
    "batch_size": 256,
    "learning_rate": 1e-3,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8,
    "clip_norm": 5.0,
    "anchor_weight": 0.1,
    "weight_l2": 1e-4,
}


def validate_contract(contract):
    if not isinstance(contract, dict):
        raise ValueError("full immutable training contract required")
    for key, value in TRAINING_SETTINGS.items():
        if contract.get(key) != value:
            raise ValueError(f"frozen optimizer setting differs: {key}")
    for key, size in (
        ("prior_sha256", 64),
        ("labels_sha256", 64),
        ("training_dataset_sha256", 64),
        ("feature_helper_sha256", 64),
        ("source_commit", 40),
    ):
        text = contract.get(key)
        if not isinstance(text, str) or len(text) != size:
            raise ValueError(f"missing immutable provenance: {key}")
        int(text, 16)
    if contract.get("input_count") != INPUTS or contract.get("hidden_count") != 32:
        raise ValueError("residual model architecture differs")


def _rows(groups):
    if not isinstance(groups, dict) or len(groups) < 16:
        raise ValueError("at least 16 complete training trajectories required")
    canonical = {}
    for game in sorted(groups):
        records = []
        if not isinstance(game, str) or not groups[game]:
            raise ValueError("empty or invalid trajectory group")
        for record in groups[game]:
            if set(record) != {"x", "prior_logit", "target"}:
                raise ValueError("training row field set differs")
            x = np.asarray(record["x"], dtype=np.float64)
            if x.shape != (INPUTS,) or not np.isfinite(x).all():
                raise ValueError("18 finite features required")
            base, target = float(record["prior_logit"]), float(record["target"])
            if not np.isfinite(base) or not np.isfinite(target) or abs(target) > 1:
                raise ValueError("finite base and clipped own-search value required")
            records.append((x, base, target))
        canonical[game] = records
    if sum(map(len, canonical.values())) != 1024:
        raise ValueError("exactly 1024 own-search roots required")
    return canonical


class ResidualTrainer:
    def __init__(self, model: ResidualCritic18, groups):
        validate_contract(model.contract)
        self.model = model
        self.groups = _rows(groups)
        self.game_keys = sorted(self.groups)

    def advance(self, target_step, *, batch_size=256, guard=lambda: None):
        updates = TRAINING_SETTINGS["updates"]
        if not isinstance(target_step, int) or not self.model.step <= target_step <= updates:
            raise ValueError("backward or excess training step")
        while self.model.step < target_step:
            guard()
            # Draw a group once per item, then draw a row from that same group.
            batch = []
            for _ in range(batch_size):
                key = self.game_keys[self.model.sampler_rng.randrange(len(self.game_keys))]
                rows = self.groups[key]
                batch.append(rows[self.model.sampler_rng.randrange(len(rows))])
            x = np.stack([row[0] for row in batch])
            base = np.asarray([row[1] for row in batch])
            target = np.asarray([row[2] for row in batch])
            self.model.train_batch(
                x,
                base,
                target,
                learning_rate=TRAINING_SETTINGS["learning_rate"],
                beta1=TRAINING_SETTINGS["beta1"],
                beta2=TRAINING_SETTINGS["beta2"],
                epsilon=TRAINING_SETTINGS["epsilon"],
                clip_norm=TRAINING_SETTINGS["clip_norm"],
                anchor_weight=TRAINING_SETTINGS["anchor_weight"],
                weight_l2=TRAINING_SETTINGS["weight_l2"],
            )
        return self.model.native()


def native_digest(native):
    payload = json.dumps(native, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def audit_six_native_loads(native_states, contract):
    """Strictly load six separately frozen native payloads without updates."""
    if len(native_states) != 6:
        raise ValueError("exactly six strict fresh-load receipts required")
    receipts = []
    for index, state in enumerate(native_states):
        model = ResidualCritic18(seed=10_000 + index, contract=contract, state=state)
        loaded = model.native()
        if native_digest(loaded) != native_digest(state):
            raise ValueError("native load/save changed stored state")
        receipts.append(native_digest(loaded))
    return receipts


def qualify_resume(seed, contract, groups, *, split_step=4, final_step=8):
    """Synthetic-sized protocol proof; caller supplies only synthetic groups."""
    validate_contract(contract)
    whole = ResidualCritic18(seed=seed, contract=contract)
    ResidualTrainer(whole, groups).advance(final_step)

    paused = ResidualCritic18(seed=seed, contract=contract)
    ResidualTrainer(paused, groups).advance(split_step)
    wire = json.loads(json.dumps(paused.native(), allow_nan=False))
    resumed = ResidualCritic18(seed=seed + 1, contract=contract, state=wire)
    ResidualTrainer(resumed, groups).advance(final_step)
    left, right = whole.native(), resumed.native()
    if native_digest(left) != native_digest(right):
        raise ValueError("whole and fresh-object resume diverged")
    return {"whole_sha256": native_digest(left), "resume_sha256": native_digest(right)}
