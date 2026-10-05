"""Prospective own-search scalar consistency on identical completed own-MC rows.

No actors, search, NN inference, training loop, external labels or persistence.
Caller provides pinned v2 journal decoder and exact existing840 feature helper.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

SCHEMA = "fresh-own-search-value-consistency-dataset-v1"
OBJECTIVE = {
    "schema": "fresh-additive-own-MC-search-value-mixed-objective-v1",
    "own_terminal_ce_weight": 0.5,
    "own_root_search_score_mse_weight": 0.5,
    "fixed_e8_anchor_kl_weight": 1.0,
    "search_target": "clamp-recorded-selected-best-move-root-mover-score-to-minus1-plus1",
    "search_score_not_played_exploration_move": True,
    "sampling": "same-completed-trajectory-groups/split/256sampler-slots-as-ownMC-control",
    "unknown_included": False,
    "teacher_labels": False,
}


def normalize_score(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("recorded own root search score must be finite")
    return min(1.0, max(-1.0, float(value)))


def extract(state, config, journal, feature_function, guard=lambda: None):
    if journal.SCHEMA != "fresh-qsearch-selfplay-journal-v2":
        raise ValueError("pinned fullhistory anchor journal v2 required")
    before = journal.digest(state)
    packets = journal.replay(state, config)  # ALL known/UNKNOWN/tail legality and actor RNG.
    x, y, anchors, groups, baseline_provenance = journal.shrink840(state, config, feature_function)
    rows, raw, normalized = [], [], []
    for source, game_index, positions, result in packets:
        guard()
        game = state["games"][game_index]
        root = config["roots"][game["root_index"]]
        if game["game_index"] != game_index or root["source_id"] != source:
            raise ValueError("known fullhistory game alignment differs")
        if len(positions) != len(game["moves"]) or result == "UNKNOWN":
            raise ValueError("exact complete known-row alignment required")
        for ordinal, ((board, mover, _anchor), row) in enumerate(
            zip(positions, game["moves"], strict=True)
        ):
            guard()
            if row["mover"] != mover or row["pre_ply"] != board.ply():
                raise ValueError("root mover/premove score alignment differs")
            value = row["search_value"]
            raw.append(value)
            normalized.append(normalize_score(value))
            rows.append(
                {
                    "source_id": source,
                    "game_index": game_index,
                    "row_in_game": ordinal,
                    "root_fen": root["root_fen"],
                    "full_pre_history": [m.uci() for m in board.move_stack],
                    "pre_fen": board.fen(),
                    "mover": mover,
                    "selected_best_move": row["selected"],
                    "played_action": row["action"],
                    "explored": row["explored"],
                    "actual_mu": row["mu"],
                    "search_nodes": row["nodes"],
                    "search_evaluations": row["evaluations"],
                    "search_depth": row["depth"],
                    "search_root_actions": row["root_actions"],
                    "raw_search_score": value,
                    "normalized_search_score": normalize_score(value),
                    "target_perspective": "pre-action-root-mover-no-sign-flip",
                    "search_target_action": "selected_best_move-not-played-exploration-action",
                    "observed_complete_result": result,
                }
            )
    if len(rows) != len(y) or len(anchors) != len(y):
        raise ValueError("MC/search/anchor rows must be identical and in identical order")
    if journal.digest(state) != before:
        raise ValueError("source journal changed during extraction")
    raw = np.asarray(raw, dtype=np.float64)
    normalized = np.asarray(normalized, dtype=np.float32)
    for array in (x, y, anchors, raw, normalized):
        array.setflags(write=False)
    receipt = {
        "schema": SCHEMA,
        "journal_state_sha256": before,
        "config_sha256": journal.digest(config),
        "source_commit": config["source_commit"],
        "actor_model_sha256": config["model_sha256"],
        "frozen_e8_anchor_sha256": config["anchor_model_sha256"],
        "search_helper_sha256": config["search_helper_sha256"],
        "value_helper_sha256": config["value_helper_sha256"],
        "anchor_helper_sha256": config["anchor_helper_sha256"],
        "source_journal_producer_sha256": config["producer_sha256"],
        "all_actions_replayed": state["actions"],
        "known_rows": len(y),
        "excluded_UNKNOWN_or_tail_rows": state["actions"] - len(y),
        "complete_games": len(groups),
        "external_teacher_labels": False,
        "search_or_NN_executed": False,
        "no_offpolicy_unbiased_return_claim": True,
    }
    return {
        "x": x,
        "y": y,
        "anchors": anchors,
        "groups": groups,
        "baseline_provenance": baseline_provenance,
        "rows": rows,
        "raw_search": raw,
        "normalized_search": normalized,
        "receipt": receipt,
    }


def mixed_loss(base_wdl, residual_logits, labels, normalized_search, shrink_regularization):
    if (
        base_wdl.ndim != 2
        or base_wdl.shape != residual_logits.shape
        or base_wdl.shape != (len(labels), 3)
        or len(labels) == 0
        or normalized_search.shape != (len(labels),)
    ):
        raise ValueError("one aligned MC/search/anchor triple per nonempty row")
    if (
        not torch.isfinite(base_wdl).all()
        or not torch.isfinite(residual_logits).all()
        or not torch.isfinite(normalized_search).all()
        or torch.any(base_wdl < 0)
        or torch.any(normalized_search.abs() > 1)
        or not torch.allclose(base_wdl.sum(1), torch.ones_like(base_wdl[:, 0]), atol=2e-6, rtol=0)
        or labels.dtype != torch.long
        or torch.any((labels < 0) | (labels > 2))
    ):
        raise ValueError("invalid finite WDL/search target/own label")
    if not torch.isfinite(torch.as_tensor(shrink_regularization)).all():
        raise ValueError("nonfinite original SHRINK regularizer")
    base = base_wdl.detach()
    target = normalized_search.detach()
    base_log = base.clamp_min(1e-30).log()
    logits = base_log + residual_logits
    learned_log = F.log_softmax(logits, dim=1)
    probs = learned_log.exp()
    expected_score = probs[:, 0] - probs[:, 2]  # WDL order0win1draw2loss, root mover.
    ce = F.nll_loss(learned_log, labels)
    mse = F.mse_loss(expected_score, target)
    anchor_kl = (base * (base_log - learned_log)).sum(1).mean()
    loss = 0.5 * ce + 0.5 * mse + anchor_kl + shrink_regularization
    return {
        "loss": loss,
        "own_terminal_ce": ce,
        "own_search_mse": mse,
        "fixed_e8_kl": anchor_kl,
        "expected_score": expected_score,
        "logits": logits,
    }


def extract_verified(
    path, expected_sha, config, journal, feature_path, feature_sha, guard=lambda: None
):
    from pathlib import Path

    if (
        journal.sha(path) != expected_sha
        or journal.sha(Path(journal.__file__)) != config["producer_sha256"]
    ):
        raise ValueError("immutable source journal/decoder SHA binding differs")
    feature_module = journal.load_module(feature_path, feature_sha)
    state = journal.read(path)
    result = extract(state, config, journal, feature_module.invariants, guard)
    if journal.sha(path) != expected_sha:
        raise ValueError("source journal changed during aligned extraction")
    result["receipt"].update(
        {
            "source_journal_file_sha256": expected_sha,
            "exact_feature_helper_sha256": feature_sha,
            "source_journal_decoder_sha256": config["producer_sha256"],
        }
    )
    return result


def bind_common_data(extracted, common):
    """Reuse staged trainer's exact12-field protected/dedup/split API unchanged."""
    import hashlib
    import json

    if len(common) != 12:
        raise ValueError("exact prospective common-data API required")
    x, y, anchor, groups, trajectories, train_g, val_g, train_i, val_i, _, base_sha, cfg = common
    for original, name in ((x, "x"), (y, "y"), (anchor, "anchors")):
        if not isinstance(original, torch.Tensor) or original.device.type != "cpu":
            raise ValueError("exact common CPUtensor data required")
        array = original.detach().numpy()
        target = extracted[name]
        if (
            array.dtype != target.dtype
            or array.shape != target.shape
            or array.tobytes() != target.tobytes()
        ):
            raise ValueError("common MC/feature/anchor rows differ before target attachment")
    if (
        len(groups) != len(trajectories)
        or len(set(trajectories)) != len(trajectories)
        or sorted((*train_g, *val_g)) != list(range(len(groups)))
        or tuple(i for gi in train_g for i in groups[gi]) != tuple(train_i)
        or tuple(i for gi in val_g for i in groups[gi]) != tuple(val_i)
        or set(train_i) & set(val_i)
        or any(not 0 <= i < len(y) for group in groups for i in group)
    ):
        raise ValueError("unchanged common trajectory groups/split/row indices required")
    if (
        extracted["receipt"]["config_sha256"]
        != hashlib.sha256(
            json.dumps(cfg, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    ):
        raise ValueError("common original actor config differs")
    digest = hashlib.sha256(bytes.fromhex(base_sha))
    digest.update(extracted["raw_search"].tobytes())
    digest.update(extracted["normalized_search"].tobytes())
    digest.update(
        json.dumps(
            extracted["rows"], sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    )
    return {
        "common_data_sha256": base_sha,
        "search_extended_data_sha256": digest.hexdigest(),
        "common_groups": groups,
        "common_train_indices": train_i,
        "common_validation_indices": val_i,
        "normalized_search": extracted["normalized_search"],
        "raw_search": extracted["raw_search"],
        "no_sampler_or_split_change": True,
    }
