"""Prospective four-comparison, paired-root strength analysis; no training/query."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SEEDS = (20261205, 20261206)
REPLICATES = 50000
CONFIDENCE = 0.9875
ANALYSIS_SEED = 20261107


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def block_scores(result, book):
    roots = book["splits"]["arena"]
    if len(roots) != 48 or len({r["source_game"] for r in roots}) != 48:
        raise ValueError("48 distinct source-root blocks required")
    if len(result["games"]) != 96:
        raise ValueError("all 96 games required; partial outcomes are incomplete")
    scores = np.full((48, 2), np.nan)
    caps = np.zeros((48, 2), dtype=bool)
    for g in result["games"]:
        pair = g["opening_pair"]
        color = {"white": 0, "black": 1}.get(str(g["candidate_color"]))
        if (
            type(pair) is not int
            or not 0 <= pair < 48
            or color is None
            or np.isfinite(scores[pair, color])
        ):
            raise ValueError("invalid/duplicate source-root colour")
        if g["opening"] != roots[pair]["opening"]["moves"]:
            raise ValueError("game does not match frozen root history")
        value = g["score"]
        cap = g["termination"] == "max_plies"
        if value not in (0, 0.5, 1) or (cap and value != 0.5):
            raise ValueError("bad outcome or unknown cap labelled a result")
        scores[pair, color] = value
        caps[pair, color] = cap
    if not np.isfinite(scores).all():
        raise ValueError("missing colour pair")
    return scores, caps


def intervals(blocks, indices):
    draws = np.mean(blocks[indices], axis=1)
    return {
        "mean": float(np.mean(blocks)),
        "ci_adjusted_98_75": np.quantile(draws, [0.00625, 0.99375], method="linear").tolist(),
        "ci_descriptive_95": np.quantile(draws, [0.025, 0.975], method="linear").tolist(),
    }


def assess_seed(direct, final_sf, initial_sf, book, seed):
    # Identical resamples preserve BOTH colours and cross-arm root matching.
    indices = np.random.default_rng(ANALYSIS_SEED + seed).integers(0, 48, (REPLICATES, 48))
    d, dc = block_scores(direct, book)
    f, fc = block_scores(final_sf, book)
    i, ic = block_scores(initial_sf, book)
    direct_result = intervals(d.mean(1), indices)
    delta_result = intervals((f - i).mean(1), indices)
    worst_direct = np.where(dc, 0, d)
    worst_delta = np.where(fc, 0, f) - np.where(ic, 1, i)
    direct_worst = intervals(worst_direct.mean(1), indices)
    delta_worst = intervals(worst_delta.mean(1), indices)
    caps = {k: float(c.mean()) for k, c in [("direct", dc), ("final_sf", fc), ("initial_sf", ic)]}
    gates = {
        "direct_mean_gt_060": direct_result["mean"] > 0.60,
        "direct_adjusted_lower_gt_050": direct_result["ci_adjusted_98_75"][0] > 0.50,
        "sf_paired_gain_gt_010": delta_result["mean"] > 0.10,
        "final_sf_absolute_at_least_025": float(f.mean()) >= 0.25,
        "sf_gain_adjusted_lower_gt_0": delta_result["ci_adjusted_98_75"][0] > 0,
        "each_arm_caps_at_most_005": all(v <= 0.05 for v in caps.values()),
        "direct_adversarial_caps_adjusted_lower_gt_050": direct_worst["ci_adjusted_98_75"][0]
        > 0.50,
        "sf_adversarial_caps_adjusted_lower_gt_0": delta_worst["ci_adjusted_98_75"][0] > 0,
    }
    return {
        "seed": seed,
        "direct": direct_result,
        "sf_paired_delta": delta_result,
        "final_sf_mean": float(f.mean()),
        "initial_sf_mean": float(i.mean()),
        "caps": caps,
        "direct_adversarial_caps": direct_worst,
        "sf_delta_adversarial_caps": delta_worst,
        "gates": gates,
        "strength_pass": all(gates.values()),
        "block_unit": (
            "48 distinct source-game roots; both colours jointly; not distinct ECO families"
        ),
    }


def validate_arm(result, book_path, candidate_sha, opponent_sha=None):
    if (
        result["opening_source_sha256"] != digest(book_path)
        or result["candidate_sha256"] != candidate_sha
    ):
        raise ValueError("arena input hashes mismatch")
    if opponent_sha is not None and result["opponent_sha256"] != opponent_sha:
        raise ValueError("opponent hash mismatch")
    if result["neural_simulations_per_move"] != 16 or result["neural_threads"] != 1:
        raise ValueError("frozen neural compute mismatch")
    if result.get("candidate_root_actions") != 4 or (
        opponent_sha is not None and result.get("opponent_root_actions") != 4
    ):
        raise ValueError("frozen max4 root-actions mismatch")
    if result.get("candidate_selection") == "raw_policy_argmax" or result["max_plies"] != 400:
        raise ValueError("policy-only or total-ply cap mismatch")
    if opponent_sha is None and (
        result["stockfish_nodes"] != 512
        or result["stockfish_threads"] != 1
        or result["stockfish_hash_mib"] != 16
    ):
        raise ValueError("SF512 reference mismatch")
    if opponent_sha is None:
        if (
            result["stockfish_sha256"]
            != "0f83d24cc46d2c66c60f16001af5444873bc112b7d028594513426894c12da19"
        ):
            raise ValueError("frozen Stockfish binary mismatch")
        if any(
            not g.get("stockfish_nodes_by_move")
            or any(
                type(r.get("nodes")) is not int or r["nodes"] < 0
                for r in g["stockfish_nodes_by_move"]
            )
            for g in result["games"]
        ):
            raise ValueError("actual engine node receipts missing")


def audit_trajectories(result, book, stockfish=False):
    import chess

    rows = book["splits"]["arena"]
    for game in result["games"]:
        prefix = rows[game["opening_pair"]]["opening"]["moves"]
        moves = game["moves"]
        if moves[: len(prefix)] != prefix or len(moves) != game["plies"] or len(moves) > 400:
            raise ValueError("recorded full history/cap mismatch")
        board = chess.Board()
        engine_plies = []
        candidate_white = game["candidate_color"] == "white"
        for index, uci in enumerate(moves):
            if index >= len(prefix):
                if board.outcome(claim_draw=True) is not None:
                    raise ValueError("move after declared terminal")
                if board.turn != candidate_white:
                    engine_plies.append(index + 1)
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                raise ValueError("illegal recorded move")
            board.push(move)
        outcome = board.outcome(claim_draw=True)
        if outcome is None:
            if len(moves) != 400 or game["termination"] != "max_plies" or game["score"] != 0.5:
                raise ValueError("unknown cap mislabeled")
        else:
            expected = 0.5 if outcome.winner is None else float(outcome.winner == candidate_white)
            if game["score"] != expected or game["termination"] != outcome.termination.name.lower():
                raise ValueError("terminal outcome/perspective mismatch")
        if stockfish and [r["ply"] for r in game["stockfish_nodes_by_move"]] != engine_plies:
            raise ValueError("engine node trace does not cover exact opponent moves")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    m = json.loads(a.manifest.read_text())
    if [s["seed"] for s in m["seeds"]] != list(SEEDS):
        raise ValueError("fixed two-seed order required")
    reports = []
    source_ids = []
    for entry in m["seeds"]:
        book = json.loads(Path(entry["book"]).read_text())
        source_ids.append({r["source_game"] for r in book["splits"]["arena"]})
        arms = {
            name: json.loads(Path(entry[name]).read_text())
            for name in ("direct", "final_sf", "initial_sf")
        }
        for name in arms:
            if (
                arms[name]["source_commit"] != m["source_commit"]
                or arms[name]["seed"] != entry["seed"]
            ):
                raise ValueError("pinned source/seed mismatch")
            validate_arm(
                arms[name],
                entry["book"],
                entry["initial_sha256"] if name == "initial_sf" else entry["final_sha256"],
                entry["initial_sha256"] if name == "direct" else None,
            )
        for name in arms:
            audit_trajectories(arms[name], book, stockfish=name != "direct")
        reports.append(
            assess_seed(
                arms["direct"],
                arms["final_sf"],
                arms["initial_sf"],
                book,
                entry["seed"],
            )
        )
    if source_ids[0] & source_ids[1]:
        raise ValueError("replication books overlap source families")
    result = {
        "schema": 1,
        "primary_comparisons": 4,
        "familywise_alpha": 0.05,
        "two_sided_ci": CONFIDENCE,
        "replicates": REPLICATES,
        "quantiles": "numpy linear",
        "analysis_seed": ANALYSIS_SEED,
        "seed_results": reports,
        "replicated_strength_pass": all(r["strength_pass"] for r in reports),
        "scope": (
            "Conditional independent training seeds and disjoint source-root s"
            "uites; no general Elo or Stockfish-level claim."
        ),
        "analysis_sha256": digest(__file__),
        "manifest_sha256": digest(a.manifest),
    }
    with a.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
