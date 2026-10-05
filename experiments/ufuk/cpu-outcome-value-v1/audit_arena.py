"""Independent rules replay of fixed development games; no engine or model queries."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import chess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arena", type=Path, required=True)
    parser.add_argument("--fits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    book_path = Path(protocol["book_path"])
    assert sha(book_path) == protocol["book_sha256"]
    book = json.loads(book_path.read_text())["splits"]["arena"]
    cohort_path = args.arena / "cohort-result.json"
    cohort = json.loads(cohort_path.read_text())
    assert cohort["status"] == "completed-games-not-strength"
    assert cohort["contract"]["protocol_sha256"] == sha(args.protocol)
    expected = {
        (seed, arm, opponent) for seed in protocol["seeds"] for arm, opponent in protocol["tasks"]
    }
    assert len(expected) == len(cohort["rows"]) == 10
    assert {(r["seed"], r["arm"], r["opponent"]) for r in cohort["rows"]} == expected
    rows, protected = [], {cohort_path: sha(cohort_path), book_path: sha(book_path)}
    pair_scores = {}

    def model_path(seed, role):
        step = 0 if role == "rebased" else 2048
        return args.fits / f"{seed}-linear/checkpoints" / f"step-{step:08d}" / "model.safetensors"

    for owner in cohort["rows"]:
        seed, arm, opponent = owner["seed"], owner["arm"], owner["opponent"]
        path = args.arena / f"{seed}-{arm}-vs-{opponent}.json"
        protected[path] = sha(path)
        assert protected[path] == owner["result_sha256"]
        assert owner["finished_epoch"] <= owner["original_deadline_epoch"]
        x = json.loads(path.read_text())
        assert x["seed"] == seed and x["source_commit"] == protocol["source_commit"]
        assert (
            x["neural_simulations_per_move"],
            x["neural_threads"],
            x["candidate_root_actions"],
            x["max_plies"],
        ) == (16, 1, 4, 400)
        assert not x["promotion_ready"]
        assert x["opening_source_sha256"] == protocol["book_sha256"]
        if arm == "e8":
            candidate_hash = protocol["initial_e8_sha256"]
        else:
            weight = model_path(seed, arm)
            protected[weight] = candidate_hash = sha(weight)
            checkpoint = json.loads((weight.parent / "checkpoint.json").read_text())
            assert checkpoint["schema"] == "cpu-own-outcome-linear-training-native-v2"
            assert checkpoint["artifacts"]["model.safetensors"] == candidate_hash
            assert checkpoint["accepted"] == (0 if arm == "rebased" else 2048)
        assert x["candidate_sha256"] == owner["candidate_sha256"] == candidate_hash
        if opponent != "SF512":
            expected_opponent = (
                protocol["initial_e8_sha256"]
                if opponent == "e8"
                else sha(model_path(seed, opponent))
            )
            assert x["opponent_sha256"] == expected_opponent
            assert x["opponent_root_actions"] == 4
        else:
            assert x["stockfish_sha256"] == protocol["stockfish_sha256"]
            assert (x["stockfish_nodes"], x["stockfish_threads"], x["stockfish_hash_mib"]) == (
                512,
                1,
                16,
            )
        assert len(x["games"]) == 16
        pairs, scores = set(), {}
        plies, nodes, wins, draws, losses, caps = 0, 0, 0, 0, 0, 0
        for g in x["games"]:
            key = (g["opening_pair"], g["candidate_color"])
            assert key not in pairs and key[1] in ("white", "black")
            pairs.add(key)
            opening = book[key[0]]["opening"]["moves"]
            assert g["opening"] == opening and g["moves"][: len(opening)] == opening
            board, sf_plies = chess.Board(), []
            for i, uci in enumerate(g["moves"]):
                if i >= len(opening):
                    assert board.outcome(claim_draw=True) is None
                    if opponent == "SF512" and board.turn != (key[1] == "white"):
                        sf_plies.append(i + 1)
                move = chess.Move.from_uci(uci)
                assert move in board.legal_moves
                board.push(move)
            assert g["plies"] == len(g["moves"])
            outcome = board.outcome(claim_draw=True)
            if outcome is None:
                assert g["termination"] == "max_plies" and g["plies"] == 400
                score = 0.5
                caps += 1
            else:
                assert g["termination"] == outcome.termination.name.lower()
                score = (
                    0.5 if outcome.winner is None else float(outcome.winner == (key[1] == "white"))
                )
            assert g["score"] == score
            wins += score == 1
            draws += score == 0.5
            losses += score == 0
            scores[key] = score
            assert len(g["move_wall_seconds"]) == len(g["moves"]) - len(opening)
            assert all(math.isfinite(t) and t >= 0 for t in g["move_wall_seconds"])
            if opponent == "SF512":
                receipt = g["stockfish_nodes_by_move"]
                assert [r["ply"] for r in receipt] == sf_plies
                assert all(type(r["nodes"]) is int and r["nodes"] >= 0 for r in receipt)
                nodes += sum(r["nodes"] for r in receipt)
            plies += len(g["moves"]) - len(opening)
        assert pairs == {(i, c) for i in range(8) for c in ("white", "black")}
        score = sum(scores.values()) / 16
        summary = x["summary"]
        assert (score, wins, draws, losses, caps) == (
            summary["score"],
            summary["wins"],
            summary["draws"],
            summary["losses"],
            summary["capped_games"],
        )
        assert nodes == x["stockfish_actual_nodes"]
        pair_scores[seed, arm, opponent] = scores
        rows.append(
            {
                "seed": seed,
                "arm": arm,
                "opponent": opponent,
                "sha256": protected[path],
                "candidate_sha256": candidate_hash,
                "score": score,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "unknown_caps": caps,
                "legal_continuation_plies": plies,
                "actual_SF_nodes": nodes,
                "wall_seconds": x["wall_seconds"],
                "bootstrap_pair_95": summary["bootstrap_pair_95"],
                "hoeffding_pair_95": summary["hoeffding_pair_95"],
            }
        )
    contrasts = []
    for arm in protocol["arms"]:
        for seed in protocol["seeds"]:

            def pick(a, o, current_seed=seed):
                return next(
                    r for r in rows if (r["seed"], r["arm"], r["opponent"]) == (current_seed, a, o)
                )

            direct, final, base = pick(arm, "e8"), pick(arm, "SF512"), pick("e8", "SF512")
            zero_direct, zero_sf = pick(arm, "rebased"), pick("rebased", "SF512")
            delta = {
                str(k): pair_scores[seed, arm, "SF512"][k] - v
                for k, v in pair_scores[seed, "e8", "SF512"].items()
            }
            gain = sum(delta.values()) / 16
            assert gain == final["score"] - base["score"]
            gates = {
                "trained_vs_untrained_rebased_gt_060": zero_direct["score"] > 0.60,
                "trainedSF_gain_over_untrained_rebased_gt_0": final["score"] > zero_sf["score"],
                "direct_gt_060": direct["score"] > 0.60,
                "SF_gain_gt_010": gain > 0.10,
                "finalSF_ge_025": final["score"] >= 0.25,
                "caps_le_005": all(
                    r["unknown_caps"] / 16 <= 0.05
                    for r in (direct, final, base, zero_direct, zero_sf)
                ),
            }
            contrasts.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "paired_SF_delta_by_game": delta,
                    "paired_SF_mean_gain": gain,
                    "screen_gates": gates,
                    "passed_screen": all(gates.values()),
                }
            )
    advance = [
        a for a in protocol["arms"] if all(c["passed_screen"] for c in contrasts if c["arm"] == a)
    ]
    assert all(sha(p) == h for p, h in protected.items())
    result = {
        "schema": "cpu-own-outcome-linear-independent-fullhistory-arena-audit-v1",
        "status": "PASS-fullhistory-integrity",
        "screen_result": "PASS-development-only" if advance else "FAIL",
        "advance_arms": advance,
        "rows": rows,
        "contrasts": contrasts,
        "games": sum(r["wins"] + r["draws"] + r["losses"] for r in rows),
        "legal_continuation_plies": sum(r["legal_continuation_plies"] for r in rows),
        "actual_SF_nodes": sum(r["actual_SF_nodes"] for r in rows),
        "source_commit": protocol["source_commit"],
        "protocol_sha256": sha(args.protocol),
        "audit_helper_sha256": sha(Path(__file__)),
        "protected_files_unchanged": True,
        "scope": protocol["scope"],
        "uncertainty": protocol["uncertainty"],
        "strength_success_claimed": False,
        "GPU_used": False,
    }
    assert result["games"] == protocol["total_planned_games"]
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "screen_result",
                    "games",
                    "legal_continuation_plies",
                    "actual_SF_nodes",
                    "advance_arms",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
