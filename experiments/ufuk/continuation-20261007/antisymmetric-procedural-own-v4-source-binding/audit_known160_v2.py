"""Independent full rule-history replay; no model/search/engine queries."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import chess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def played_depth_valid(depth):
    return type(depth) is int and 1 <= depth <= 8


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "arena", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    p = json.loads(args.protocol.read_text())
    path = args.arena / "cohort-result.json"
    cohort = json.loads(path.read_text())
    assert cohort["status"] == "completed-games-not-strength"
    assert cohort["contract"]["inputs"]["protocol.json"] == sha(args.protocol)
    expected = {(s, a, o) for s in p["match_seeds"] for a, o in p["tasks"]}
    assert len(cohort["rows"]) == len(expected) == 10
    assert {(r["seed"], r["arm"], r["opponent"]) for r in cohort["rows"]} == expected
    book_path = Path(p["book_path"])
    assert sha(book_path) == p["book_sha256"]
    book = json.loads(book_path.read_text())["splits"]["arena"]
    protected = {path: sha(path), book_path: sha(book_path)}
    rows = []
    scores = {}
    for owner in cohort["rows"]:
        seed, arm, opp = owner["seed"], owner["arm"], owner["opponent"]
        assert owner["status"] == "completed-games-awaiting-independent-audit"
        assert owner["finished_epoch"] <= owner["original_deadline_epoch"]
        path = args.arena / f"{seed}-{arm}-vs-{opp}.json"
        protected[path] = sha(path)
        assert protected[path] == owner["result_sha256"]
        x = json.loads(path.read_text())
        assert x["schema"] == "cpu-all-root-quiescent-alpha-beta-arena-v1"
        assert x["source_commit"] == p["source_commit"] and x["protocol_sha256"] == sha(
            args.protocol
        )
        assert x["seed"] == seed and not x["GPU_used"] and not x["promotion_ready"]
        assert x["search_nodes_per_move"] == 512 and x["quiescence_plies"] == 2
        assert x["max_plies"] == 400 and x["opening_source_sha256"] == p["book_sha256"]
        assert (
            x["finished_epoch"] <= x["original_deadline_epoch"] == owner["original_deadline_epoch"]
        )
        for header, helper in [
            ("helper_sha256", "tournament.py"),
            ("search_helper_sha256", "search.py"),
            ("value_helper_sha256", "value.py"),
        ]:
            assert x[header] == cohort["contract"]["inputs"][helper]
        for role, header in [(arm, "candidate_sha256")] + (
            [] if opp == "SF512" else [(opp, "opponent_sha256")]
        ):
            info = p["models"][str(seed)][role]
            weight = Path(info["path"])
            protected[weight] = sha(weight)
            assert protected[weight] == info["sha256"] == x[header]
        if opp == "SF512":
            assert (
                x["stockfish_sha256"],
                x["stockfish_nodes"],
                x["stockfish_threads"],
                x["stockfish_hash_mib"],
            ) == (p["stockfish_sha256"], 512, 1, 16)
        assert len(x["games"]) == 16
        pair_scores = {}
        plies = sf_nodes = nn_nodes = nn_evals = caps = 0
        sf_overruns = sf_max = 0
        times = []
        for g in x["games"]:
            key = (g["opening_pair"], g["candidate_color"])
            assert key not in pair_scores
            assert key[1] in ("white", "black") and 0 <= key[0] < 8
            color = key[1] == "white"
            opening = book[key[0]]["opening"]["moves"]
            assert g["opening"] == opening and g["moves"][: len(opening)] == opening
            neural = {r["ply"]: r for r in g["search_by_move"]}
            sf = {r["ply"]: r for r in g["stockfish_nodes_by_move"]}
            assert len(neural) == len(g["search_by_move"]) and len(sf) == len(
                g["stockfish_nodes_by_move"]
            )
            board = chess.Board()
            expected_nn = []
            expected_sf = []
            for i, uci in enumerate(g["moves"]):
                move = chess.Move.from_uci(uci)
                assert board.is_legal(move)
                if i >= len(opening):
                    assert board.outcome(claim_draw=True) is None
                    ply = i + 1
                    if board.turn == color or opp != "SF512":
                        expected_nn.append(ply)
                        r = neural[ply]
                        assert r["candidate"] == (board.turn == color) and r["selected_move"] == uci
                        assert (
                            r["root_actions"]
                            == r["legal_root_actions"]
                            == board.legal_moves.count()
                        )
                        assert r["root_actions"] + 1 <= r["nodes"] <= 512
                        assert (
                            0 <= r["evaluations"] <= r["nodes"]
                            and played_depth_valid(r["completed_depth"])
                        )
                        assert math.isfinite(r["value"]) and -2 <= r["value"] <= 2
                        assert math.isfinite(r["wall_seconds"]) and r["wall_seconds"] >= 0
                        nn_nodes += r["nodes"]
                        nn_evals += r["evaluations"]
                        if r["candidate"]:
                            times.append(r["wall_seconds"])
                    else:
                        expected_sf.append(ply)
                        r = sf[ply]
                        assert type(r["nodes"]) is int and r["nodes"] >= 0
                        sf_overruns += int(r["nodes"] > 512)
                        sf_max = max(sf_max, r["nodes"])
                        assert math.isfinite(r["wall_seconds"]) and r["wall_seconds"] >= 0
                        sf_nodes += r["nodes"]
                board.push(move)
            assert sorted(neural) == expected_nn and sorted(sf) == expected_sf
            assert len(g["move_wall_seconds"]) == len(g["moves"]) - len(opening)
            assert all(math.isfinite(t) and t >= 0 for t in g["move_wall_seconds"])
            assert board.ply() == g["plies"]
            outcome = board.outcome(claim_draw=True)
            if outcome is None:
                assert g["termination"] == "max_plies" and g["plies"] == 400
                caps += 1
                score = 0.5
            else:
                assert g["termination"] == outcome.termination.name.lower()
                score = 0.5 if outcome.winner is None else float(outcome.winner == color)
            assert score == g["score"]
            pair_scores[key] = score
            plies += len(g["moves"]) - len(opening)
        assert set(pair_scores) == {(i, c) for i in range(8) for c in ("white", "black")}
        wins = sum(s == 1 for s in pair_scores.values())
        draws = sum(s == 0.5 for s in pair_scores.values())
        losses = 16 - wins - draws
        mean = (wins + draws / 2) / 16
        assert (
            x["summary"]["wins"],
            x["summary"]["draws"],
            x["summary"]["losses"],
            x["summary"]["score"],
            x["summary"]["capped_games"],
        ) == (wins, draws, losses, mean, caps)
        scores[seed, arm, opp] = pair_scores
        times.sort()
        rows.append(
            {
                "seed": seed,
                "arm": arm,
                "opponent": opp,
                "sha256": protected[path],
                "score": mean,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "unknown_caps": caps,
                "legal_continuation_plies": plies,
                "actual_SF_nodes": sf_nodes,
                "actual_SF_moves_over_nominal512": sf_overruns,
                "actual_SF_max_move_nodes": sf_max,
                "search_nodes": nn_nodes,
                "static_evaluations": nn_evals,
                "candidate_median_move_seconds": times[len(times) // 2],
                "bootstrap_pair_95": x["summary"]["bootstrap_pair_95"],
                "hoeffding_pair_95": x["summary"]["hoeffding_pair_95"],
            }
        )
    contrasts = []
    for seed in p["match_seeds"]:

        def row(a, o, seed=seed):
            return next(r for r in rows if (r["seed"], r["arm"], r["opponent"]) == (seed, a, o))

        final = row("learned", "SF512")
        base = row("e8", "SF512")
        zero = row("parent", "SF512")
        direct = row("learned", "e8")
        learning = row("learned", "parent")
        delta = {
            str(k): v - scores[seed, "e8", "SF512"][k]
            for k, v in scores[seed, "learned", "SF512"].items()
        }
        gain = sum(delta.values()) / 16
        assert gain == final["score"] - base["score"]
        gates = {
            "direct_same_search_e8_gt_060": direct["score"] > 0.6,
            "pairedSF_gain_same_search_e8_gt_010": gain > 0.1,
            "finalSF_ge_025": final["score"] >= 0.25,
            "trained_vs_exact_current_parent_gt_060": learning["score"] > 0.6,
            "pairedSF_gain_exact_current_parent_gt_0": final["score"] > zero["score"],
            "caps_le_005": all(
                r["unknown_caps"] / 16 <= 0.05 for r in (final, base, zero, direct, learning)
            ),
        }
        contrasts.append(
            {
                "seed": seed,
                "arm": "learned",
                "screen_gates": gates,
                "passed_screen": all(gates.values()),
                "paired_SF_delta_by_game": delta,
                "paired_SF_gain": gain,
            }
        )
    assert len(contrasts) == 2 and {c["seed"] for c in contrasts} == set(p["match_seeds"])
    advance = all(c["passed_screen"] for c in contrasts)
    assert all(sha(path) == digest for path, digest in protected.items())
    result = {
        "status": "PASS-fullhistory-integrity",
        "screen_result": "PASS-development-only" if advance else "FAIL",
        "rows": rows,
        "contrasts": contrasts,
        "games": 160,
        "legal_continuation_plies": sum(r["legal_continuation_plies"] for r in rows),
        "actual_SF_nodes": sum(r["actual_SF_nodes"] for r in rows),
        "search_nodes": sum(r["search_nodes"] for r in rows),
        "protocol_sha256": sha(args.protocol),
        "helper_sha256": sha(Path(__file__)),
        "protected_files_unchanged": True,
        "strength_success_claimed": False,
        "unchanged_inferential_strength_gate": (
            "NOT-QUALIFIED: known8 point screen; approved ONE confirmation requires "
            "all8 LCBs on new48 paired source blocks perseed"
        ),
        "GPU_used": False,
        "scope": p["scope"],
        "formal_requirement": {
            "status": "not-registered-known-book-screen-only",
            "prospective_memo_sha256": (
                "b6e58037e2c579a2254f4b1510281869bef5a535c277b81fd9788228c513a325"
            ),
        },
        "uncertainty": (
            "8 shared known opening families; descriptive paired intervals, "
            "not virgin confirmation or independent seeds-as-families. "
            "Degenerate empirical bootstrap is not population zero."
        ),
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"status": result["status"], "screen_result": result["screen_result"], "games": 160}
        )
    )


if __name__ == "__main__":
    main()
