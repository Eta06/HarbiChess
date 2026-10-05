"""Independent full legal-history replay and frozen paired-screen calculation."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import chess
from contracts import (
    load_fit_manifest,
    model_record,
    verify_fit_inputs,
    verify_fixed_arena_protocol,
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for n in ("protocol", "arena", "output"):
        ap.add_argument("--" + n, type=Path, required=True)
    args = ap.parse_args()
    p = json.loads(args.protocol.read_text())
    verify_fixed_arena_protocol(p)
    if p.get("status") != "registered":
        raise ValueError("registered protocol required")
    cohort_path = args.arena / "cohort-result.json"
    cohort = json.loads(cohort_path.read_text())
    fit_path, fit_manifest = load_fit_manifest(p)
    verify_fit_inputs(p, fit_manifest)
    expected = {(s, a, o) for s in p["match_seeds"] for a, o in p["tasks"]}
    if cohort.get("status") != "completed-games-not-strength" or len(expected) != 14:
        raise ValueError("complete registered 14-tournament arena required")
    if {(r["seed"], r["arm"], r["opponent"]) for r in cohort["rows"]} != expected:
        raise ValueError("arena tournament inventory differs")
    bookpath = Path(p["book_path"])
    if sha(bookpath) != p["book_sha256"]:
        raise ValueError("book bytes changed")
    book = json.loads(bookpath.read_text())["splits"]["arena"]
    protected = {
        args.protocol: sha(args.protocol),
        cohort_path: sha(cohort_path),
        bookpath: sha(bookpath),
        fit_path: sha(fit_path),
    }
    if cohort["contract"]["inputs"].get("protocol.json") != sha(args.protocol):
        raise ValueError("owner did not bind registered protocol")
    if cohort["contract"].get("fit_manifest_sha256") != sha(fit_path):
        raise ValueError("owner did not bind fit provenance manifest")
    rows, scores = [], {}
    for owner in cohort["rows"]:
        seed, arm, opp = owner["seed"], owner["arm"], owner["opponent"]
        if owner["status"] != "completed-games-awaiting-independent-audit":
            raise ValueError("incomplete tournament row")
        if owner["finished_epoch"] > owner["original_deadline_epoch"]:
            raise ValueError("tournament exceeded its original clock")
        path = args.arena / f"{seed}-{arm}-vs-{opp}.json"
        if sha(path) != owner["result_sha256"]:
            raise ValueError("tournament result hash differs")
        protected[path] = sha(path)
        result = json.loads(path.read_text())
        if (
            result.get("schema") != "cpu-all-root-quiescent-alpha-beta-arena-v1"
            or result.get("source_commit") != p["source_commit"]
            or result.get("protocol_sha256") != sha(args.protocol)
            or result.get("seed") != seed
            or result.get("GPU_used") is not False
            or result.get("promotion_ready") is not False
        ):
            raise ValueError("result provenance differs")
        if (
            result["finished_epoch"]
            > result["original_deadline_epoch"]
            != owner["original_deadline_epoch"]
        ):
            raise ValueError("result finished outside registered clock")
        if (
            result["search_nodes_per_move"],
            result["quiescence_plies"],
            result["max_plies"],
            result["opening_source_sha256"],
        ) != (512, 2, 400, p["book_sha256"]):
            raise ValueError("neural search/book configuration differs")
        if result["helper_sha256"] != cohort["contract"]["inputs"]["tournament.py"]:
            raise ValueError("tournament helper hash differs")
        candidate_info = model_record(p, fit_manifest, seed, arm)
        if result["candidate_sha256"] != candidate_info["sha256"]:
            raise ValueError("candidate SHA differs from fit manifest")
        protected[Path(candidate_info["path"])] = candidate_info["sha256"]
        if opp != "SF512":
            opponent_info = model_record(p, fit_manifest, seed, opp)
            if result["opponent_sha256"] != opponent_info["sha256"]:
                raise ValueError("opponent SHA differs from fit manifest")
            protected[Path(opponent_info["path"])] = opponent_info["sha256"]
        else:
            if (
                result["stockfish_sha256"],
                result["stockfish_nodes"],
                result["stockfish_threads"],
                result["stockfish_hash_mib"],
            ) != (p["stockfish_sha256"], 512, 1, 16):
                raise ValueError("Stockfish settings differ")
        if len(result["games"]) != p["games_per_tournament"]:
            raise ValueError("expected 16 games")
        game_scores, caps = {}, 0
        nn_nodes = sf_nodes = plies_total = 0
        sf_over_request = sf_max_actual = sf_move_count = 0
        for game in result["games"]:
            pair, colorname = game["opening_pair"], game["candidate_color"]
            key = (pair, colorname)
            if (
                key in game_scores
                or colorname not in ("white", "black")
                or not 0 <= pair < 8
            ):
                raise ValueError("duplicate/out-of-range paired game")
            opening = book[pair]["opening"]["moves"]
            if game["opening"] != opening or game["moves"][: len(opening)] != opening:
                raise ValueError("opening prefix differs")
            color = colorname == "white"
            neural = {r["ply"]: r for r in game["search_by_move"]}
            sfmoves = {r["ply"]: r for r in game["stockfish_nodes_by_move"]}
            if len(neural) != len(game["search_by_move"]) or len(sfmoves) != len(
                game["stockfish_nodes_by_move"]
            ):
                raise ValueError("duplicate search trace ply")
            board = chess.Board()
            expected_nn, expected_sf = [], []
            for i, uci in enumerate(game["moves"]):
                move = chess.Move.from_uci(uci)
                if not board.is_legal(move):
                    raise ValueError("illegal move in recorded full history")
                if i >= len(opening):
                    if board.outcome(claim_draw=True) is not None:
                        raise ValueError("moves continued after terminal")
                    ply = i + 1
                    if board.turn == color or opp != "SF512":
                        expected_nn.append(ply)
                        r = neural[ply]
                        if (
                            r["candidate"] != (board.turn == color)
                            or r["selected_move"] != uci
                        ):
                            raise ValueError("neural trace does not match played move")
                        if (
                            r["root_actions"] != r["legal_root_actions"]
                            or r["root_actions"] != board.legal_moves.count()
                            or not r["root_actions"] + 1 <= r["nodes"] <= 512
                        ):
                            raise ValueError(
                                "neural legal-root/node accounting differs"
                            )
                        if not (
                            0 <= r["evaluations"] <= r["nodes"]
                            and 1 <= r["completed_depth"] <= 8
                            and math.isfinite(r["value"])
                        ):
                            raise ValueError("neural trace range invalid")
                        nn_nodes += r["nodes"]
                    else:
                        expected_sf.append(ply)
                        r = sfmoves[ply]
                        if type(r["nodes"]) is not int or r["nodes"] < 0:
                            raise ValueError("SF node trace invalid")
                        sf_nodes += r["nodes"]
                        sf_move_count += 1
                        sf_max_actual = max(sf_max_actual, r["nodes"])
                        sf_over_request += int(r["nodes"] > p["stockfish_nodes"])
                board.push(move)
            if sorted(neural) != expected_nn or sorted(sfmoves) != expected_sf:
                raise ValueError("move trace inventory incomplete or unexpected")
            if board.ply() != game["plies"]:
                raise ValueError("ply total mismatch")
            outcome = board.outcome(claim_draw=True)
            if outcome is None:
                if game["termination"] != "max_plies" or game["plies"] != 400:
                    raise ValueError("unknown termination must be recorded as cap")
                caps += 1
                score = 0.5
            else:
                if game["termination"] != outcome.termination.name.lower():
                    raise ValueError("terminal reason mismatch")
                score = (
                    0.5 if outcome.winner is None else float(outcome.winner == color)
                )
            if game["score"] != score:
                raise ValueError("game score/mover POV differs")
            game_scores[key] = score
            plies_total += len(game["moves"]) - len(opening)
        if set(game_scores) != {(i, c) for i in range(8) for c in ("white", "black")}:
            raise ValueError("paired opening/color coverage incomplete")
        wins = sum(x == 1 for x in game_scores.values())
        draws = sum(x == 0.5 for x in game_scores.values())
        mean = (wins + draws / 2) / 16
        if (
            result["summary"]["score"] != mean
            or result["summary"]["capped_games"] != caps
        ):
            raise ValueError("summary does not reproduce from full histories")
        scores[seed, arm, opp] = game_scores
        rows.append(
            {
                "seed": seed,
                "arm": arm,
                "opponent": opp,
                "score": mean,
                "wins": wins,
                "draws": draws,
                "losses": 16 - wins - draws,
                "capped_games": caps,
                "continuation_plies": plies_total,
                "nn_nodes": nn_nodes,
                "sf_nodes": sf_nodes,
                "sf_moves": sf_move_count,
                "sf_moves_over_requested_nodes": sf_over_request,
                "sf_max_actual_nodes": sf_max_actual,
                "sf_requested_nodes": p["stockfish_nodes"],
                "result_sha256": sha(path),
            }
        )
    contrasts = []
    for seed in p["match_seeds"]:
        for arm in ("mc", "sc", "full"):
            direct = next(
                r["score"]
                for r in rows
                if (r["seed"], r["arm"], r["opponent"]) == (seed, arm, "e8")
            )
            candidate = next(
                r
                for r in rows
                if (r["seed"], r["arm"], r["opponent"]) == (seed, arm, "SF512")
            )
            baseline = next(
                r
                for r in rows
                if (r["seed"], r["arm"], r["opponent"]) == (seed, "e8", "SF512")
            )
            delta = {
                str(k): scores[seed, arm, "SF512"][k] - scores[seed, "e8", "SF512"][k]
                for k in scores[seed, arm, "SF512"]
            }
            gain = sum(delta.values()) / 16
            caps_rows = [
                r
                for r in rows
                if r["seed"] == seed
                and (
                    (r["arm"] == arm and r["opponent"] in ("e8", "SF512"))
                    or (r["arm"] == "e8" and r["opponent"] == "SF512")
                )
            ]
            caps_ok = all(
                r["capped_games"] / 16 <= p["screen"]["caps_at_most"] for r in caps_rows
            )
            gates = {
                "direct_gt_060": direct
                > p["screen"]["direct_same_search_e8_score_strictly_above"],
                "paired_sf_gain_gt_010": gain
                > p["screen"]["paired_sf_gain_over_e8_strictly_above"],
                "final_sf_ge_025": candidate["score"]
                >= p["screen"]["final_sf_score_at_least"],
                "caps_le_005": caps_ok,
            }
            contrasts.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "direct_e8_score": direct,
                    "candidate_sf_score": candidate["score"],
                    "baseline_sf_score": baseline["score"],
                    "paired_sf_gain": gain,
                    "paired_delta_by_game": delta,
                    "screen_gates": gates,
                    "passed_screen": all(gates.values()),
                }
            )
    for path, digest in protected.items():
        if sha(path) != digest:
            raise ValueError("protected input/result changed during independent audit")
    eligible = [
        role
        for role in ("mc", "sc", "full")
        if all(item["passed_screen"] for item in contrasts if item["arm"] == role)
    ]
    screen_result = "PASS-development-only" if eligible else "FAIL"
    result = {
        "integrity_status": "PASS-fullhistory-integrity",
        "status": screen_result,
        "screen_result": screen_result,
        "rows": rows,
        "contrasts": contrasts,
        "eligible_development_methods": eligible,
        "eligibility_rule": (
            "A method is eligible only when every frozen screen gate passes for "
            "both registered seeds; eligibility is development-only."
        ),
        "games": 224,
        "protocol_sha256": sha(args.protocol),
        "helper_sha256": sha(Path(__file__)),
        "protected_files_unchanged": True,
        "strength_success_claimed": False,
        "GPU_used": False,
        "scope": p["arena_scope"],
    }
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "screen_result": result["screen_result"],
                "games": 224,
            }
        )
    )


if __name__ == "__main__":
    main()
