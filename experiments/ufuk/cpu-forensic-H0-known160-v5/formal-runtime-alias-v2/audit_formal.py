"""ROOT-only exact960 legal histories and unchanged bootstrap plus additive e-LCBs."""

import argparse
import json
import math
import time
from pathlib import Path

import chess
from bindings import SEEDS, TASKS, read, sha
from bounded_betting import METRICS, eight_primary_bounds
from confirmation_stats import evaluate_seed, paired_arrays
from runtime_support import guard_factory
from select_books import publish


def game_packet(game, opening, opponent):
    if game["opening"] != opening or game["moves"][:len(opening)] != opening:
        raise ValueError("exact full-history opening")
    neural = {x["ply"]: x for x in game["search_by_move"]}
    sf = {x["ply"]: x for x in game["stockfish_nodes_by_move"]}
    if (len(neural) != len(game["search_by_move"])
            or len(sf) != len(game["stockfish_nodes_by_move"])):
        raise ValueError("duplicate per-move receipts")
    board = chess.Board()
    expected_nn, expected_sf = [], []
    color = game["candidate_color"] == "white"
    for index, uci in enumerate(game["moves"]):
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError("illegal complete trajectory")
        if index >= len(opening):
            if board.outcome(claim_draw=True):
                raise ValueError("move after terminal state")
            ply = index + 1
            if board.turn == color or opponent != "SF512":
                expected_nn.append(ply)
                r = neural[ply]
                if (r["candidate"] != (board.turn == color) or r["selected_move"] != uci
                        or r["root_actions"] != r["legal_root_actions"]
                        or r["root_actions"] != board.legal_moves.count()
                        or type(r["nodes"]) is not int
                        or not r["root_actions"] + 1 <= r["nodes"] <= 512
                        or type(r["evaluations"]) is not int
                        or not 0 <= r["evaluations"] <= r["nodes"]
                        or type(r["completed_depth"]) is not int
                        or not 1 <= r["completed_depth"] <= 8
                        or not math.isfinite(r["value"]) or not -2 <= r["value"] <= 2
                        or not math.isfinite(r["wall_seconds"]) or r["wall_seconds"] < 0):
                    raise ValueError("exact NN mover/counter/value/depth packet")
            else:
                expected_sf.append(ply)
                r = sf[ply]
                if (type(r["nodes"]) is not int or r["nodes"] < 0
                        or not math.isfinite(r["wall_seconds"]) or r["wall_seconds"] < 0):
                    raise ValueError("actual SF node receipt; nominal overrun is reported")
        board.push(move)
    if (sorted(neural) != expected_nn or sorted(sf) != expected_sf
            or board.ply() != game["plies"] or board.ply() > 400
            or len(game["move_wall_seconds"]) != len(game["moves"]) - len(opening)
            or any(not math.isfinite(x) or x < 0 for x in game["move_wall_seconds"])):
        raise ValueError("ALL played move packets/history/400total cap")
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        if board.ply() != 400 or game["termination"] != "max_plies":
            raise ValueError("exact UNKNOWN400 cap")
        score = 0.5
    else:
        if game["termination"] != outcome.termination.name.lower():
            raise ValueError("exact terminal reason")
        score = 0.5 if outcome.winner is None else float(outcome.winner == color)
    if game["score"] != score:
        raise ValueError("candidate mover outcome score")
    return dict(plies=len(game["moves"]) - len(opening),
                NN_nodes=sum(x["nodes"] for x in neural.values()),
                SF_nodes=sum(x["nodes"] for x in sf.values()),
                SF_overrun_moves=sum(x["nodes"] > 512 for x in sf.values()),
                SF_max_move_nodes=max((x["nodes"] for x in sf.values()), default=0))


def validate_progress(path, games):
    with Path(path).open("rb") as f:
        for game in games:
            start = json.loads(next(f))
            if start != dict(type="game_start", pair=game["opening_pair"],
                             color=game["candidate_color"]):
                raise ValueError("exact original progress game-start order")
            for index, move in enumerate(game["moves"][len(game["opening"]):],
                                         start=len(game["opening"]) + 1):
                if json.loads(next(f)) != dict(type="move", ply=index, uci=move):
                    raise ValueError("all chronological progress moves must match raw games")
            if json.loads(next(f)) != dict(type="game_end", game=game):
                raise ValueError("exact closed progress game packet")
        if next(f, None) is not None:
            raise ValueError("unexpected progress tail after allclosed96games")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("registration", "arena", "clock", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    a = parser.parse_args()
    reg = json.loads(a.registration.read_bytes())
    clock = json.loads(a.clock.read_bytes())
    if (clock["registration_sha256"] != sha(a.registration)
            or not clock["first"] <= time.time() < clock["deadline"] <= reg["audit_deadline"]
            or clock["deadline"] > clock["first"] + 900):
        raise ValueError("ROOT original bounded independent audit900")
    known = read(reg["known_protocol"])
    guard = guard_factory(known["core_repo"], reg["cpu_core"], clock["deadline"])
    if any(sha(Path(__file__).with_name(n)) != d
           for n, d in reg["helper_sha256"].items()):
        raise ValueError("entire frozen audit helper closure")
    cohort = json.loads((a.arena / "cohort-result.json").read_bytes())
    expected = {(s, x, y) for s in SEEDS for x, y in TASKS}
    if (cohort["status"] != "completed-960-awaiting-independent-audit"
            or cohort["registration_sha256"] != sha(a.registration)
            or {(r["seed"], r["arm"], r["opponent"]) for r in cohort["rows"]} != expected
            or len(cohort["rows"]) != 10):
        raise ValueError("ALL10groups/960, no retry or baseline reuse")
    metrics, bootstrap, summaries, input_pins = {}, {}, [], {}
    for seed in SEEDS:
        guard()
        q = read(reg["views"][str(seed)])
        book = read(dict(path=q["book_path"], sha256=q["book_sha256"]))["splits"]["arena"]
        root_ids = [r["root_id"] for r in book]
        arms = {}
        for arm, opponent in TASKS:
            path = a.arena / f"{seed}-{arm}-vs-{opponent}.json"
            owner = next(r for r in cohort["rows"]
                         if (r["seed"], r["arm"], r["opponent"]) == (seed, arm, opponent))
            packet = read(dict(path=str(path), sha256=owner["result_sha256"]))
            input_pins[str(path)] = owner["result_sha256"]
            progress = a.arena / f"{seed}-{arm}-vs-{opponent}.progress.jsonl"
            input_pins[str(progress)] = sha(progress)
            validate_progress(progress, packet["games"])
            if (packet["protocol_sha256"] != reg["views"][str(seed)]["sha256"]
                    or packet["source_commit"] != q["source_commit"]
                    or packet["opening_source_sha256"] != q["book_sha256"]
                    or packet["finished_epoch"] > owner["original_deadline_epoch"]
                    or owner["original_deadline_epoch"] > reg["deadline"]
                    or owner["finished_epoch"] > owner["original_deadline_epoch"]
                    or owner["status"] != "completed-games-awaiting-independent-audit"
                    or packet["search_nodes_per_move"] != 512 or packet["quiescence_plies"] != 2
                    or packet["max_plies"] != 400 or packet["GPU_used"]
                    or len(packet["games"]) != 96
                    or packet["candidate_sha256"] != q["models"][str(seed)][arm]["sha256"]):
                raise ValueError("exact model/runtime/source/input/original-clock headers")
            for header, name in (("helper_sha256", "tournament.py"),
                                 ("search_helper_sha256", "search.py"),
                                 ("value_helper_sha256", "value.py")):
                if packet[header] != reg["helper_sha256"][name]:
                    raise ValueError("actual executor source SHA")
            if opponent == "SF512":
                if (packet["stockfish_sha256"], packet["stockfish_nodes"],
                        packet["stockfish_threads"], packet["stockfish_hash_mib"]) != (
                        q["stockfish_sha256"], 512, 1, 16):
                    raise ValueError("same SF binary/512/threads1/hash16")
            elif packet["opponent_sha256"] != q["models"][str(seed)][opponent]["sha256"]:
                raise ValueError("exact opponent model")
            seen, games = set(), []
            for game in packet["games"]:
                guard()
                pair, color = game["opening_pair"], game["candidate_color"]
                if (type(pair) is not int or not 0 <= pair < 48
                        or color not in ("white", "black") or (pair, color) in seen):
                    raise ValueError("exact48 paired root/color set")
                seen.add((pair, color))
                summary = game_packet(game, book[pair]["opening"]["moves"], opponent)
                summaries.append(dict(seed=seed, arm=arm, opponent=opponent, **summary))
                games.append(dict(game, root_id=root_ids[pair]))
            if len(seen) != 96:
                raise ValueError("all root/color blocks closed")
            arms[f"{arm}-{opponent}"] = games
        bootstrap[seed] = evaluate_seed(arms, root_ids, seed)
        scores = {k: paired_arrays(v, root_ids, adverse=True,
                                  baseline=k in ("e8-SF512", "parent-SF512"))
                  for k, v in arms.items()}
        metrics[seed] = dict(zip(METRICS, [scores["learned-e8"],
                            scores["learned-SF512"] - scores["e8-SF512"],
                            scores["learned-parent"],
                            scores["learned-SF512"] - scores["parent-SF512"]], strict=True))
    betting = eight_primary_bounds(metrics)
    if any(sha(path) != digest for path, digest in input_pins.items()):
        raise ValueError("immutable games changed during audit")
    if time.time() >= clock["deadline"]:
        raise TimeoutError("same original900 includes allbootstrap/betting calculations")
    result = dict(schema="ONE-formal960-independent-audit-v2", integrity="PASS-all960-histories",
                  bootstrap=bootstrap, betting=betting,
                  statistical_gates_pass=all(x["passed"] for x in bootstrap.values())
                  and betting["passed"],
                  qualification_claim=False,
                  pending="final source/native/latency/eligibility join must also PASS",
                  registration_sha256=sha(a.registration), helper_sha256=sha(__file__),
                  first=clock["first"], deadline=clock["deadline"], finished=time.time(),
                  total_games=960, legal_plies=sum(x["plies"] for x in summaries),
                  actual_NN_nodes=sum(x["NN_nodes"] for x in summaries),
                  actual_SF_nodes=sum(x["SF_nodes"] for x in summaries),
                  actual_SF_overruns=sum(x["SF_overrun_moves"] for x in summaries),
                  actual_SF_max_move_nodes=max(x["SF_max_move_nodes"] for x in summaries))
    guard()
    publish(a.output, result)


if __name__ == "__main__":
    main()
