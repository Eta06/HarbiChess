"""Strict read-only parent parsing, original clocks and recovery packet accounting."""

import hashlib
import json
from pathlib import Path

import chess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_json(path, expected):
    if sha(path) != expected:
        raise ValueError("parent SHA differs")
    return json.loads(Path(path).read_text())


def clock(invocation, whole_end, now):
    first = invocation["original_first_epoch"]
    deadline = invocation["original_deadline_epoch"]
    if deadline != min(first + 2700, whole_end) or first > now or now >= deadline:
        raise ValueError("original tournament deadline invalid/exhausted; no reset")
    return first, deadline


def parse_progress(path, expected_sha, book, opponent):
    if sha(path) != expected_sha:
        raise ValueError("immutable parent progress SHA differs")
    expected = [(pair, color) for pair in range(len(book)) for color in ("white", "black")]
    completed, raw_ends, active = {}, {}, None
    for line in Path(path).read_text().splitlines():
        packet = json.loads(line)
        kind = packet["type"]
        if kind == "game_start":
            if active is not None:
                raise ValueError("overlapping game starts")
            key = (packet["pair"], packet["color"])
            if key != expected[len(completed)]:
                raise ValueError("parent game schedule differs")
            board = chess.Board()
            for uci in book[key[0]]["opening"]["moves"]:
                board.push_uci(uci)
            active = {"key": key, "board": board, "moves": []}
        elif kind == "move":
            if active is None or active["board"].outcome(claim_draw=True) is not None:
                raise ValueError("move without live game")
            board = active["board"]
            move = chess.Move.from_uci(packet["uci"])
            if move not in board.legal_moves or packet["ply"] != board.ply() + 1:
                raise ValueError("illegal parent move/ply")
            board.push(move)
            active["moves"].append(packet["uci"])
        elif kind == "game_end":
            if active is None:
                raise ValueError("end without active game")
            game = packet["game"]
            key = active["key"]
            if (game["opening_pair"], game["candidate_color"]) != key:
                raise ValueError("game_end identity differs")
            if game["moves"] != [m.uci() for m in active["board"].move_stack]:
                raise ValueError("game_end differs from published moves")
            completed[key] = game
            raw_ends[key] = line
            active = None
        else:
            raise ValueError("unknown parent event")
    if active is not None and active["moves"] and opponent == "SF512":
        raise ValueError("SF partial game cannot resume hidden engine TT")
    if active is not None:
        active = {"key": active["key"], "moves": active["moves"]}
    return completed, raw_ends, active


def match_prefix(packet, recorded_uci):
    if packet["selected_move"] != recorded_uci:
        raise ValueError("chronological neural prefix does not reproduce exactly")


def validate_recovery(x, owner, contract):
    """No old-helper alias: actual executor SHA separately bound to old parent."""
    if x["schema"] != "cpu-all-root-quiescent-alpha-beta-arena-recovery-v2":
        raise ValueError("not recovery v2")
    recovery = x["recovery"]
    if x["helper_sha256"] != contract["recovery_executor_sha256"]:
        raise ValueError("actual executor SHA differs")
    if recovery["old_parent_executor_sha256"] != contract["inputs"]["tournament.py"]:
        raise ValueError("old parent helper SHA differs")
    parent = checked_json(recovery["parent_invocation_path"], recovery["parent_invocation_sha256"])
    if (x["started_epoch"], x["original_deadline_epoch"]) != (
        parent["original_first_epoch"],
        parent["original_deadline_epoch"],
    ):
        raise ValueError("original group clock reset")
    if owner["original_deadline_epoch"] != parent["original_deadline_epoch"]:
        raise ValueError("owner clock differs")
    progress = Path(recovery["parent_progress_path"])
    if sha(progress) != recovery["parent_progress_sha256"]:
        raise ValueError("parent progress changed")
    ends = [
        json.loads(line)["game"]
        for line in progress.read_text().splitlines()
        if json.loads(line)["type"] == "game_end"
    ]
    if x["games"][: len(ends)] != ends:
        raise ValueError("original complete games changed")
    active = None
    for line in progress.read_text().splitlines():
        event = json.loads(line)
        if event["type"] == "game_start":
            active = {"key": (event["pair"], event["color"]), "moves": []}
        elif event["type"] == "move":
            active["moves"].append((event["ply"], event["uci"]))
        elif event["type"] == "game_end":
            active = None
    if active and active["moves"]:
        game = x["games"][len(ends)]
        if (game["opening_pair"], game["candidate_color"]) != active["key"]:
            raise ValueError("partial game identity differs")
        packets = [
            r
            for r in game["search_by_move"]
            if r.get("measurement_origin") == "actual-recovery-prefix-research"
        ]
        if [(r["ply"], r["selected_move"]) for r in packets] != active["moves"]:
            raise ValueError("partial prefix recovery packets differ")
        if recovery["original_closed_games_reused"] != len(ends):
            raise ValueError("closed prefix count differs")
    replayed = [
        r
        for g in x["games"]
        for r in g["search_by_move"]
        if r.get("measurement_origin") == "actual-recovery-prefix-research"
    ]
    if len(replayed) != recovery["duplicate_NN_moves_researched"]:
        raise ValueError("duplicate work count differs")
    for field, packet in [
        ("duplicate_NN_nodes", "nodes"),
        ("duplicate_NN_evaluations", "evaluations"),
    ]:
        if recovery[field] != sum(r[packet] for r in replayed):
            raise ValueError("duplicate NN work differs")
    if not recovery["old_partial_latency_unknown"] or recovery["SF_partial_resume_claimed"]:
        raise ValueError("latency/engine recovery claim differs")
