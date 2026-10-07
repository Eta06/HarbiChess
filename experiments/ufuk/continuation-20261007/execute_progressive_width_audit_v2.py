"""Independent closed-game rule replay and six exact chronological query replays."""

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import chess

BASE = Path("/workspace/work/harbichess/continuation-20261007")
ARENA = Path("/dev/shm/harbichess-progressive-width-arena-v1")
OUT = BASE / "progressive-width-independent-audit-v2"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    OUT.mkdir(exist_ok=False)
    first = time.time()
    deadline = min(first + 900, 1791448916.685839)
    os.sched_setaffinity(0, {0})
    source = BASE / "progressive-width-search-v1/qualification/develop.py"
    q = json.loads((ARENA / "result.json").read_bytes())
    pins = {
        str(source): q["helper_sha256"],
        str(ARENA / "result.json"): digest(ARENA / "result.json"),
    }
    for row in q["source_pins"].values():
        pins[row["path"]] = row["sha256"]
    registration = dict(
        first=first,
        deadline=deadline,
        cpu_core=0,
        pins=pins,
        helper_sha256=digest(__file__),
    )
    (OUT / "registration.json").write_text(json.dumps(registration, indent=2))
    result = dict(
        status="FAILED-preserved",
        first=first,
        deadline=deadline,
        strength_success=False,
    )
    try:

        def guard():
            if not first <= time.time() < deadline:
                raise TimeoutError("original independent audit900")
            for path, sha in pins.items():
                if digest(path) != sha:
                    raise ValueError("immutable source/arena input changed")

        guard()
        dev = load(source, "progressive_root_audit_packets")
        old = dev.load(q["source_pins"]["old_search"], "progressive_root_audit_old")
        advanced = dev.load(
            q["source_pins"]["advanced_search"], "progressive_root_audit_advanced"
        )
        prior = dev.load(
            q["source_pins"]["prior_helper"], "progressive_root_audit_prior"
        )
        value = prior.ClassicalValue()
        if len(q["games"]) != 32 or q["selflearning_claim"]:
            raise ValueError("fixed32 development search-only scope")
        for row in q["trace_inventory"]:
            p = ARENA / row["file"]
            if p.stat().st_size != row["bytes"] or digest(p) != row["sha256"]:
                raise ValueError("all original trace file bytes")
        total = 0
        sf_nodes = 0
        seen = set()
        packets = {"de53": [], "advanced": []}
        for game in q["games"]:
            guard()
            key = (game["role"], game["pair"], game["color"])
            if key in seen:
                raise ValueError("duplicate root/role/color")
            seen.add(key)
            color = game["color"] == "white"
            board = chess.Board()
            own = iter(game["searches"])
            sf = iter(game["stockfish"])
            if game["moves"][: len(game["opening"])] != game["opening"]:
                raise ValueError("original full opening prefix")
            for ply, uci in enumerate(game["moves"]):
                if board.outcome(claim_draw=True) is not None:
                    raise ValueError("move after terminal")
                move = chess.Move.from_uci(uci)
                if move not in board.legal_moves:
                    raise ValueError("illegal historical move")
                if ply >= len(game["opening"]):
                    pk = next(own if board.turn == color else sf)
                    if pk["move"] != uci or pk["preaction_fen"] != board.fen():
                        raise ValueError("full rule state and selected move")
                    if board.turn == color:
                        if pk["history_uci"] != game["moves"][:ply]:
                            raise ValueError("exact full history")
                        if (
                            pk["root_actions"] != board.legal_moves.count()
                            or pk["nodes"] > 512
                        ):
                            raise ValueError("all legal roots/charged node budget")
                        trace = pk["eval_trace"]
                        raw = (ARENA / trace["file"]).read_bytes()[
                            trace["offset_bytes"] : trace["offset_bytes"]
                            + trace["bytes"]
                        ]
                        if (
                            len(raw) != trace["bytes"]
                            or hashlib.sha256(raw).hexdigest() != trace["sha256"]
                        ):
                            raise ValueError("complete ordered query trace bytes")
                        packets[game["role"]].append(pk)
                    else:
                        if (
                            pk["ply"] != ply
                            or pk["nominal_nodes"] != 512
                            or pk["actual_nodes"] < 0
                        ):
                            raise ValueError("actual Stockfish counters")
                        sf_nodes += pk["actual_nodes"]
                board.push(move)
                total += 1
            if next(own, None) is not None or next(sf, None) is not None:
                raise ValueError("unused move/search packets")
            outcome = board.outcome(claim_draw=True)
            if outcome is None or game["outcome_status"] != "KNOWN":
                raise ValueError("this closed32 experiment has no caps")
            score = 0.5 if outcome.winner is None else float(outcome.winner == color)
            if (
                score != game["score"]
                or outcome.termination.name != game["termination"]
            ):
                raise ValueError("independently recomputed terminal and score")
        if seen != {
            (role, pair, color)
            for role in packets
            for pair in range(8)
            for color in ["white", "black"]
        }:
            raise ValueError("all32 paired games exactly once")
        replays = []
        for role, rows in packets.items():
            for index in [0, len(rows) // 2, len(rows) - 1]:
                guard()
                pk = rows[index]
                board = dev.replay(chess.STARTING_FEN, pk["history_uci"])
                trace = dev.Trace(OUT, value.nonterminal)
                trace.begin(board)
                kwargs = dict(
                    nodes=512, quiescence_plies=2, max_depth=8, guard=lambda: None
                )
                search = (
                    old.BudgetSearch(trace, **kwargs)
                    if role == "de53"
                    else advanced.BudgetSearch(
                        trace, **kwargs, incheck_extensions=1, root_width="progressive"
                    )
                )
                before = (board.fen(), tuple(board.move_stack))
                fresh = json.loads(
                    dev.canonical(dev.packet(search.search(board), board))
                )
                if before != (board.fen(), tuple(board.move_stack)):
                    raise ValueError("search mutated rule history")
                if (
                    any(pk[k] != v for k, v in fresh.items())
                    or hashlib.sha256(trace.raw).hexdigest()
                    != pk["eval_trace"]["sha256"]
                ):
                    raise ValueError(
                        "six actual scalar HEX/counter/ordered query replays"
                    )
                replays.append(
                    dict(
                        role=role,
                        chronological_index=index,
                        query_sha256=pk["eval_trace"]["sha256"],
                    )
                )
        guard()
        result.update(
            status="PASS-closed32-and-six-actual-replays-not-strength",
            legal_plies=total,
            stockfish_actual_nodes=sf_nodes,
            replays=replays,
            original_observed_scores=q["summary"],
        )
    except BaseException as exc:
        result["error"] = repr(exc)
        raise
    finally:
        result["finished"] = time.time()
        (OUT / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
