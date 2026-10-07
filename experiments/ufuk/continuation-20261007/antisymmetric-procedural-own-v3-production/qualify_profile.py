"""ROOT actual zero24 packet/alltrace or trained48 C/Torch + paired latency gate."""

import argparse
import hashlib
import json
import statistics
import time

import chess
import torch
from admission import child
from compiled_evaluator import Evaluator, load_prior
from model import NNUE16, tensor_batch
from support import canonical, guard, module, publish, read, ref
from value import MixedValue
from zero_parent import admit


def board_at(row):
    b = chess.Board(row["root_fen"])
    for token in row["history_uci"]:
        m = chess.Move.from_uci(token)
        if not b.is_legal(m):
            raise ValueError("complete actual legal history")
        b.push(m)
    if not b.is_valid() or b.outcome(claim_draw=True) is not None:
        raise ValueError("actual nonterminal TRAIN root")
    return b


def packet(search, value, b, check):
    trace = []

    def evaluator(board):
        trace.append(
            dict(root_fen=board.root().fen(), history_uci=[m.uci() for m in board.move_stack])
        )
        return value(board)

    first = time.perf_counter()
    r = search.BudgetSearch(
        evaluator, nodes=512, quiescence_plies=2, max_depth=8, guard=check
    ).search(b)
    elapsed = time.perf_counter() - first
    result = dict(
        move=None if r.move is None else r.move.uci(),
        value_hex=r.value.hex(),
        nodes=r.nodes,
        evaluations=r.evaluations,
        completed_depth=r.completed_depth,
        root_actions=r.root_actions,
    )
    return result, trace, elapsed


def execute(q, clock, mode, out):
    torch.set_num_threads(1)
    check = guard(clock, 600)
    if q["schema"] != "antisymmetric-procedural-known160-protocol-v3":
        raise ValueError("new typed strength protocol")
    if (
        q["original_search"]["sha256"]
        != "de53c14728a67b7772f18b396ac8ef5c35a144d4e4e616fec40099cd461a6670"
    ):
        raise ValueError("original same512 search")
    search = module(q["original_search"], "paired_actual_search")
    prior = load_prior(q["prior_helper"]).ClassicalValue()
    packets = []
    parity = []
    ratios = []
    for seed in q["match_seeds"]:
        state, zc = admit(q["zeros"][str(seed)], seed)
        weights = state["model"]
        if mode == "trained":
            weights, c = child(q["children"][str(seed)])
            weights = weights["model"]
        else:
            c = zc
        evaluator = Evaluator(
            weights, compiled_refs=c["compiled_refs"], prior_ref=c["prior_helper"]
        )
        data = read(q["paired_datasets"][str(seed)])
        if len(data["rows"]) != 1024:
            raise ValueError("fixed completed common1024 data")
        model = NNUE16()
        model.load_state_dict(weights)
        if mode == "trained":
            for row in data["rows"][:24]:
                check()
                b = board_at(row)
                ids, offsets, p, _ = tensor_batch([row])
                with torch.no_grad():
                    result = float(torch.tanh(p + model.residual(ids, offsets)).item())
                compiled = evaluator.nonterminal(b)
                error = abs(result - compiled)
                if error > 1e-12:
                    raise ValueError("actual trained paired C/Torch numerical bound")
                parity.append(
                    dict(
                        seed=seed,
                        row_id=row["source_row_id"],
                        Torch_hex=result.hex(),
                        C_hex=compiled.hex(),
                        error=error,
                    )
                )
        count = 12 if mode == "zero" else 24
        times = {"learned": [], "parent": [], "e8": []}
        e8 = MixedValue(q["models"][str(seed)]["e8"]["path"], q) if mode == "trained" else None
        for i, row in enumerate(data["rows"][:count]):
            check()
            b = board_at(row)
            roles = [("learned", evaluator), ("parent", prior)]
            if mode == "trained":
                roles.append(("e8", e8))
            roles = roles[i % len(roles) :] + roles[: i % len(roles)]
            actual = {}
            for name, fn in roles:
                r, t, wall = packet(search, fn, b, check)
                times[name].append(wall)
                actual[name] = dict(
                    packet=r,
                    trace=t,
                    seconds=wall,
                    trace_sha256=hashlib.sha256(canonical(t)).hexdigest(),
                )
            if mode == "zero" and (
                actual["learned"]["packet"] != actual["parent"]["packet"]
                or actual["learned"]["trace"] != actual["parent"]["trace"]
                or evaluator.nonterminal(b).hex() != prior.nonterminal(b).hex()
            ):
                raise ValueError("zero parent exact packets/all evaluator histories/HEX")
            packets.append(dict(seed=seed, row_id=row["source_row_id"], roles=actual))
        if mode == "trained":
            r = dict(
                seed=seed,
                learned_vs_parent_median=statistics.median(times["learned"])
                / statistics.median(times["parent"]),
                learned_vs_E8_median=statistics.median(times["learned"])
                / statistics.median(times["e8"]),
            )
            if max(r["learned_vs_parent_median"], r["learned_vs_E8_median"]) > 1.10:
                raise ValueError("unchanged original1.10 latency gate")
            ratios.append(r)
    fixed_arena_root_probes = []
    if mode == "trained":
        book = read(dict(path=q["book_path"], sha256=q["book_sha256"]))["splits"]["arena"]
        for seed in q["match_seeds"]:
            for role in ["learned", "parent", "e8"]:
                check()
                board = chess.Board()
                for token in book[0]["opening"]["moves"]:
                    move = chess.Move.from_uci(token)
                    if not board.is_legal(move):
                        raise ValueError("fixed known0 fullhistory")
                    board.push(move)
                result, trace, elapsed = packet(
                    search, MixedValue(q["models"][str(seed)][role]["path"], q), board, check
                )
                fixed_arena_root_probes.append(
                    dict(
                        seed=seed,
                        role=role,
                        opening=0,
                        **result,
                        evaluator_history_trace=trace,
                        seconds=elapsed,
                    )
                )
    check()
    publish(
        out,
        dict(
            schema="antisymmetric-actual-profile-result-v3",
            status="PASS-zero24-exact-packets-traces-not-strength"
            if mode == "zero"
            else "PASS-trained48-parity-paired-latency-not-strength",
            mode=mode,
            protocol_sha256=clock["protocol"]["sha256"],
            helper_sha256=clock["helper"]["sha256"],
            original_deadline_epoch=clock["deadline"],
            clock=clock,
            finished=time.time(),
            packets=packets,
            parity=parity,
            latency_ratios=ratios,
            fixed_arena_root_probes=fixed_arena_root_probes,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["protocol", "clock", "output"]:
        p.add_argument("--" + name, required=True, type=__import__("pathlib").Path)
    p.add_argument("--mode", choices=["zero", "trained"], required=True)
    a = p.parse_args()
    q = json.loads(a.protocol.read_bytes())
    c = json.loads(a.clock.read_bytes())
    if c["protocol"] != ref(a.protocol) or c["helper"] != ref(__file__):
        raise ValueError("exact bound profile/controller helper")
    try:
        execute(q, c, a.mode, a.output)
    except BaseException as error:
        if not a.output.exists():
            publish(
                a.output,
                dict(
                    schema="antisymmetric-profile-failure-preserved-v3",
                    status="FAILED-preserved",
                    mode=a.mode,
                    clock=c,
                    error=repr(error),
                    finished=time.time(),
                ),
            )
        raise
