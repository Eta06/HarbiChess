"""ROOT actual48 TRAIN C/Torch parity then48 known-root role-rotated search calls."""

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

from admission import child
from runtime import import_runtime
from teacher_admission import module, pin, read, sha
from value import MixedValue


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["protocol", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--first-epoch", type=float, required=True)
    a = p.parse_args()
    q = json.loads(a.protocol.read_bytes())
    first, end = a.first_epoch, a.first_epoch + 600
    if (
        not first
        == q["profile_first_epoch"]
        <= time.time()
        < end
        == q["profile_deadline_epoch"]
        <= q["ROOToperator_end_epoch"]
    ):
        raise ValueError("new observed ROOT600; old qualification clocks not reset")
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=q["core_repo"], text=True).strip()
        != q["source_commit"]
        or subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=q["core_repo"], text=True
        ).strip()
    ):
        raise ValueError("clean pinned original chess/evaluator core")
    for name, h in q["arena_helper_sha256"].items():
        if sha(Path(__file__).with_name(name)) != h:
            raise ValueError("all reviewed new arena helpers fixed before profile")
    os.sched_setaffinity(0, {q["cpu_core"]})
    sys.path.insert(0, q["core_repo"] + "/src")
    from harbichess.training.cgroup_budget import CgroupMemoryBudget

    memory = CgroupMemoryBudget(15 * 2**30)
    import chess
    import torch

    torch.set_num_threads(1)
    if torch.cuda.is_available():
        raise ValueError("CPU only")
    model, native, evaluator, compiled = import_runtime(q, q["match_seeds"][0])
    search = module(q["original_search"], "nnue_exact_original_budget_search")
    prior = module(q["prior_helper"], "nnue_profile_authoritative_prior")
    authoritative = evaluator.AuthoritativePrior(prior, prior.ClassicalValue())
    book = read(dict(path=q["book_path"], sha256=q["book_sha256"]))["splits"]["arena"]
    if len(book) != 8:
        raise ValueError("exact known8 profile")

    def guard():
        memory.check()
        if time.time() >= end:
            raise TimeoutError("original600 inclusive native/parity/search")
        if shutil.disk_usage("/workspace").free < 256 * 2**20:
            raise RuntimeError("diskfloor256")

    rows = []
    parity = []
    admissions = []
    ratios = []
    try:
        for seed in q["match_seeds"]:
            guard()
            state, parent, admitted = child(q, seed, native)
            admissions.append(admitted)
            info = q["children"][str(seed)]
            receipt = read(info["collection_receipt"])
            eligible = set(receipt["training_row_ids"])
            events = [json.loads(x) for x in pin(info["events"]).read_bytes().splitlines()]
            selected = [
                e["row"]
                for e in events
                if e["type"] == "search_row"
                and e["row"]["root_id"] + ":" + str(e["row"]["local_ply"]) in eligible
            ][:24]
            if len(selected) != 24:
                raise ValueError("fixed FIRST24 eligible actualTRAIN histories")
            net = model.NNUE16()
            net.load_state_dict(state, strict=True)
            compiled_value = evaluator.Evaluator(state, prior=authoritative, compiled=compiled)
            checks = []
            for ordinal, row in enumerate(selected):
                guard()
                board = chess.Board(row["root_fen"])
                for u in row["history_uci"]:
                    if not board.is_legal(chess.Move.from_uci(u)):
                        raise ValueError("fullTRAIN history")
                    board.push_uci(u)
                with torch.no_grad():
                    residual = net.residual(
                        torch.tensor(model.board_indices(board)), torch.tensor([0])
                    ).item()
                expected = math.tanh(authoritative.logit(board) + residual)
                actual = compiled_value.nonterminal(board)
                error = abs(actual - expected)
                if error > 1e-12:
                    raise ValueError("actual C/PythonTorch parity")
                checks.append(
                    dict(
                        ordinal=ordinal,
                        row_id=row["root_id"] + ":" + str(row["local_ply"]),
                        error=error,
                    )
                )
            parity.append(dict(seed=seed, rows=checks))
            values = {
                role: MixedValue(q["models"][str(seed)][role]["path"], q)
                for role in ["learned", "parent", "e8"]
            }
            for index, opening in enumerate(book):
                board = chess.Board()
                for u in opening["opening"]["moves"]:
                    board.push_uci(u)
                for role in ["learned", "parent", "e8"][index % 3 :] + ["learned", "parent", "e8"][
                    : index % 3
                ]:
                    guard()
                    before = (board.fen(), tuple(board.move_stack))
                    started = time.perf_counter()
                    r = search.BudgetSearch(
                        values[role], nodes=512, quiescence_plies=2, max_depth=8, guard=guard
                    ).search(board)
                    wall = time.perf_counter() - started
                    if (
                        before != (board.fen(), tuple(board.move_stack))
                        or not board.is_legal(r.move)
                        or r.root_actions != board.legal_moves.count()
                        or r.nodes > 512
                    ):
                        raise ValueError("alllegal unchanged original search packet")
                    rows.append(
                        dict(
                            seed=seed,
                            role=role,
                            opening=index,
                            wall_seconds=wall,
                            move=r.move.uci(),
                            value=r.value,
                            value_hex=float(r.value).hex(),
                            nodes=r.nodes,
                            evaluations=r.evaluations,
                            completed_depth=r.completed_depth,
                            root_actions=r.root_actions,
                        )
                    )
            medians = {
                role: statistics.median(
                    x["wall_seconds"] for x in rows if x["seed"] == seed and x["role"] == role
                )
                for role in values
            }
            ratio = dict(
                seed=seed,
                learned_vs_parent_median=medians["learned"] / medians["parent"],
                learned_vs_E8_median=medians["learned"] / medians["e8"],
            )
            ratios.append(ratio)
            if max(ratio["learned_vs_parent_median"], ratio["learned_vs_E8_median"]) > 1.10:
                raise ValueError("fixed actual median1.10 latency gate")
        guard()
        result = dict(status="PASS-ownNNUE64-parent-trained-profile-not-strength")
    except BaseException as e:
        result = dict(status="FAILED-preserved", error=repr(e))
        raise
    finally:
        result.update(
            schema="NNUE-own64-parent-profile-v2",
            original_first_epoch=first,
            original_deadline_epoch=end,
            finished_epoch=time.time(),
            protocol_sha256=sha(a.protocol),
            helper_sha256=sha(__file__),
            packets=rows,
            parity=parity,
            native_admissions=admissions,
            latency_ratios=ratios,
            strength_success_claimed=False,
        )
        with a.output.open("x") as stream:
            json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
