"""ROOT-only six actual packet comparisons inside its registered original900 audit."""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

from teacher_admission import module, read, sha
from value import MixedValue


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["protocol", "profile", "arena", "clock", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    q = json.loads(a.protocol.read_bytes())
    clock = json.loads(a.clock.read_bytes())
    if (
        clock["schema"] != "NNUE-strength-root-independent-audit-clock-v2"
        or clock["protocol_sha256"] != sha(a.protocol)
        or not clock["first"]
        <= time.time()
        < clock["deadline"]
        <= min(clock["first"] + 900, q["ROOToperator_end_epoch"])
    ):
        raise ValueError("actual immutable original audit900, not new recovery clock")
    os.sched_setaffinity(0, {clock["cpu_core"]})
    sys.path.insert(0, q["core_repo"] + "/src")
    import chess
    import torch

    from harbichess.training.cgroup_budget import CgroupMemoryBudget

    torch.set_num_threads(1)
    memory = CgroupMemoryBudget(15 * 2**30)

    def guard():
        memory.check()
        if time.time() >= clock["deadline"]:
            raise TimeoutError("same original900 audit")
        if shutil.disk_usage("/workspace").free < 256 * 2**20:
            raise RuntimeError("floor256")

    profile = json.loads(a.profile.read_bytes())
    if (
        profile["protocol_sha256"] != sha(a.protocol)
        or profile["status"] != "PASS-ownNNUE64-parent-trained-profile-not-strength"
    ):
        raise ValueError("actual registered profile packet reference")
    search = module(q["original_search"], "nnue_six_original_budgetsearch")
    book = read(dict(path=q["book_path"], sha256=q["book_sha256"]))["splits"]["arena"]
    out = []
    for seed in q["match_seeds"]:
        for role in ["learned", "parent", "e8"]:
            guard()
            board = chess.Board()
            for u in book[0]["opening"]["moves"]:
                board.push_uci(u)
            before = board.fen(), tuple(board.move_stack)
            packet = search.BudgetSearch(
                MixedValue(q["models"][str(seed)][role]["path"], q),
                nodes=512,
                quiescence_plies=2,
                max_depth=8,
                guard=guard,
            ).search(board)
            expected = next(
                r
                for r in profile["packets"]
                if (r["seed"], r["role"], r["opening"]) == (seed, role, 0)
            )
            actual = dict(
                move=packet.move.uci(),
                value_hex=float(packet.value).hex(),
                nodes=packet.nodes,
                evaluations=packet.evaluations,
                completed_depth=packet.completed_depth,
                root_actions=packet.root_actions,
            )
            if any(actual[k] != expected[k] for k in actual) or before != (
                board.fen(),
                tuple(board.move_stack),
            ):
                raise ValueError("fixed actual opening0 packet differs from profile")
            arena = json.loads((a.arena / f"{seed}-{role}-vs-SF512.json").read_bytes())
            game = next(
                g
                for g in arena["games"]
                if g["opening_pair"] == 0 and g["candidate_color"] == "white"
            )
            first = next(r for r in game["search_by_move"] if r["candidate"])
            comparisons = dict(
                move=first["selected_move"],
                value_hex=float(first["value"]).hex(),
                nodes=first["nodes"],
                evaluations=first["evaluations"],
                completed_depth=first["completed_depth"],
                root_actions=first["root_actions"],
            )
            if actual != comparisons:
                raise ValueError("actual arena first chronological packet differs")
            out.append(dict(seed=seed, role=role, profile_and_arena_matches=True, packet=actual))
    guard()
    with a.output.open("x") as stream:
        json.dump(
            dict(
                status="PASS-six-actual-fixed-chronological-search-packets",
                packets=out,
                first=clock["first"],
                deadline=clock["deadline"],
                finished=time.time(),
                clock_sha256=sha(a.clock),
                protocol_sha256=sha(a.protocol),
                profile_sha256=sha(a.profile),
                helper_sha256=sha(__file__),
                scope="six exact packets, not all NN move research",
            ),
            stream,
            sort_keys=True,
            indent=2,
        )


if __name__ == "__main__":
    main()
