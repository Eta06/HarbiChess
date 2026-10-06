"""ROOT-registered trained teacher profile; no self-learning claim."""
import argparse
import gzip
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from admission import module, pin, read, sha, teacher


def isolate_stdlib_profile():
    """Keep frozen profile.py evidence out of cProfile standard-library lookup."""
    here = Path(__file__).resolve().parent
    sys.path[:] = [entry for entry in sys.path
                   if Path(entry or os.getcwd()).resolve() != here]
    cached = sys.modules.get("profile")
    if cached is not None and Path(getattr(cached, "__file__", "/")).resolve().parent == here:
        raise RuntimeError("local profile already imported; require fresh interpreter")


def main(regpath):
    isolate_stdlib_profile()
    reg = json.loads(Path(regpath).read_bytes())
    if reg["schema"] != "NNUE-trained-teacher-profile-v1" or reg["status"] != "registered":
        raise ValueError("prospective ROOT registration")
    first, end = reg["first"], reg["deadline"]
    if not first <= time.time() < end == first + 600 <= 1791273600:
        raise ValueError("original trained profile600")
    os.sched_setaffinity(0, {reg["cpu_core"]})
    for path, h in reg["input_pins"].items():
        pin(dict(path=path, sha256=h))
    core = reg["core_repo"]
    if (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=core, text=True).strip()
            != reg["core_commit"]
            or subprocess.check_output(["git", "status", "--porcelain"], cwd=core, text=True)):
        raise ValueError("clean original core")
    sys.path.insert(0, str(Path(core) / "src"))
    sys.path.insert(0, reg["nnue_directory"])
    import _kingbucket16
    import chess
    import model
    import native
    import torch
    from evaluator import AuthoritativePrior, Evaluator

    from harbichess.training.cgroup_budget import CgroupMemoryBudget

    torch.set_num_threads(1)
    if torch.cuda.is_available():
        raise ValueError("CPU-only")
    pin(reg["binary"])
    if Path(_kingbucket16.__file__).resolve() != Path(reg["binary"]["path"]).resolve():
        raise ValueError("compiled reader identity")
    value = module(reg["prior_helper"], "nnue_profile_original_prior")
    search = module(reg["search_helper"], "nnue_profile_original_search")
    mixed = module(reg["mixed_helper"], "nnue_profile_original_e8")
    budget = CgroupMemoryBudget(15 * 2**30)

    def guard():
        if time.time() >= end:
            raise TimeoutError("original trained profile clock")
        budget.check()
        if shutil.disk_usage("/workspace").free < 256 * 2**20:
            raise RuntimeError("original diskfloor")

    e8_protocol = read(reg["e8_protocol"])
    e8 = mixed.MixedValue(pin(reg["e8"]), e8_protocol)
    prior = value.ClassicalValue()
    authoritative = AuthoritativePrior(value, prior)
    book = read(reg["book"])["splits"]["arena"]
    if len(book) != 8 or reg["search_math"] != dict(nodes=512, quiescence_plies=2, max_depth=8):
        raise ValueError("fixed known8/original512q2max8")
    output = Path(reg["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError("immutable profile result")
    rows, admissions, parity, ratios = [], [], [], []
    try:
        for seed in [20262905, 20262906]:
            info = reg["teachers"][str(seed)]
            state, zero_state, admitted = teacher(info, native)
            admissions.append(dict(seed=seed, **admitted))
            net = model.NNUE16()
            net.load_state_dict(state, strict=True)
            learned = Evaluator(state, prior=authoritative, compiled=_kingbucket16)
            zero = Evaluator(zero_state, prior=authoritative, compiled=_kingbucket16)
            labels = json.loads(gzip.decompress(pin(info["labels"]).read_bytes()))
            identities = []
            for i, row in enumerate(labels["rows"][:24]):
                guard()
                board = chess.Board(row["root_fen"])
                for uci in row["prefix_uci"]:
                    move = chess.Move.from_uci(uci)
                    if move not in board.legal_moves:
                        raise ValueError("illegal TRAIN history")
                    board.push(move)
                if board.outcome(claim_draw=True) is not None or row["ordinal"] != i:
                    raise ValueError("nonterminal fixed TRAIN selection")
                with torch.no_grad():
                    residual = net.residual(torch.tensor(model.board_indices(board)),
                                            torch.tensor([0])).item()
                expected = math.tanh(authoritative.logit(board) + residual)
                error = abs(learned.nonterminal(board) - expected)
                if error > 1e-12 or zero.nonterminal(board).hex() != prior.nonterminal(board).hex():
                    raise ValueError("trained C/Torch or exactzero parity")
                identities.append(dict(ordinal=i, row_id=row["row_id"],
                                       history_sha256=row["history_sha256"], error=error))
            if len(identities) != 24:
                raise ValueError("fixed 24 TRAIN histories eachseed")
            parity.append(dict(seed=seed, rows=identities))
            roles = dict(teacher=learned, prior=prior, e8=e8)
            for index, opening in enumerate(book):
                board = chess.Board()
                for uci in opening["opening"]["moves"]:
                    if chess.Move.from_uci(uci) not in board.legal_moves:
                        raise ValueError("known8 full legal history")
                    board.push_uci(uci)
                before = board.fen(), tuple(board.move_stack)
                # Fixed role rotation balances warm order without RNG draws.
                order = list(roles)
                order = order[index % 3:] + order[:index % 3]
                for role in order:
                    guard()
                    engine = search.BudgetSearch(roles[role], guard=guard, **reg["search_math"])
                    started = time.perf_counter()
                    packet = engine.search(board)
                    elapsed = time.perf_counter() - started
                    if (packet.move not in board.legal_moves
                            or packet.root_actions != board.legal_moves.count()
                            or packet.nodes > 512
                            or before != (board.fen(), tuple(board.move_stack))):
                        raise ValueError("fulllegal/node/history search packet")
                    rows.append(dict(seed=seed, role=role, opening=index, seconds=elapsed,
                                     move=packet.move.uci(), value=packet.value, nodes=packet.nodes,
                                     evaluations=packet.evaluations,
                                     completed_depth=packet.completed_depth,
                                     root_actions=packet.root_actions))
            totals = {r: sum(x["seconds"] for x in rows if x["seed"] == seed and x["role"] == r)
                      for r in roles}
            ratio = dict(seed=seed, teacher_vs_prior=totals["teacher"] / totals["prior"],
                         teacher_vs_e8=totals["teacher"] / totals["e8"])
            ratios.append(ratio)
            if max(ratio["teacher_vs_prior"], ratio["teacher_vs_e8"]) > 1.10:
                raise ValueError("fixed trained speed gate1.10")
        guard()
        result = dict(status="PASS-trained-teacher-profile-not-selflearning",
                      scope="48 parity and 48 search calls, zeroHEX48", latency_ratios=ratios)
    except BaseException as e:
        result = dict(status="FAILED-preserved", error=repr(e))
        raise
    finally:
        result.update(schema="NNUE-trained-teacher-profile-result-v1", first=first, deadline=end,
                      finished=time.time(), registration_sha256=sha(regpath),
                      helper_sha256=sha(__file__), admission_sha256=sha(Path(__file__).with_name(
                          "admission.py")), admissions=admissions, parity=parity, packets=rows,
                      strength_success_claimed=False, teacher_is_selflearning=False)
        with output.open("x") as f:
            json.dump(result, f, sort_keys=True, indent=2, allow_nan=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registration", required=True)
    main(p.parse_args().registration)
