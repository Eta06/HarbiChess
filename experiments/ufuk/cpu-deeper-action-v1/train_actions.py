"""ROOT-only fixed64 legal-CE learner; no automatic jobs or external labels."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import chess
from action_learner import MATH, Learner
from action_model import features, model_dict


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def publish(path, obj):
    with Path(path).open("xb") as stream:
        stream.write(canonical(obj) + b"\n")


def prepare(path, expected):
    if sha(path) != expected:
        raise ValueError("sealed labels SHA")
    compressed = Path(path).read_bytes()
    if len(compressed) > 16 * 1024**2:
        raise ValueError("compressed labels bound")
    import zlib

    dec = zlib.decompressobj(31)
    raw = dec.decompress(compressed, 64 * 1024**2 + 1)
    if not dec.eof or dec.unused_data or dec.unconsumed_tail or len(raw) > 64 * 1024**2:
        raise ValueError("labels inflation/trailing bound")
    payload = json.loads(raw)
    if payload["schema"] != "own-search-deeper-bestmove-labels-v1" or len(payload["roots"]) != 1024:
        raise ValueError("fixed1024 own action labels")
    if (
        payload["search"]["nodes"] != 8192
        or payload["search"]["qdepth"] != 2
        or payload["search"]["max_depth"] != 8
    ):
        raise ValueError("frozen search budget")
    groups = {}
    ids = set()
    for row in payload["roots"]:
        if row["row_id"] in ids:
            raise ValueError("duplicate root")
        ids.add(row["row_id"])
        h = row["history"]
        b = chess.Board(h["root_fen"])
        if not b.is_valid():
            raise ValueError("invalid full root")
        for action in h["prefix_uci"]:
            m = chess.Move.from_uci(action)
            if m not in b.legal_moves:
                raise ValueError("illegal full prefix")
            b.push(m)
        if b.is_game_over(claim_draw=True):
            raise ValueError("terminal root")
        actions = sorted(b.legal_moves, key=lambda m: m.uci())
        if (
            row["actions"] != [m.uci() for m in actions]
            or row["target"] != row["actions"].index(row["selected_uci"])
            or row["features"] != [list(features(b, m)) for m in actions]
        ):
            raise ValueError("legal action context/target alignment")
        expected_h = hashlib.sha256(
            (h["root_fen"] + "\n" + " ".join(h["prefix_uci"])).encode()
        ).hexdigest()
        if row["history_sha256"] != expected_h:
            raise ValueError("fullhistory binding")
        groups.setdefault(row["trajectory_id"], []).append(row)
    if len(groups) < 16:
        raise ValueError("original training trajectory coverage")
    return groups, payload


def main():
    p = argparse.ArgumentParser()
    for name in ["protocol", "labels", "output", "source-repo"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--protocol-sha256", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--cpu-core", type=int, required=True)
    p.add_argument("--deadline", type=float, required=True)
    p.add_argument("--stop-at", type=int, default=64)
    p.add_argument("--resume", type=Path)
    p.add_argument("--resume-sha256")
    p.add_argument("--audit-only", action="store_true")
    a = p.parse_args()
    if sha(a.protocol) != a.protocol_sha256:
        raise ValueError("protocol SHA")
    q = json.loads(a.protocol.read_bytes())
    if (
        q["schema"] != "own-deeper-bestmove-training-protocol-v1"
        or q["math"] != MATH
        or a.deadline != q["deadline_by_seed"][str(a.seed)]
    ):
        raise ValueError("fixed math/clock")
    old = Path(q["classical_helper_directory"])
    if sha(old / "runtime.py") != q["runtime_sha256"] or sha(old / "value.py") != q["value_sha256"]:
        raise ValueError("exact immutable runtime/value source")
    sys.path.append(str(old))
    from runtime import guard as runtime_guard
    from value import features as state_features

    def guard():
        if not time.time() < a.deadline <= 1791273600:
            raise TimeoutError("registered original clock expired")
        runtime_guard(a.deadline, a.source_repo, a.output)

    guard()
    if set(q["helpers"]) != {
        "action_model.py",
        "action_learner.py",
        "action_labels.py",
        "produce_actions.py",
        "select_roots.py",
        "student_search.py",
        "train_actions.py",
        "qualify_actions.py",
    }:
        raise ValueError("full helper closure")
    for name, digest in q["helpers"].items():
        if sha(Path(__file__).with_name(name)) != digest:
            raise ValueError("helper source changed")
    if subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=a.source_repo, text=True
    ).strip() != q["source_commit"] or subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=a.source_repo, text=True
    ):
        raise ValueError("clean core source")
    os.sched_setaffinity(0, {a.cpu_core})
    pair = q["inputs"][str(a.seed)]
    groups, labels = prepare(a.labels, pair["labels_sha256"])
    for rows in groups.values():
        for row in rows:
            b = chess.Board(row["history"]["root_fen"])
            for text in row["history"]["prefix_uci"]:
                b.push_uci(text)
            if row["state18"] != list(state_features(b)):
                raise ValueError("frozen18 context differs")
    if (
        labels["seed"] != a.seed
        or labels["source_commit"] != q["source_commit"]
        or labels["registration_sha256"] != pair["producer_registration_sha256"]
    ):
        raise ValueError("original producer binding")
    contract = dict(
        updates=64,
        seed=a.seed,
        source_commit=q["source_commit"],
        protocol_sha256=a.protocol_sha256,
        labels_sha256=sha(a.labels),
        producer_registration_sha256=labels["registration_sha256"],
        math=MATH,
        deadline=a.deadline,
        scope="offline-own-bestmove-CE-not-value-visits",
    )
    state = None
    if a.resume:
        if sha(a.resume) != a.resume_sha256:
            raise ValueError("native SHA")
        state = json.loads(a.resume.read_bytes())
    learner = Learner(a.seed, contract, state)
    if not learner.step <= a.stop_at <= 64:
        raise ValueError("no counter rewind/extension")
    if a.audit_only:
        guard()
        print(json.dumps({"status": "strict-action-native-load-PASS", "step": learner.step}))
        return
    if state is None:
        a.output.mkdir(parents=True, exist_ok=False)
        publish(a.output / "step-00000000.json", learner.native())
    elif json.loads((a.output / "step-00000000.json").read_bytes())["contract"] != contract:
        raise ValueError("output original contract")
    while learner.step < a.stop_at:
        learner.advance(groups, min(a.stop_at, learner.step + 4), guard)
        publish(a.output / f"step-{learner.step:08d}.json", learner.native())
    if learner.step == 64:
        publish(a.output / "candidate.json", model_dict(learner.weights))
    guard()
    print(json.dumps({"status": "fixed-action-fit-not-strength", "step": learner.step}))


if __name__ == "__main__":
    main()
