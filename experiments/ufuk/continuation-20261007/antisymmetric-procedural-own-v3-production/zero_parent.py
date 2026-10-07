"""Actual typed zero0 parent, two fresh strict opens; weights-only bridge not resume."""

import random

import torch
from native import bits_equal, load_native
from support import pinned, pins_tree, read, sha


def admit(binding, seed):
    before = random.getstate(), torch.get_rng_state().clone()
    try:
        c = read(binding["contract"])
        pins_tree(c)
        r = read(binding["result"])
        if (
            c["phase"] != "human-prior-zero-antisymmetric-init-v3"
            or c["seed"] != seed
            or c["updates"] != 0
            or r["schema"] != "antisymmetric-zero-two-fresh-opens-result-v3"
            or r["status"] != "PASS-typedzero-fullnative-two-fresh-opens-not-playing-strength"
            or r["contract"] != binding["contract"]
            or r["native"] != binding["native"]
            or r["candidate"] != binding["candidate"]
            or len(r["commands"]) != 2
            or not c["first"] <= r["finished"] <= c["deadline"]
            or r["first"] != c["first"]
            or r["deadline"] != c["deadline"]
        ):
            raise ValueError("actual typed zero parent/result/clock")
        for item in r["commands"]:
            argv = item["command"]
            if (
                item["returncode"] != 0
                or "--audit-only" not in argv
                or argv[1] != c["source_refs"]["initialize.py"]["path"]
                or argv[argv.index("--native") + 1] != binding["native"]["path"]
                or argv[argv.index("--contract") + 1] != binding["contract"]["path"]
                or not c["first"] <= item["started"] <= item["finished"] <= c["deadline"]
                or sha(item["log_path"]) != item["log_sha256"]
                or read({"path": item["log_path"], "sha256": item["log_sha256"]})
                != dict(status="PASS-strict-antisymmetric-zero-native", step=0)
            ):
                raise ValueError("both actual source-bound fresh interpreter opens")
        learner = load_native(pinned(binding["native"]), c)
        packet = torch.load(pinned(binding["candidate"]), weights_only=False, map_location="cpu")
        if not bits_equal(packet, learner.candidate()) or learner.optimizer.state:
            raise ValueError("candidate/native initial model baseline emptyAdam")
        return learner.native(), c
    finally:
        random.setstate(before[0])
        torch.set_rng_state(before[1])


def admitted_weights(binding, seed):
    state, _ = admit(binding, seed)
    return state["model"]
