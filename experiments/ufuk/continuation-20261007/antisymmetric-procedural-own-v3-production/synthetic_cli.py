"""Only source-bound synthetic native math, no production receipts or admission."""

import argparse
import json
from pathlib import Path

import torch
import zero_parent
from model import NNUE16
from native import Learner, load_native


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["contract", "rows", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--stop", type=int, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--audit-only", action="store_true")
    a = p.parse_args()
    c = json.loads(a.contract.read_bytes())
    torch.set_num_threads(1)
    if (
        c.get("synthetic_only") is not True
        or c["phase"] != "antisymmetric-procedural-own-learning-v3"
    ):
        raise ValueError("only explicit synthetic production-math fixture")

    def fixture_weights(binding, seed):
        state = torch.get_rng_state().clone()
        try:
            torch.manual_seed(seed)
            return NNUE16().state_dict()
        finally:
            torch.set_rng_state(state)

    zero_parent.admitted_weights = fixture_weights
    learner = load_native(a.resume, c) if a.resume else Learner(c)
    if not a.audit_only:
        learner.advance(json.loads(a.rows.read_bytes()), a.stop)
    with a.output.open("xb") as f:
        torch.save(learner.native(), f)


if __name__ == "__main__":
    main()
