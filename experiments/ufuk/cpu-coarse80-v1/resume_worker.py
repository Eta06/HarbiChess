from __future__ import annotations

import argparse
import json
from pathlib import Path

from learner80 import Learner, decode_native, encode_native


def main():
    parser = argparse.ArgumentParser(description="strict coarse80 full-state resume worker")
    parser.add_argument("--groups", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--native", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target", type=int, required=True)
    args = parser.parse_args()
    groups = json.loads(Path(args.groups).read_text())
    contract = json.loads(Path(args.contract).read_text())
    state = decode_native(Path(args.native).read_bytes(), contract)
    learner = Learner(contract["seed"], contract, state=state)
    learner.advance(groups, args.target)
    Path(args.out).write_bytes(encode_native(learner.native()))


if __name__ == "__main__":
    main()
