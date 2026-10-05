"""One bounded segment. No automatic training, epoch cycling or production default."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from journal_v2 import Actor, load_module, read, save, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--search-helper", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path)
    parser.add_argument("--value-helper", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--anchor-model", type=Path)
    parser.add_argument("--anchor-helper", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-actions", type=int, required=True)
    parser.add_argument("--original-deadline", type=float, required=True)
    parser.add_argument("--synthetic-zero-fixture", action="store_true")
    args = parser.parse_args()
    if sha(args.config) != args.config_sha256:
        raise ValueError("config input SHA differs")
    config = json.loads(args.config.read_text())
    if args.original_deadline != config["original_deadline_epoch"]:
        raise ValueError("original deadline cannot change across segments")
    if sha(Path(__file__).with_name("journal_v2.py")) != config["producer_sha256"]:
        raise ValueError("journal producer SHA differs")
    if sha(__file__) != config["runner_sha256"]:
        raise ValueError("runner SHA differs")
    search = load_module(args.search_helper, config["search_helper_sha256"])

    def guard():
        if time.time() >= args.original_deadline:
            raise TimeoutError("original deadline exhausted; no reset")
        if (
            sum(p.stat().st_size for p in args.output.parent.iterdir() if p.is_file())
            >= 16 * 1024 * 1024
        ):
            raise RuntimeError("v2 proposal artifact directory reached 16MiB")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    guard()
    if args.synthetic_zero_fixture:
        if config["evaluator_identity"] != "synthetic-zero-fixture-NOT-NN-qualified":
            raise ValueError("synthetic fixture identity missing")

        def evaluator(board):
            return 0.0

        def anchor_evaluator(board):
            return (1 / 3, 1 / 3, 1 / 3)
    else:
        if config["evaluator_identity"] != "frozen-real-NeuralValue-CPU":
            raise ValueError("real evaluator identity missing")
        if (
            args.source_repo is None or args.model is None or args.value_helper is None
            or args.anchor_model is None or args.anchor_helper is None
        ):
            raise ValueError("real source/model/value and frozen e8 anchor inputs required")
        source = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.source_repo, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=args.source_repo, text=True
        )
        if source != config["source_commit"] or dirty:
            raise ValueError("real source is not clean pinned producer")
        if sha(args.model) != config["model_sha256"]:
            raise ValueError("frozen model SHA differs")
        if sha(args.anchor_model) != config["anchor_model_sha256"]:
            raise ValueError("frozen e8 anchor model SHA differs")
        if sha(args.anchor_helper) != config["anchor_helper_sha256"]:
            raise ValueError("frozen e8 anchor helper SHA differs")
        import sys

        sys.path.insert(0, str(args.source_repo / "src"))
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if torch.__version__ != config["torch_version"]:
            raise ValueError("Torch runtime differs")
        evaluator = load_module(args.value_helper, config["value_helper_sha256"]).NeuralValue(
            args.model
        )
        anchor_evaluator = load_module(
            args.anchor_helper, config["anchor_helper_sha256"]
        ).FrozenWDLAnchor(
            args.anchor_model,
            load_module(args.value_helper, config["value_helper_sha256"]),
        )
    state = None
    if args.resume is not None:
        if args.resume_sha256 is None or sha(args.resume) != args.resume_sha256:
            raise ValueError("resume artifact SHA differs")
        state = read(args.resume)

    def factory():
        return search.BudgetSearch(
            evaluator, nodes=512, quiescence_plies=2, max_depth=8, guard=guard
        )

    def guarded_anchor(board):
        guard()
        result = anchor_evaluator(board)
        guard()
        return result

    actor = Actor(config, factory, guarded_anchor, state)
    actor.advance(args.target_actions)
    guard()
    if not args.synthetic_zero_fixture and sha(args.model) != config["model_sha256"]:
        raise ValueError("model changed while snapshot was active")
    if not args.synthetic_zero_fixture and sha(args.anchor_model) != config["anchor_model_sha256"]:
        raise ValueError("frozen e8 anchor changed while snapshot was active")
    receipt = save(args.output, actor.state)
    print(
        json.dumps(
            {
                "schema": "fresh-qsearch-actor-segment-receipt-v2",
                "journal_sha256": receipt,
                "actions": actor.state["actions"],
                "completed_games": len(actor.state["games"]),
                "resume_sha256": args.resume_sha256,
                "original_deadline_epoch": args.original_deadline,
                "new_actor_checkpoint_only": True,
                "NN_qualified": False,
                "anchor_model_sha256": config["anchor_model_sha256"],
                "anchor_helper_sha256": config["anchor_helper_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
