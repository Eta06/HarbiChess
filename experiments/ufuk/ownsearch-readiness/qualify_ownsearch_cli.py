"""Root-invoked prospective clean-source actual CUDA CLI qualification, never automatic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkout", "output", "weights", "book", "config", "protocol"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--memory-max-bytes", type=int, required=True)
    parser.add_argument("--disk-min-free-bytes", type=int, required=True)
    args = parser.parse_args()
    if args.deadline_epoch <= time.time() or args.deadline_epoch - time.time() > 180:
        raise ValueError(
            "qualification requires absolute shared deadline within180seconds"
        )
    cfg = json.loads(args.config.read_text())
    if cfg["device"] != "cuda:0" or cfg["actors"]["temperature"] != 1:
        raise ValueError("qualification is actualCUDA/T1 only")
    if args.output.exists():
        raise FileExistsError(args.output)
    initial = {
        name: sha(getattr(args, name))
        for name in ("weights", "book", "config", "protocol")
    }
    if (
        sha(args.weights)
        != "e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03"
    ):
        raise ValueError(
            "actual sparse-ownsearch qualification requires immutable real e8"
        )
    args.output.mkdir(parents=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(args.checkout.resolve() / "src")
    shared = [
        sys.executable,
        "-m",
        "harbichess.training.torch_ownsearch_run",
        "--weights",
        str(args.weights.resolve()),
        "--book",
        str(args.book.resolve()),
        "--config",
        str(args.config.resolve()),
        "--protocol",
        str(args.protocol.resolve()),
        "--source-commit",
        args.source_commit,
        "--max-epochs",
        "2",
        "--checkpoint-interval",
        "1",
        "--deadline-epoch",
        str(args.deadline_epoch),
        "--memory-max-bytes",
        str(args.memory_max_bytes),
        "--disk-min-free-bytes",
        str(args.disk_min_free_bytes),
    ]
    invocations = [
        ("whole2", args.output / "whole", []),
        ("pause1", args.output / "split", ["--stop-at", "1"]),
        (
            "resume2",
            args.output / "split",
            ["--resume", str(args.output / "split/checkpoints/epoch-00000001")],
        ),
    ]
    started = time.time()
    for name, run, extra in invocations:
        if time.time() >= args.deadline_epoch:
            raise TimeoutError("shared qualification deadline exhausted")
        with (
            (args.output / f"{name}.stdout.log").open("xb") as out,
            (args.output / f"{name}.stderr.log").open("xb") as err,
        ):
            subprocess.run(
                [*shared, str(run), *extra],
                cwd=args.checkout,
                env=env,
                stdout=out,
                stderr=err,
                check=True,
                timeout=max(1, args.deadline_epoch - time.time()),
            )
    paths = [f"journal/epoch-{i:08d}.json.gz" for i in (1, 2)]
    files = (
        "model.safetensors",
        "base.safetensors",
        "behavior.safetensors",
        "training.pt",
        "actor.json",
        "last-frozen-epoch.json.gz",
    )
    paths += [f"checkpoints/epoch-00000002/{name}" for name in files]
    hashes = {}
    for name in paths:
        whole, split = args.output / "whole" / name, args.output / "split" / name
        if whole.read_bytes() != split.read_bytes():
            raise ValueError(f"actualCUDA ownsearch freshprocess bytes differ: {name}")
        hashes[name] = sha(whole)
    state = json.loads(
        (args.output / "whole/checkpoints/epoch-00000002/actor.json").read_text()
    )
    if state["optimizer_accepted_updates"] <= 0:
        raise ValueError(
            "actualCUDA qualification collected no accepted supervised updates"
        )
    import gzip

    final_record = json.loads(
        gzip.decompress(
            (args.output / "whole/journal/epoch-00000002.json.gz").read_bytes()
        )
    )
    if (
        final_record["training"]["policy_target_rows"] <= 0
        or final_record["training"]["trained_transitions"] <= 0
        or final_record["target_counts"]["complete_games"] <= 0
        or final_record["target_counts"]["excluded_actions"] <= 0
    ):
        raise ValueError(
            "CUDA qualification requires BOTH own-search targets and "
            "fresh terminal targets plus UNKNOWN rows"
        )
    if initial != {name: sha(getattr(args, name)) for name in initial}:
        raise ValueError("qualification immutable inputs changed")
    for name in ("whole", "split"):
        native = json.loads(
            (
                args.output / name / "checkpoints/epoch-00000002/checkpoint.json"
            ).read_text()
        )
        if native["schema"] != "torch-ownsearch-native-cuda-v1":
            raise ValueError("qualification native is not actualCUDA ownsearch schema")
    # Independent fresh process strictly opens BOTH final full-native containers.
    verify_code = """
import json,sys
from pathlib import Path
from harbichess.selfplay.online_actor import OnlineActorConfig
from harbichess.training.torch_fullgame_ppo import FullGamePPOTrainConfig
from harbichess.training.torch_ownsearch_core import OwnSearchObjective
from harbichess.training.ownsearch_targets import OwnSearchConfig
from harbichess.training.torch_ownsearch_learner import TorchOwnSearchConfig,TorchOwnSearchLearner
c=json.loads(Path(sys.argv[1]).read_text())
c['actors']=OnlineActorConfig(**c['actors'])
c['objective']=OwnSearchObjective(**c['objective'])
c['search']=OwnSearchConfig(**c['search'])
c['schedule']=FullGamePPOTrainConfig(**c['schedule'])
paths=dict(initial_weights=Path(sys.argv[2]),book=Path(sys.argv[3]),experiment_config=Path(sys.argv[1]),protocol=Path(sys.argv[4]))
for name in ('whole','split'):
 l=TorchOwnSearchLearner.resume(Path(sys.argv[5])/name/'checkpoints/epoch-00000002',config=TorchOwnSearchConfig(**c),input_paths=paths,source_commit=sys.argv[6])
 assert l.epoch==2 and l.closed and l.optimizer_accepted_updates>0
print('both final CUDA fullnative containers strictly loaded in fresh process')
"""
    with (
        (args.output / "final-native-audit.stdout.log").open("xb") as out,
        (args.output / "final-native-audit.stderr.log").open("xb") as err,
    ):
        subprocess.run(
            [
                sys.executable,
                "-c",
                verify_code,
                str(args.config.resolve()),
                str(args.weights.resolve()),
                str(args.book.resolve()),
                str(args.protocol.resolve()),
                str(args.output.resolve()),
                args.source_commit,
            ],
            cwd=args.checkout,
            env=env,
            stdout=out,
            stderr=err,
            check=True,
            timeout=max(1, args.deadline_epoch - time.time()),
        )
    if time.time() >= args.deadline_epoch:
        raise TimeoutError(
            "shared qualification deadline exhausted during native audit"
        )
    receipt = {
        "schema": "actual-CUDA-sparse-ownsearch-CLI-qualification-v1",
        "status": "pass",
        "source_commit": args.source_commit,
        "input_sha256": initial,
        "byte_exact_artifacts": hashes,
        "both_final_full_native_freshprocess_strictload": True,
        "absolute_deadline_epoch": args.deadline_epoch,
        "started_epoch": started,
        "finished_epoch": time.time(),
        "per_run_epochs": 2,
        "last_epoch_policy_target_rows": final_record["training"]["policy_target_rows"],
        "last_epoch_known_terminal_rows": final_record["training"][
            "trained_transitions"
        ],
        "last_epoch_UNKNOWN_excluded_value_rows": final_record["target_counts"][
            "excluded_actions"
        ],
        "audit_optimizer_attempted_updates": state["optimizer_attempted_updates"] * 2,
        "unique_fresh_transitions": cfg["epoch_steps"] * cfg["actors"]["games"] * 2,
        "duplicate_qualification_fresh_transitions": cfg["epoch_steps"]
        * cfg["actors"]["games"]
        * 2,
        "scope": "Infrastructure replay qualification; excluded from production/strength claims.",
    }
    (args.output / "result.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
