"""Root-invoked prospective clean-source actual CPU CLI qualification, never automatic."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


FILES = (
    "model.safetensors",
    "base.safetensors",
    "behavior.safetensors",
    "training.pt",
    "actor.json",
    "last-frozen-epoch.json.gz",
)
LEDGER = "pre-action-greedy-mixture-search-behavior-v5"
END = float("inf")  # new CPU-only control; historical campaign clocks remain unchanged


def publish(path, value):
    data = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    descriptor, name = tempfile.mkstemp(prefix=".receipt-", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
        descriptor = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temp.unlink(missing_ok=True)


def check_source(checkout, source):
    if not re.fullmatch(r"[0-9a-f]{40}", source):
        raise ValueError("Requires explicit frozen source commit")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=checkout, text=True
    ).strip()
    if head != source or dirty:
        raise ValueError("Requires exact clean producer checkout")


def expected_artifacts():
    return [f"journal/epoch-{i:08d}.json.gz" for i in (1, 2)] + [
        f"checkpoints/epoch-{i:08d}/{name}" for i in (0, 1, 2) for name in FILES
    ]


def validate_ledger(record, expected):
    if expected != LEDGER or record["own_search"]["schema"] != expected:
        raise ValueError(
            "Explicit greedy-mixture certificate ledger v5 required; "
            "legacy acting ledgers are not aligned-policy proof"
        )
    roots = record["own_search"]["roots"]
    if not roots or any(type(r.get("certified_mates")) is not list for r in roots):
        raise ValueError("Certificate inventory required for EVERY searched root")


def semantic_native(manifest):
    value = dict(manifest)
    value["inputs"] = {
        key: {"sha256": row["sha256"]} for key, row in value["inputs"].items()
    }
    return value


def run_owned(command, **kwargs):
    deadline = kwargs.pop("deadline")
    if time.time() >= deadline:
        raise TimeoutError("Original whole deadline exhausted")
    process = subprocess.Popen(command, start_new_session=True, **kwargs)
    try:
        while process.poll() is None:
            if time.time() >= deadline:
                raise TimeoutError("Original whole deadline exhausted")
            time.sleep(min(0.2, deadline - time.time()))
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkout", "output", "weights", "book", "config", "protocol"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--started-epoch", type=float, required=True)
    parser.add_argument("--memory-max-bytes", type=int, required=True)
    parser.add_argument("--disk-min-free-bytes", type=int, required=True)
    parser.add_argument("--expected-ledger-schema", required=True, choices=[LEDGER])
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    check_source(args.checkout, args.source_commit)
    if (
        args.started_epoch > time.time()
        or args.deadline_epoch != args.started_epoch + 600
    ):
        raise ValueError(
            "Original observed first clock plus600 required; no reset/futurefirst"
        )
    if args.deadline_epoch >= END:
        raise ValueError("Actual06UTC hard ceiling exceeded")
    if args.deadline_epoch <= time.time() or args.deadline_epoch - time.time() > 600:
        raise ValueError(
            "qualification requires absolute shared deadline within600seconds"
        )
    cfg = json.loads(args.config.read_text())
    if cfg["device"] != "cpu" or cfg["actors"]["temperature"] != 1:
        raise ValueError("qualification is actualCPU/T1 only")
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
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "preflight-only-no-CPU-executed",
                    "source_commit": args.source_commit,
                    "expected_ledger_schema": args.expected_ledger_schema,
                    "comparison_artifacts": expected_artifacts(),
                    "deadline_epoch": args.deadline_epoch,
                }
            )
        )
        return
    args.output.mkdir(parents=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(args.checkout.resolve() / "src")
    env.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )
    shared = [
        sys.executable,
        "-m",
        "harbichess.training.torch_search_acting_run",
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
        "--memory-policy",
        "inactive-file-v1",
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

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Owned source8 qualification interrupted {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    started = time.time()
    for name, run, extra in invocations:
        if time.time() >= args.deadline_epoch:
            raise TimeoutError("shared qualification deadline exhausted")
        with (
            (args.output / f"{name}.stdout.log").open("xb") as out,
            (args.output / f"{name}.stderr.log").open("xb") as err,
        ):
            run_owned(
                [*shared, str(run), *extra],
                cwd=args.checkout,
                env=env,
                stdout=out,
                stderr=err,
                deadline=args.deadline_epoch,
            )
    paths = expected_artifacts()
    hashes = {}
    for name in paths:
        whole, split = args.output / "whole" / name, args.output / "split" / name
        if whole.read_bytes() != split.read_bytes():
            raise ValueError(f"actualCPU ownsearch freshprocess bytes differ: {name}")
        hashes[name] = sha(whole)
    state = json.loads(
        (args.output / "whole/checkpoints/epoch-00000002/actor.json").read_text()
    )
    if state["optimizer_accepted_updates"] <= 0:
        raise ValueError(
            "actualCPU qualification collected no accepted supervised updates"
        )
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
            "CPU qualification requires BOTH own-search targets and "
            "fresh terminal targets plus UNKNOWN rows"
        )
    if initial != {name: sha(getattr(args, name)) for name in initial}:
        raise ValueError("qualification immutable inputs changed")
    native_manifest_hashes = {}
    for epoch in (0, 1, 2):
        pair = []
        for name in ("whole", "split"):
            path = args.output / name / f"checkpoints/epoch-{epoch:08d}/checkpoint.json"
            native = json.loads(path.read_text())
            if (
                native["schema"] != "torch-search-acting-native-cpu-v3"
                or native["source_commit"] != args.source_commit
            ):
                raise ValueError("ActualCPU native schema/source differs")
            if native["state"]["epoch"] != epoch or set(native["artifacts"]) != set(
                FILES
            ):
                raise ValueError("Native full payload/state differs")
            for filename, digest in native["artifacts"].items():
                if sha(path.parent / filename) != digest:
                    raise ValueError("Native manifest artifact hash differs")
            native_manifest_hashes[f"{name}/epoch-{epoch:08d}"] = sha(path)
            pair.append(semantic_native(native))
        if pair[0] != pair[1]:
            raise ValueError("Native metadata/state/runtime/input semantic mismatch")
    for epoch in (1, 2):
        record = json.loads(
            gzip.decompress(
                (args.output / f"whole/journal/epoch-{epoch:08d}.json.gz").read_bytes()
            )
        )
        validate_ledger(record, args.expected_ledger_schema)
    # Independent fresh process strictly opens BOTH final full-native containers.
    verify_code = """
import json,sys
from pathlib import Path
from harbichess.selfplay.online_actor import OnlineActorConfig
from harbichess.training.torch_fullgame_ppo import FullGamePPOTrainConfig
from harbichess.training.torch_ownsearch_core import OwnSearchObjective
from harbichess.training.search_acting_policy import search_config_from_dict
from harbichess.training.torch_search_acting_learner import (
 TorchSearchActingConfig,TorchSearchActingLearner
)
c=json.loads(Path(sys.argv[1]).read_text())
c['actors']=OnlineActorConfig(**c['actors'])
c['objective']=OwnSearchObjective(**c['objective'])
c['search']=search_config_from_dict(c['search'])
c['schedule']=FullGamePPOTrainConfig(**c['schedule'])
paths=dict(initial_weights=Path(sys.argv[2]),book=Path(sys.argv[3]),experiment_config=Path(sys.argv[1]),protocol=Path(sys.argv[4]))
import torch
assert torch.__version__ == '2.14.1+cpu' and not torch.cuda.is_available()
for name in ('whole','split'):
 for epoch in (0,1,2):
  l=TorchSearchActingLearner.resume(Path(sys.argv[5])/name/'checkpoints'/f'epoch-{epoch:08d}',config=TorchSearchActingConfig(**c),input_paths=paths,source_commit=sys.argv[6])
  assert l.epoch==epoch and l.closed
  if epoch==2: assert l.optimizer_accepted_updates>0
print('both final CPU fullnative containers strictly loaded in fresh process')
"""
    with (
        (args.output / "final-native-audit.stdout.log").open("xb") as out,
        (args.output / "final-native-audit.stderr.log").open("xb") as err,
    ):
        run_owned(
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
            deadline=args.deadline_epoch,
        )
    if time.time() >= args.deadline_epoch:
        raise TimeoutError(
            "shared qualification deadline exhausted during native audit"
        )
    receipt = {
        "schema": "actual-CPU-greedy-mixture-CLI-qualification-v1",
        "expected_ledger_schema": args.expected_ledger_schema,
        "native_manifest_sha256": native_manifest_hashes,
        "native_manifest_comparison": (
            "exact semantic state/runtime/source/input/artifacts "
            "excluding run-relative input paths only"
        ),
        "both_all_native_epochs_freshprocess_strictload": [0, 1, 2],
        "status": "pass",
        "source_commit": args.source_commit,
        "input_sha256": initial,
        "byte_exact_artifacts": hashes,
        "both_final_full_native_freshprocess_strictload": True,
        "absolute_deadline_epoch": args.deadline_epoch,
        "started_epoch": started,
        "original_started_epoch": args.started_epoch,
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
        "scope": (
            "NEW producer infrastructure only; old428 proof is separate ancestor, "
            "not source8 evidence. Certificate inventory semantics require "
            "independent E1 audit."
        ),
    }
    check_source(args.checkout, args.source_commit)
    publish(args.output / "result.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        if "--execute" in sys.argv and "--output" in sys.argv:
            output = Path(sys.argv[sys.argv.index("--output") + 1])
            if output.is_dir() and not (output / "result.json").exists():
                publish(
                    output / "result.json",
                    {
                        "schema": "actual-CPU-greedy-mixture-CLI-qualification-v1",
                        "status": "failed-preserved",
                        "error": repr(exc),
                        "finished_epoch": time.time(),
                        "original_argv": sys.argv,
                    },
                )
        raise
