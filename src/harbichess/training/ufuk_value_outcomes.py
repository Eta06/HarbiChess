"""Controlled late-outcome value experiment; native anchor and self data stay distinct."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

import chess
import torch
from torch.nn import functional as F

from harbichess.backends.torch_network import load_weights, sha256
from harbichess.chess.encoding import BoardEncoder
from harbichess.replay.shard import read_shard
from harbichess.training.config import LearnerConfig, NonFiniteTrainingError
from harbichess.training.oracle_data import prepare_rows, publish_json, read_game
from harbichess.training.oracle_train import evaluate
from harbichess.training.torch_checkpoint import load_checkpoint, save_checkpoint
from harbichess.training.torch_learner import TorchLearner, TorchTrainingBatch

VALUE_PREFIXES = (
    "value_",
    "invariant_value_",
    "material_value_",
    "global_value_",
    "plastic_value_",
)


def value_parameters(network) -> None:
    for name, parameter in network.named_parameters():
        parameter.requires_grad_(name.startswith(VALUE_PREFIXES))
    if not any(p.requires_grad for p in network.parameters()):
        raise ValueError("no value-specific parameters")


def late_games(records, *, plies: int = 32) -> list:
    if plies <= 0:
        raise ValueError("positive terminal window required")
    games = {}
    for record in records:
        games.setdefault(record.game_id, []).append(record)
    selected = []
    for game in games.values():
        game.sort(key=lambda record: record.ply)
        known = [r.outcome_value is not None for r in game]
        if any(known) != all(known):
            raise ValueError("mixed known/unknown outcome within a game")
        if all(known):
            selected.append(game[-plies:])
    return selected


def prepare(destination: Path, native: Path, replay: Path, book: Path, *, seed=20261016) -> dict:
    started = time.perf_counter()
    torch.set_num_threads(1)
    if destination.exists():
        raise FileExistsError(destination)
    families = json.loads(book.read_text())["splits"]["train"]
    allowed = {row["family"] for index, row in enumerate(families) if index % 4}
    records = [r for path in sorted(replay.glob("*.gz")) for r in read_shard(path).records]
    games = late_games(records)
    held = [g for g in games if g[0].game_index % 48 % 4 == 0]
    train = [g for g in games if g[0].game_index % 48 % 4 != 0]
    encoder = BoardEncoder()

    def encoded_games(groups):
        inputs, targets, metadata, indices = [], [], [], []
        for game in groups:
            board = chess.Board(game[0].root_fen)
            for uci in game[0].moves:
                board.push_uci(uci)
            group = []
            for index, record in enumerate(game):
                if index:
                    board.push_uci(record.moves[-1])
                assert board.ply() == record.ply
                record.validate_board(board)
                group.append(len(inputs))
                inputs.append(encoder.encode_board(board).values)
                targets.append([float(record.outcome_value == v) for v in (1, 0, -1)])
                metadata.append(
                    {
                        "game_id": record.game_id,
                        "game_index": record.game_index,
                        "ply": record.ply,
                        "position_key": " ".join(board.fen().split()[:4]),
                        "outcome_value": record.outcome_value,
                    }
                )
            indices.append(group)
        return inputs, targets, metadata, indices

    vi, vt, vm, vg = encoded_games(held)
    held_keys = {r["position_key"] for r in vm}
    ti, tt, tm, tg = encoded_games(train)
    keep = [i for i, r in enumerate(tm) if r["position_key"] not in held_keys]
    remap = {old: new for new, old in enumerate(keep)}
    tg = [[remap[i] for i in group if i in remap] for group in tg]
    tg = [g for g in tg if g]
    ti, tt, tm = ([values[i] for i in keep] for values in (ti, tt, tm))
    held_families = {r["game_index"] % 48 for r in vm}
    if len(ti) < 1000 or len(vi) < 400 or len(vg) < 16 or len(held_families) < 6:
        raise ValueError("preregistered own-outcome data qualification failed")
    manifest = json.loads((native / "dataset.json").read_text())
    anchor_rows, anchor_meta = [], []
    for name, digest in manifest["files"].items():
        path = native / name
        if sha256(path) != digest:
            raise ValueError("native input checksum mismatch")
        game = read_game(path)
        if game["job"]["split"] != "train" or game["job"]["family"] not in allowed:
            continue
        for index, row in enumerate(game["rows"]):
            if row["family"] != game["job"]["family"] or row["split"] != "train":
                raise ValueError("native family/split mismatch")
            if " ".join(row["fen"].split()[:4]) not in held_keys:
                anchor_rows.append(row)
                anchor_meta.append({"file": name, "row": index, "family": row["family"]})
    selected = sorted(
        random.Random(seed).sample(range(len(anchor_rows)), min(10000, len(anchor_rows)))
    )
    panel = prepare_rows([anchor_rows[i] for i in selected])
    tensors = {
        "native_inputs": panel.inputs,
        "native_wdl": panel.wdl,
        "own_inputs": torch.tensor(ti, dtype=torch.float32).reshape(-1, 8, 8, 104),
        "own_wdl": torch.tensor(tt, dtype=torch.float32),
        "held_inputs": torch.tensor(vi, dtype=torch.float32).reshape(-1, 8, 8, 104),
        "held_wdl": torch.tensor(vt, dtype=torch.float32),
        "own_groups": tg,
        "held_groups": vg,
    }
    if time.perf_counter() - started > 600:
        raise TimeoutError("input preparation budget exceeded")
    destination.mkdir(parents=True)
    torch.save(tensors, destination / "panel.pt")
    paths = [
        book,
        native / "dataset.json",
        *sorted(native.glob("*.json.gz")),
        *sorted(replay.glob("*.gz")),
    ]
    metadata = {
        "schema": 1,
        "seed": seed,
        "cache_sha256": sha256(destination / "panel.pt"),
        "semantics": (
            "Native STM soft reference versus observed terminal STM one-hot; "
            "neither unknown nor adjudicated outcomes."
        ),
        "inputs": {os.path.relpath(p.resolve(), destination.resolve()): sha256(p) for p in paths},
        "native_rows": len(selected),
        "native_families": sorted(allowed),
        "own_training_rows": len(ti),
        "own_training_games": len(tg),
        "held_rows": len(vi),
        "held_games": len(vg),
        "held_family_indices": sorted(held_families),
        "removed_own_training_overlap": sum(len(g) for g in train) - len(keep),
        "own_train_metadata": tm,
        "own_held_metadata": vm,
        "native_selection": [anchor_meta[i] for i in selected],
        "preparation_wall_seconds": time.perf_counter() - started,
    }
    publish_json(destination / "panel.json", metadata)
    if time.perf_counter() - started > 600:
        raise TimeoutError("complete input preparation budget exceeded")
    return {
        k: v
        for k, v in metadata.items()
        if k not in {"inputs", "own_train_metadata", "own_held_metadata", "native_selection"}
    }


def sample_indices(panel: dict, *, half_batch=32):
    native = torch.randint(len(panel["native_inputs"]), (half_batch,)).tolist()
    groups = panel["own_groups"]
    own = [
        group[int(torch.randint(len(group), ()).item())]
        for group in (groups[i] for i in torch.randint(len(groups), (half_batch,)).tolist())
    ]
    return native, own


def value_step(learner, native_inputs, native_wdl, own_inputs, own_wdl, *, include_self):
    learner.optimizer.zero_grad(set_to_none=True)
    native = F.cross_entropy(learner.network(native_inputs)[1], native_wdl)
    own = F.cross_entropy(learner.network(own_inputs)[1], own_wdl)
    loss = 0.5 * native + (0.5 if include_self else 0.0) * own
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(learner.network.parameters(), 5.0)
    if not all(math.isfinite(float(x.detach())) for x in (loss, native, own, norm)):
        raise NonFiniteTrainingError("nonfinite calibration loss/gradient")
    if any(
        p.grad is not None and not torch.isfinite(p.grad).all()
        for p in learner.network.parameters()
    ):
        raise NonFiniteTrainingError("nonfinite calibration gradient")
    learner.optimizer.step()
    if any(not torch.isfinite(p).all() for p in learner.network.parameters()):
        raise NonFiniteTrainingError("nonfinite calibration update")
    learner.step += 1
    return {
        "step": learner.step,
        "loss": float(loss.detach()),
        "native_ce": float(native.detach()),
        "own_ce": float(own.detach()),
        "unclipped_gradient_norm": float(norm),
    }


def held_value(network, panel):
    losses = []
    with torch.no_grad():
        for start in range(0, len(panel["held_inputs"]), 64):
            logits = network(panel["held_inputs"][start : start + 64])[1]
            losses.extend(
                F.cross_entropy(
                    logits, panel["held_wdl"][start : start + 64], reduction="none"
                ).tolist()
            )
    means = [sum(losses[i] for i in group) / len(group) for group in panel["held_groups"]]
    return {
        "game_balanced_ce": sum(means) / len(means),
        "game_ce": means,
        "games": len(means),
        "rows": len(losses),
    }


def train(
    directory,
    cache,
    observer,
    weights,
    *,
    arm,
    resume=None,
    stop_at=None,
    max_steps=4000,
    interval=250,
    patience=1000,
    wall_seconds=1800,
    seed=20261016,
):
    if arm not in {"anchor", "self"} or min(max_steps, interval, patience, wall_seconds) <= 0:
        raise ValueError("invalid calibration arm/budget")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    meta = json.loads((cache / "panel.json").read_text())
    if meta["schema"] != 1 or meta["seed"] != seed:
        raise ValueError("value input schema/seed mismatch")
    if sha256(cache / "panel.pt") != meta["cache_sha256"]:
        raise ValueError("cache checksum mismatch")
    for relative, digest in meta["inputs"].items():
        if sha256(cache / relative) != digest:
            raise ValueError("source input checksum mismatch")
    panel = torch.load(cache / "panel.pt", weights_only=True)
    observer_meta = json.loads(observer.with_suffix(".json").read_text())
    if sha256(observer) != observer_meta["cache_sha256"]:
        raise ValueError("observer checksum mismatch")
    native_cache = torch.load(observer, weights_only=True)
    native = TorchTrainingBatch(**native_cache["tensors"])
    base = load_weights(weights)
    frozen = {
        k: v.clone() for k, v in base.state_dict().items() if not k.startswith(VALUE_PREFIXES)
    }
    config = {
        "task": "late-outcome-value-calibration",
        "schema": 1,
        "arm": arm,
        "initial_weights_sha256": sha256(weights),
        "cache_sha256": meta["cache_sha256"],
        "cache_metadata_sha256": sha256(cache / "panel.json"),
        "observer_sha256": observer_meta["cache_sha256"],
        "seed": seed,
        "max_steps": max_steps,
        "interval": interval,
        "patience": patience,
        "half_batch": 32,
        "native_loss_weight": 0.5,
        "self_loss_weight": 0.5 if arm == "self" else 0.0,
        "learning_rate": 5e-5,
        "sampling": "Torch global RNG: native uniform; own game-balanced",
        "code_sha256": {
            path.name: sha256(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("torch_learner.py"),
                Path(__file__).with_name("torch_checkpoint.py"),
                Path(__file__).with_name("oracle_train.py"),
                Path(__file__).parents[1] / "backends/torch_network.py",
            )
        },
    }
    trace = directory / "sampling.jsonl"
    if resume:
        learner, manifest, _ = load_checkpoint(resume, expected_run_config=config)
        state = manifest["run_state"]
        if (
            manifest["step"] != state["cursor"]
            or resume.parent.parent.resolve() != directory.resolve()
        ):
            raise ValueError("resume cursor/directory mismatch")
        lines = trace.read_text().splitlines()
        digest = "0" * 64
        for i, line in enumerate(lines, 1):
            event = json.loads(line)
            if event["step"] != i:
                raise ValueError("sample trace chronology mismatch")
            digest = hashlib.sha256((digest + line).encode()).hexdigest()
        if len(lines) != learner.step or digest != state["sample_trace_sha256"]:
            raise ValueError("resume requires exact saved sample trace prefix")
    else:
        if directory.exists():
            raise FileExistsError(directory)
        directory.mkdir(parents=True)
        torch.manual_seed(seed)
        value_parameters(base)
        learner = TorchLearner(base, config=LearnerConfig(learning_rate=5e-5, policy_weight=0))
        state = {
            "cursor": 0,
            "best_step": 0,
            "best_ce": math.inf,
            "evaluations": [],
            "sample_trace_sha256": "0" * 64,
            "initial_native": evaluate(learner, native),
            "initial_held": held_value(learner.network, panel),
        }
        trace.touch()
        publish_json(
            directory / "metadata.json",
            {
                "source_commit": source,
                "config": config,
                "trainable_parameters": sum(
                    p.numel() for p in base.parameters() if p.requires_grad
                ),
                "frozen_parameters": sum(v.numel() for v in frozen.values()),
            },
        )

    def inspect():
        changed = [
            k for k, v in frozen.items() if not torch.equal(v, learner.network.state_dict()[k])
        ]
        metrics = evaluate(learner, native)
        held = held_value(learner.network, panel)
        if changed or metrics["policy_ce"] != state["initial_native"]["policy_ce"]:
            raise ValueError("frozen policy changed")
        valid = (
            metrics["value_ce"] <= state["initial_native"]["value_ce"] + 0.03
            and metrics["value_mae"] <= state["initial_native"]["value_mae"] + 0.02
        )
        state["evaluations"].append(
            {
                "step": learner.step,
                "native": metrics,
                "held": held,
                "retention_passed": valid,
                "sample_trace_sha256": state["sample_trace_sha256"],
                "torch_rng_sha256": hashlib.sha256(
                    torch.get_rng_state().numpy().tobytes()
                ).hexdigest(),
            }
        )
        if valid and held["game_balanced_ce"] < state["best_ce"]:
            state["best_step"], state["best_ce"] = learner.step, held["game_balanced_ce"]
        state["cursor"] = learner.step
        save_checkpoint(
            directory / "checkpoints" / f"step-{learner.step:06d}",
            learner=learner,
            sampler=None,
            replay_paths=(cache / "panel.pt", cache / "panel.json"),
            run_state=state,
            run_config=config,
            source_commit=source,
        )
        return valid

    valid = True if resume else inspect()
    reason = "max_steps"
    with trace.open("a") as stream:
        while valid and learner.step < max_steps:
            if time.perf_counter() - started >= wall_seconds:
                reason = "wall budget"
                break
            indices, own = sample_indices(panel)
            metrics = value_step(
                learner,
                panel["native_inputs"][indices],
                panel["native_wdl"][indices],
                panel["own_inputs"][own],
                panel["own_wdl"][own],
                include_self=arm == "self",
            )
            line = json.dumps(
                {"step": learner.step, "native": indices, "own": own}, separators=(",", ":")
            )
            stream.write(line + "\n")
            state["sample_trace_sha256"] = hashlib.sha256(
                (state["sample_trace_sha256"] + line).encode()
            ).hexdigest()
            if learner.step % interval == 0 or learner.step == max_steps:
                stream.flush()
                valid = inspect()
                print(
                    json.dumps(
                        {
                            "arm": arm,
                            "step": learner.step,
                            "metrics": metrics,
                            "held_ce": state["evaluations"][-1]["held"]["game_balanced_ce"],
                            "retention_passed": valid,
                        }
                    ),
                    flush=True,
                )
                if not valid:
                    reason = "native retention failed"
                    break
                if stop_at is not None and learner.step >= stop_at:
                    reason = "explicit checkpoint pause"
                    break
                if learner.step - state["best_step"] >= patience:
                    reason = "heldout patience"
                    break
    if learner.step != state["cursor"]:
        valid = inspect()
    report = {
        "status": "failed"
        if not valid
        else ("incomplete" if reason == "wall budget" else "completed"),
        "stop_reason": reason,
        "source_commit": source,
        "state": state,
        "wall_seconds": time.perf_counter() - started,
        "best_checkpoint": f"checkpoints/step-{state['best_step']:06d}",
        "config": config,
        "new_paid_resources": False,
        "promotion_ready": False,
    }
    publish_json(directory / "result.json", report)
    return {k: v for k, v in report.items() if k not in {"state", "config"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    prep = modes.add_parser("prepare")
    for name in ("destination", "native", "replay", "book"):
        prep.add_argument(name, type=Path)
    fit = modes.add_parser("train")
    for name in ("directory", "cache", "observer", "weights"):
        fit.add_argument(name, type=Path)
    fit.add_argument("--arm", choices=("anchor", "self"), required=True)
    fit.add_argument("--resume", type=Path)
    fit.add_argument("--stop-at", type=int)
    args = vars(parser.parse_args())
    mode = args.pop("mode")
    print(json.dumps(prepare(**args) if mode == "prepare" else train(**args)), flush=True)


if __name__ == "__main__":
    main()
