"""Bounded fresh self-play policy iteration; generation-boundary exact resume.

Actors share an inference snapshot through a batching thread, or explicitly use
spawned CPU processes. Learning occurs only after actors drain, without forks.
Partial generations remain diagnostic artifacts and restart from the last commit.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import chess
import torch

from harbichess.backends.torch_network import TorchChessNetwork, load_weights, sha256
from harbichess.chess.rules import PythonChessRules
from harbichess.core.network_config import NetworkConfig
from harbichess.replay.schema import records_from_game
from harbichess.replay.shard import ShardMetadata, read_shard, write_shard_atomic
from harbichess.replay.split import ReplaySplit
from harbichess.selfplay.game import derive_game_seed
from harbichess.selfplay.torch_actors import collect_process_games, collect_thread_games
from harbichess.training.batch import GameBalancedSampler, build_training_batch
from harbichess.training.config import LearnerConfig
from harbichess.training.torch_checkpoint import load_checkpoint, save_checkpoint
from harbichess.training.torch_learner import TorchLearner


@dataclass(frozen=True)
class LoopConfig:
    seed: int = 20261002
    games: int = 8
    simulations: int = 16
    max_plies: int = 192
    steps: int = 32
    batch_size: int = 32
    replay_generations: int = 3
    workers: int = 4
    threads: int = 1
    learning_rate: float = 2e-4
    device: str = "cpu"
    gumbel_scale: float = 1.0

    def __post_init__(self) -> None:
        counts = (
            self.games,
            self.simulations,
            self.max_plies,
            self.steps,
            self.batch_size,
            self.replay_generations,
            self.workers,
            self.threads,
        )
        if min(counts) <= 0 or self.games < 2 or self.seed < 0 or self.gumbel_scale < 0:
            raise ValueError("loop counts must be positive, games >= 2, seed/scales non-negative")
        LearnerConfig(learning_rate=self.learning_rate)


def run_loop(
    directory: Path,
    *,
    config: LoopConfig,
    generations: int,
    wall_seconds: float = 1800,
    weights: Path | None = None,
    legacy_mlx: bool = False,
    resume: Path | None = None,
    actor_mode: str = "thread",
    trainable_prefixes: tuple[str, ...] = (),
    opening_book: Path | None = None,
) -> dict:
    if generations <= 0 or wall_seconds <= 0 or actor_mode not in ("thread", "process"):
        raise ValueError("generation target and wall budget must be positive")
    if resume and weights:
        raise ValueError("resume and weights warm-start are mutually exclusive")
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    started = time.perf_counter()
    cpu_started = time.process_time()
    children_started = resource.getrusage(resource.RUSAGE_CHILDREN)
    deadline = started + wall_seconds
    run_config = asdict(config)
    if trainable_prefixes:
        if any(not isinstance(prefix, str) or not prefix for prefix in trainable_prefixes):
            raise ValueError("trainable prefixes must be nonempty strings")
        run_config["trainable_prefixes"] = list(trainable_prefixes)
    if opening_book is not None:
        book = json.loads(opening_book.read_text())
        if book["schema"] != 1:
            raise ValueError("unsupported opening book schema")
        openings = [row["opening"]["moves"] for row in book["splits"]["train"]]
        if not openings:
            raise ValueError("training opening book is empty")
        for moves in openings:
            board = chess.Board()
            for uci in moves:
                board.push_uci(uci)
            if board.outcome(claim_draw=True) is not None or len(moves) >= config.max_plies:
                raise ValueError("training opening must be nonterminal within ply budget")
        run_config["actor_openings"] = openings
        run_config["opening_source_sha256"] = sha256(opening_book)
    if actor_mode == "process":
        if config.device != "cpu" or config.threads != 1:
            raise ValueError("process actors require CPU and one Torch thread")
        run_config["actor_mode"] = actor_mode
    sampler = None
    rules = PythonChessRules()
    if resume:
        if directory.resolve() != resume.parent.parent.resolve():
            raise ValueError("resume checkpoint must belong to this run directory")
        learner, manifest, _ = load_checkpoint(
            resume, expected_run_config=run_config, device=config.device
        )
        state = manifest["run_state"]
        active_paths = tuple((resume / p).resolve() for p in manifest["replay"])
        # Migrate early v1 cursors using immutable replay provenance after relocation.
        if "run_id" not in state:
            names = {read_shard(p).header.run_id for p in active_paths}
            if len(names) > 1:
                raise ValueError("resume replay has inconsistent run identities")
            state["run_id"] = next(iter(names), directory.name)
    else:
        directory.mkdir(parents=True, exist_ok=False)
        torch.manual_seed(config.seed)
        network = (
            load_weights(weights, legacy_mlx=legacy_mlx)
            if weights
            else TorchChessNetwork(
                NetworkConfig(
                    trunk_channels=16,
                    residual_blocks=2,
                    policy_channels=4,
                    value_channels=2,
                    value_hidden=32,
                )
            )
        )
        if trainable_prefixes:
            for name, parameter in network.named_parameters():
                parameter.requires_grad_(name.startswith(trainable_prefixes))
            if not any(parameter.requires_grad for parameter in network.parameters()):
                raise ValueError("trainable prefixes select no network parameters")
        learner = TorchLearner(
            network, config=LearnerConfig(learning_rate=config.learning_rate), device=config.device
        )
        state = {
            "run_id": directory.name,
            "generation": 0,
            "next_game": 0,
            "history": [],
            "initial_weights_sha256": sha256(weights) if weights else None,
            "initial_transfer": "weights-only" if weights else "random-init",
        }
        active_paths = ()
        save_checkpoint(
            directory / "checkpoints/generation-000000",
            learner=learner,
            sampler=None,
            replay_paths=(),
            run_state=state,
            run_config=run_config,
            source_commit=source,
        )
    status = "completed"
    reason = "generation target reached"
    try:
        while state["generation"] < generations:
            generation = state["generation"] + 1
            generation_start = time.perf_counter()
            if generation_start >= deadline:
                raise TimeoutError("wall-clock budget exhausted")
            indices = range(state["next_game"], state["next_game"] + config.games)
            if actor_mode == "process":
                games, stats = collect_process_games(
                    directory
                    / f"checkpoints/generation-{generation - 1:06d}"
                    / "model.safetensors",
                    run_config,
                    indices,
                    deadline,
                )
            else:
                games, stats = collect_thread_games(
                    learner.network, run_config, indices, deadline
                )
            generation_seconds = time.perf_counter() - generation_start
            fresh = {
                split: tuple(
                    r
                    for g in games[:-1]
                    if split == ReplaySplit.TRAIN
                    for r in records_from_game(g, run_id=state["run_id"], rules=rules)
                )
                for split in (ReplaySplit.TRAIN,)
            }
            fresh[ReplaySplit.VALIDATION] = records_from_game(
                games[-1], run_id=state["run_id"], rules=rules
            )
            paths = []
            snapshot_sha = sha256(
                directory / f"checkpoints/generation-{generation - 1:06d}" / "model.safetensors"
            )
            for split, records in fresh.items():
                path = directory / f"replay/generation-{generation:06d}-{split}.jsonl.gz"
                # Recompute an interrupted generation safely; earlier bytes must agree.
                if path.exists():
                    previous = read_shard(path)
                    if (
                        previous.records != records
                        or previous.header.source_checkpoint != snapshot_sha
                    ):
                        raise ValueError(
                            "uncommitted replay differs from deterministic regeneration"
                        )
                else:
                    write_shard_atomic(
                        path,
                        records,
                        ShardMetadata(
                            state["run_id"],
                            generation,
                            snapshot_sha,
                            source,
                            datetime.now(UTC).isoformat(),
                            split,
                        ),
                    )
                paths.append(path.resolve())
            active_paths = tuple(
                p
                for p in (*active_paths, *paths)
                if read_shard(p).header.generation > generation - config.replay_generations
            )
            train_records = tuple(
                r
                for p in active_paths
                if p.name.endswith("-train.jsonl.gz")
                for r in read_shard(p).records
            )
            validation_records = tuple(
                r
                for p in active_paths
                if p.name.endswith("-validation.jsonl.gz")
                for r in read_shard(p).records
            )
            sampler = GameBalancedSampler(
                train_records, seed=derive_game_seed(config.seed, generation)
            )
            # Bounded window is materialized once, not reconstructed at each learner step.
            train = learner.prepare_batch(build_training_batch(train_records, rules=rules))
            validation = learner.prepare_batch(
                build_training_batch(validation_records, rules=rules)
            )
            fixed_indices = tuple(range(min(config.batch_size, train.size)))
            fixed = train.select(fixed_indices)
            before, held_before = learner.evaluate_loss(fixed), learner.evaluate_loss(validation)
            training_start = time.perf_counter()
            metrics = []
            for _ in range(config.steps):
                if time.perf_counter() >= deadline:
                    raise TimeoutError("wall-clock budget exhausted during training")
                metrics.append(
                    asdict(
                        learner.train_step(train.select(sampler.sample_indices(config.batch_size)))
                    )
                )
            training_seconds = time.perf_counter() - training_start
            record = {
                "generation": generation,
                "generated_games": len(games),
                "generated_positions": sum(len(g.samples) for g in games),
                "terminal_observed_games": sum(g.outcome.termination != "max_plies" for g in games),
                "terminal_observed_rows": sum(
                    r.outcome_value is not None for records in fresh.values() for r in records
                ),
                "terminations": [g.outcome.termination for g in games],
                "selfplay_seconds": generation_seconds,
                "inference_statistics": stats,
                "inference_positions_per_wall_second": stats["positions"] / generation_seconds,
                "training_seconds": training_seconds,
                "sampled_training_rows": config.steps * config.batch_size,
                "sampled_training_rows_per_second": config.steps
                * config.batch_size
                / training_seconds,
                "rolling_train_rows": len(train_records),
                "rolling_validation_rows": len(validation_records),
                "fixed_loss_before": before,
                "fixed_loss_after": learner.evaluate_loss(fixed),
                "heldout_loss_before": held_before,
                "heldout_loss_after": learner.evaluate_loss(validation),
                "metrics": metrics,
            }
            state = {
                **state,
                "generation": generation,
                "next_game": state["next_game"] + config.games,
                "history": [*state["history"], record],
            }
            checkpoint = directory / f"checkpoints/generation-{generation:06d}"
            save_checkpoint(
                checkpoint,
                learner=learner,
                sampler=sampler,
                replay_paths=active_paths,
                run_state=state,
                run_config=run_config,
                source_commit=source,
            )
            print(json.dumps({k: v for k, v in record.items() if k != "metrics"}), flush=True)
    except TimeoutError as error:
        status, reason = "budget-stopped", str(error)
    except BaseException as error:
        status, reason = "failed", f"{type(error).__name__}: {error}"
        raise
    finally:
        result = {
            "status": status,
            "stop_reason": reason,
            "source_commit": source,
            "config": run_config,
            "state": state,
            "last_complete_checkpoint": f"checkpoints/generation-{state['generation']:06d}",
            "wall_seconds": time.perf_counter() - started,
            "process_cpu_seconds": time.process_time() - cpu_started,
            "children_cpu_seconds": resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime
            + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime
            - children_started.ru_utime
            - children_started.ru_stime,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "promotion_ready": False,
            "new_paid_resources": False,
        }
        # A session report is replaceable; immutable checkpoints retain every committed history.
        temporary = directory / ".result.tmp"
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        temporary.replace(directory / "result.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--generations", type=int, default=3, help="total completed generation target"
    )
    parser.add_argument("--wall-seconds", type=float, default=1800)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--legacy-mlx", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--actor-mode", choices=("thread", "process"), default="thread")
    parser.add_argument("--trainable-prefix", action="append", default=[])
    parser.add_argument("--opening-book", type=Path, help="Versioned book; train split only")
    for name, field in LoopConfig.__dataclass_fields__.items():
        parser.add_argument(
            "--" + name.replace("_", "-"), type=type(field.default), default=field.default
        )
    args = parser.parse_args()
    config = LoopConfig(**{name: getattr(args, name) for name in LoopConfig.__dataclass_fields__})
    run_loop(
        args.directory,
        config=config,
        generations=args.generations,
        wall_seconds=args.wall_seconds,
        weights=args.weights,
        legacy_mlx=args.legacy_mlx,
        resume=args.resume,
        actor_mode=args.actor_mode,
        trainable_prefixes=tuple(args.trainable_prefix),
        opening_book=args.opening_book,
    )


if __name__ == "__main__":
    main()
