"""Qualify policy/value search teachers before starting a self-learning generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from harbichess.backends.mlx_backend import MLXPolicyValueBackend
from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove
from harbichess.dashboard.state import RunMode, SnapshotStore
from harbichess.replay.schema import ReplayRecord
from harbichess.replay.shard import read_shard
from harbichess.search.batching import SharedBatchEvaluator
from harbichess.search.evaluator import NeuralPositionEvaluator, PositionEvaluation
from harbichess.search.gumbel import GumbelSearchConfig, gumbel_sequential_halving
from harbichess.search.mcts import MCTS, SearchConfig
from harbichess.search.targets import prune_noise_attributable_visits, visit_policy

Policy = tuple[tuple[ChessMove, float], ...]


@dataclass(frozen=True, slots=True)
class QualificationConfig:
    run_result: Path
    shards: tuple[Path, ...]
    output_dir: Path
    positions: int = 32
    workers: int = 96
    seed: int = 20260827
    search_budgets: tuple[int, ...] = (64, 128, 256, 512, 800)
    gumbel_budgets: tuple[int, ...] = (64, 128)
    search_repetitions: int = 2
    verification_simulations: int = 128
    bootstrap_samples: int = 2_000
    maximum_stability_tv: float = 0.10
    maximum_value_mse_regression: float = 0.0

    def __post_init__(self) -> None:
        counts = (
            self.positions,
            self.workers,
            self.search_repetitions,
            self.verification_simulations,
            self.bootstrap_samples,
        )
        if (
            not self.shards
            or any(value <= 0 for value in counts)
            or any(value <= 0 for value in (*self.search_budgets, *self.gumbel_budgets))
            or not 0.0 <= self.maximum_stability_tv <= 1.0
            or self.maximum_value_mse_regression < 0.0
        ):
            raise ValueError("teacher qualification configuration is invalid")


@dataclass(frozen=True, slots=True)
class PositionTarget:
    policy: Policy
    root_value: float


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = handle.name
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def _network_config(payload: Mapping[str, object]) -> NetworkConfig:
    config = payload["config"]
    return NetworkConfig(
        trunk_channels=int(config["trunk_channels"]),
        residual_blocks=int(config["residual_blocks"]),
        policy_channels=int(config["policy_channels"]),
        value_channels=int(config["value_channels"]),
        value_hidden=int(config["value_hidden"]),
    )


def _phase(ply: int) -> str:
    if ply < 20:
        return "opening"
    if ply < 80:
        return "middlegame"
    return "endgame"


def _branching(count: int) -> str:
    if count <= 20:
        return "low"
    if count <= 35:
        return "medium"
    return "high"


def _outcome(record: ReplayRecord) -> str:
    if record.outcome_value is None:
        return "unknown"
    return "draw" if record.outcome_value == 0 else "decisive"


def _stable_rank(record: ReplayRecord, seed: int) -> bytes:
    payload = f"{seed}:{record.game_id}:{record.ply}".encode()
    return hashlib.blake2b(payload, digest_size=16).digest()


def _stratum(record: ReplayRecord, rules: PythonChessRules) -> tuple[str, str, str, str]:
    legal_moves = rules.legal_moves(record.state)
    repetition = (
        "repeat-risk" if rules.claimable_threefold_moves(record.state, legal_moves) else "ordinary"
    )
    return (_phase(record.ply), _branching(len(legal_moves)), _outcome(record), repetition)


def select_stratified_records(
    records: Sequence[ReplayRecord],
    *,
    rules: PythonChessRules,
    count: int,
    seed: int,
) -> tuple[ReplayRecord, ...]:
    """Round-robin deterministic samples across phase, branching, result, repetition."""

    if count <= 0 or not records:
        raise ValueError("stratified selection requires records and a positive count")
    strata: dict[tuple[str, str, str, str], list[ReplayRecord]] = defaultdict(list)
    for record in records:
        strata[_stratum(record, rules)].append(record)
    for items in strata.values():
        items.sort(key=lambda record: _stable_rank(record, seed))

    selected: list[ReplayRecord] = []
    keys = tuple(sorted(strata))
    while len(selected) < min(count, len(records)):
        progressed = False
        for key in keys:
            if strata[key]:
                selected.append(strata[key].pop())
                progressed = True
                if len(selected) == min(count, len(records)):
                    break
        if not progressed:
            break
    return tuple(selected)


def _policy(evaluation: PositionEvaluation) -> Policy:
    return tuple(evaluation.priors)


def _mean_policy(policies: Sequence[Policy]) -> Policy:
    if not policies:
        raise ValueError("cannot average an empty policy collection")
    totals: dict[ChessMove, float] = defaultdict(float)
    for policy in policies:
        for move, probability in policy:
            totals[move] += probability / len(policies)
    return tuple(sorted(totals.items(), key=lambda item: item[0].uci))


def _probabilities(policy: Policy) -> dict[ChessMove, float]:
    return dict(policy)


def _tv(first: Policy, second: Policy) -> float:
    left, right = _probabilities(first), _probabilities(second)
    return 0.5 * sum(
        abs(left.get(move, 0.0) - right.get(move, 0.0)) for move in left.keys() | right.keys()
    )


def _kl(target: Policy, prior: Policy) -> float:
    clean = _probabilities(prior)
    return sum(
        probability * math.log(probability / max(clean.get(move, 0.0), 1e-12))
        for move, probability in target
        if probability > 0.0
    )


def _entropy(policy: Policy) -> float:
    return -sum(probability * math.log(probability) for _, probability in policy if probability)


def _argmax(policy: Policy) -> ChessMove:
    return min(policy, key=lambda item: (-item[1], item[0].uci))[0]


def _interval(values: Sequence[float], *, samples: int, seed: int) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(samples))
    return means[round(0.025 * (samples - 1))], means[round(0.975 * (samples - 1))]


def _seed(seed: int, variant: str, position: int, repetition: int) -> int:
    payload = f"{seed}:{variant}:{position}:{repetition}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest())


def _puct_target(
    search: MCTS,
    record: ReplayRecord,
    raw: PositionEvaluation,
    *,
    noisy: bool,
    seed: int,
    pruned: bool,
) -> PositionTarget:
    result = search.search(record.state, rng=random.Random(seed), add_root_noise=noisy)
    policy = visit_policy(result)
    if pruned:
        policy = prune_noise_attributable_visits(result, dict(raw.priors))
    return PositionTarget(policy, result.root_value)


def _gumbel_target(
    search: MCTS,
    record: ReplayRecord,
    *,
    budget: int,
    seed: int,
) -> PositionTarget:
    result = gumbel_sequential_halving(
        search,
        record.state,
        rng=random.Random(seed),
        config=GumbelSearchConfig(simulations=budget),
    )
    action_values = dict(result.action_values)
    root_value = sum(probability * action_values[move] for move, probability in result.policy)
    return PositionTarget(result.policy, root_value)


def _verify_action(search: MCTS, record: ReplayRecord, move: ChessMove) -> float:
    child = search.rules.apply(record.state, move)
    result = search.search(child, rng=random.Random(0), add_root_noise=False)
    return -result.root_value


def run_teacher_qualification(config: QualificationConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"qualification output already exists: {config.output_dir}")
    run = json.loads(config.run_result.read_text(encoding="utf-8"))
    baseline = run.get("baseline")
    if baseline is None:
        raise ValueError("qualification requires a persisted baseline model")
    rules = PythonChessRules()
    shards = tuple(read_shard(path, rules=rules) for path in config.shards)
    records = tuple(record for shard in shards for record in shard.records)
    selected = select_stratified_records(
        records,
        rules=rules,
        count=config.positions,
        seed=config.seed,
    )
    network = HarbiChessNetwork(_network_config(run))
    network.load_weights(str(baseline["path"]))
    batcher = SharedBatchEvaluator(
        MLXPolicyValueBackend(network),
        max_batch_size=min(128, config.workers * 2),
        max_wait_seconds=0.00025,
    )
    evaluator = NeuralPositionEvaluator(batcher, rules=rules)
    started = time.perf_counter()
    workers = min(config.workers, len(selected))

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            raw_evaluations = tuple(
                pool.map(lambda record: evaluator.evaluate(record.state), selected)
            )
        targets: dict[str, tuple[tuple[PositionTarget, ...], ...]] = {
            "raw": tuple((PositionTarget(_policy(raw), raw.value),) for raw in raw_evaluations)
        }
        for budget in config.search_budgets:
            search = MCTS(
                evaluator,
                rules=rules,
                config=SearchConfig(simulations=budget),
            )
            for noisy in (False, True):
                variant = f"puct-{budget}-{'noise' if noisy else 'clean'}"

                def inspect(
                    index: int,
                    search: MCTS = search,
                    noisy: bool = noisy,
                    variant: str = variant,
                ) -> tuple[PositionTarget, ...]:
                    return tuple(
                        _puct_target(
                            search,
                            selected[index],
                            raw_evaluations[index],
                            noisy=noisy,
                            seed=_seed(config.seed, variant, index, repetition),
                            pruned=False,
                        )
                        for repetition in range(config.search_repetitions)
                    )

                with ThreadPoolExecutor(max_workers=workers) as pool:
                    variant_targets = tuple(pool.map(inspect, range(len(selected))))
                targets[variant] = variant_targets
                if noisy:
                    pruned_name = f"{variant}-pruned"

                    def prune(
                        index: int,
                        search: MCTS = search,
                        variant: str = variant,
                    ) -> tuple[PositionTarget, ...]:
                        return tuple(
                            _puct_target(
                                search,
                                selected[index],
                                raw_evaluations[index],
                                noisy=True,
                                seed=_seed(config.seed, variant, index, repetition),
                                pruned=True,
                            )
                            for repetition in range(config.search_repetitions)
                        )

                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        targets[pruned_name] = tuple(pool.map(prune, range(len(selected))))

        for budget in config.gumbel_budgets:
            variant = f"gumbel-{budget}"
            search = MCTS(
                evaluator,
                rules=rules,
                config=SearchConfig(simulations=1, dirichlet_fraction=0.0),
            )

            def inspect_gumbel(
                index: int,
                search: MCTS = search,
                budget: int = budget,
                variant: str = variant,
            ) -> tuple[PositionTarget, ...]:
                return tuple(
                    _gumbel_target(
                        search,
                        selected[index],
                        budget=budget,
                        seed=_seed(config.seed, variant, index, repetition),
                    )
                    for repetition in range(config.search_repetitions)
                )

            with ThreadPoolExecutor(max_workers=workers) as pool:
                targets[variant] = tuple(pool.map(inspect_gumbel, range(len(selected))))

        averaged = {
            variant: tuple(
                PositionTarget(
                    _mean_policy(tuple(item.policy for item in repetitions)),
                    sum(item.root_value for item in repetitions) / len(repetitions),
                )
                for repetitions in per_position
            )
            for variant, per_position in targets.items()
        }
        actions = {
            (index, _argmax(target.policy))
            for variant_targets in averaged.values()
            for index, target in enumerate(variant_targets)
        }
        verifier = MCTS(
            evaluator,
            rules=rules,
            config=SearchConfig(
                simulations=config.verification_simulations,
                dirichlet_fraction=0.0,
            ),
        )

        def verify(item: tuple[int, ChessMove]) -> tuple[tuple[int, ChessMove], float]:
            index, move = item
            return item, _verify_action(verifier, selected[index], move)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            verified_values = dict(
                pool.map(
                    verify,
                    sorted(actions, key=lambda item: (item[0], item[1].uci)),
                )
            )

        raw_targets = averaged["raw"]
        labeled = tuple(
            index for index, record in enumerate(selected) if record.outcome_value is not None
        )
        raw_mse = (
            sum(
                (raw_targets[index].root_value - selected[index].outcome_value) ** 2
                for index in labeled
            )
            / len(labeled)
            if labeled
            else None
        )
        summaries = {}
        for variant, per_position in averaged.items():
            stability = tuple(
                _tv(repetitions[0].policy, repetitions[1].policy)
                for repetitions in targets[variant]
                if len(repetitions) > 1
            )
            deltas = tuple(
                verified_values[(index, _argmax(target.policy))]
                - verified_values[(index, _argmax(raw_targets[index].policy))]
                for index, target in enumerate(per_position)
            )
            interval = _interval(
                deltas,
                samples=config.bootstrap_samples,
                seed=_seed(config.seed, variant, 0, 0),
            )
            value_mse = (
                sum(
                    (per_position[index].root_value - selected[index].outcome_value) ** 2
                    for index in labeled
                )
                / len(labeled)
                if labeled
                else None
            )
            mean_stability = sum(stability) / len(stability) if stability else 0.0
            qualified = (
                variant != "raw"
                and interval[0] > 0.0
                and mean_stability <= config.maximum_stability_tv
                and (
                    raw_mse is None
                    or value_mse is None
                    or value_mse <= raw_mse + config.maximum_value_mse_regression
                )
            )
            summaries[variant] = {
                "qualified": qualified,
                "mean_policy_tv_from_raw": sum(
                    _tv(target.policy, raw_targets[index].policy)
                    for index, target in enumerate(per_position)
                )
                / len(per_position),
                "mean_policy_kl_from_raw": sum(
                    _kl(target.policy, raw_targets[index].policy)
                    for index, target in enumerate(per_position)
                )
                / len(per_position),
                "mean_policy_entropy": sum(_entropy(target.policy) for target in per_position)
                / len(per_position),
                "raw_argmax_agreement": sum(
                    _argmax(target.policy) == _argmax(raw_targets[index].policy)
                    for index, target in enumerate(per_position)
                )
                / len(per_position),
                "mean_seed_stability_tv": mean_stability,
                "value_mse": value_mse,
                "mean_verified_action_value_delta": sum(deltas) / len(deltas),
                "verified_action_value_delta_95_interval": interval,
            }
    finally:
        batcher.close()

    elapsed = time.perf_counter() - started
    statistics = batcher.statistics
    config.output_dir.mkdir(parents=True, exist_ok=False)
    result_path = config.output_dir / "qualification.json"
    _atomic_json(
        result_path,
        {
            "created_at": datetime.now(UTC).isoformat(),
            "source_commit": _source_commit(),
            "run_result": str(config.run_result),
            "baseline": baseline,
            "config": {
                **asdict(config),
                "run_result": str(config.run_result),
                "shards": [str(path) for path in config.shards],
                "output_dir": str(config.output_dir),
            },
            "gate": {
                "qualified": any(summary["qualified"] for summary in summaries.values()),
                "qualified_variants": [
                    variant for variant, summary in summaries.items() if summary["qualified"]
                ],
                "continuous_learner_authorized": False,
                "note": "qualification never starts a generation or continuous learner",
            },
            "selection": {
                "available_records": len(records),
                "selected_records": len(selected),
                "strata": dict(
                    sorted(
                        Counter("|".join(_stratum(record, rules)) for record in selected).items()
                    )
                ),
                "positions": [
                    {
                        "game_id": record.game_id,
                        "ply": record.ply,
                        "side_to_move": record.side_to_move,
                        "outcome_value": record.outcome_value,
                    }
                    for record in selected
                ],
            },
            "raw_value_mse": raw_mse,
            "variants": summaries,
            "timing": {"elapsed_seconds": elapsed},
            "inference": {
                **asdict(statistics),
                "average_batch_size": statistics.average_batch_size,
                "average_queue_wait_ms": statistics.average_queue_wait_ms,
            },
        },
    )
    return result_path


def publish_qualification_result(result_path: Path, telemetry_path: Path) -> None:
    """Publish one completed qualification without changing training state."""

    result = json.loads(result_path.read_text(encoding="utf-8"))
    variants = {name: summary for name, summary in result["variants"].items() if name != "raw"}
    best_name, best = max(
        variants.items(),
        key=lambda item: item[1]["mean_verified_action_value_delta"],
    )
    low, high = best["verified_action_value_delta_95_interval"]
    qualified = bool(result["gate"]["qualified"])
    store = SnapshotStore(telemetry_path)
    snapshot = store.read()
    store.write_atomic(
        replace(
            snapshot,
            updated_at=datetime.now(UTC).isoformat(),
            mode=RunMode.IDLE,
            mode_detail=(
                "OMURGA teacher qualified · continuous learner still requires approval"
                if qualified
                else "OMURGA teacher rejected · continuous learner blocked"
            ),
            run_id=result_path.parent.name,
            source_commit=result["source_commit"],
            teacher_qualification_status="passed" if qualified else "failed",
            teacher_qualification_positions=result["selection"]["selected_records"],
            teacher_qualification_variants=len(variants),
            teacher_qualified_variants=tuple(result["gate"]["qualified_variants"]),
            teacher_best_variant=best_name,
            teacher_best_value_delta=best["mean_verified_action_value_delta"],
            teacher_best_value_delta_low=low,
            teacher_best_value_delta_high=high,
            teacher_best_stability_tv=best["mean_seed_stability_tv"],
            teacher_raw_value_mse=result["raw_value_mse"],
            teacher_best_value_mse=best["value_mse"],
            teacher_qualification_result=str(result_path),
            promotion_ready=False,
        )
    )


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-result", required=True, type=Path)
    parser.add_argument("--shard", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--positions", type=int, default=32)
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--search-budgets", default="64,128,256,512,800")
    parser.add_argument("--gumbel-budgets", default="64,128")
    parser.add_argument("--search-repetitions", type=int, default=2)
    parser.add_argument("--verification-simulations", type=int, default=128)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--maximum-stability-tv", type=float, default=0.10)
    parser.add_argument("--maximum-value-mse-regression", type=float, default=0.0)
    parser.add_argument("--telemetry", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    path = run_teacher_qualification(
        QualificationConfig(
            run_result=arguments.run_result,
            shards=tuple(arguments.shard),
            output_dir=arguments.output_dir,
            positions=arguments.positions,
            workers=arguments.workers,
            seed=arguments.seed,
            search_budgets=_csv_ints(arguments.search_budgets),
            gumbel_budgets=_csv_ints(arguments.gumbel_budgets),
            search_repetitions=arguments.search_repetitions,
            verification_simulations=arguments.verification_simulations,
            bootstrap_samples=arguments.bootstrap_samples,
            maximum_stability_tv=arguments.maximum_stability_tv,
            maximum_value_mse_regression=arguments.maximum_value_mse_regression,
        )
    )
    if arguments.telemetry is not None:
        publish_qualification_result(path, arguments.telemetry)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
