"""Explicit CPU spawn actors alongside the shared threaded inference path."""

from __future__ import annotations

import atexit
import multiprocessing
import resource
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import torch

from harbichess.backends.torch_backend import TorchPolicyValueBackend
from harbichess.backends.torch_network import TorchChessNetwork, load_weights
from harbichess.chess.rules import PythonChessRules
from harbichess.search.batching import SharedBatchEvaluator
from harbichess.search.evaluator import NeuralPositionEvaluator
from harbichess.search.full_gumbel import FullGumbelConfig, FullGumbelMCTS
from harbichess.selfplay.game import SelfPlayConfig, derive_game_seed, play_game

_actor_search = _actor_rules = _actor_bridge = _actor_config = None


class DeadlineEvaluator:
    def __init__(self, evaluator: NeuralPositionEvaluator, deadline: float) -> None:
        self.evaluator, self.deadline = evaluator, deadline

    def evaluate(self, state):
        if time.perf_counter() >= self.deadline:
            raise TimeoutError("wall-clock budget exhausted; resume last complete generation")
        return self.evaluator.evaluate(state)


def search_snapshot(network, config, deadline, *, threaded):
    rules = PythonChessRules()
    bridge = SharedBatchEvaluator(
        TorchPolicyValueBackend(network, device=config["device"]),
        max_batch_size=config["workers"] if threaded else 1,
        max_wait_seconds=0.00025 if threaded else 0,
    )
    search = FullGumbelMCTS(
        DeadlineEvaluator(NeuralPositionEvaluator(bridge, rules=rules), deadline),
        rules=rules,
        config=FullGumbelConfig(
            simulations=config["simulations"],
            max_considered_actions=min(16, config["simulations"]),
            gumbel_scale=config["gumbel_scale"],
        ),
    )
    return search, rules, bridge


def actor_game(search, rules, config, index):
    return play_game(
        search,
        rules,
        rules.initial_state(),
        game_index=index,
        seed=derive_game_seed(config["seed"], index),
        config=SelfPlayConfig(max_plies=config["max_plies"], search_root_noise=False),
    )


def collect_thread_games(network: TorchChessNetwork, config: dict, indices: range, deadline: float):
    search, rules, bridge = search_snapshot(network, config, deadline, threaded=True)
    try:

        def game(index):
            return actor_game(search, rules, config, index)

        with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
            games = tuple(pool.map(game, indices))
        return games, asdict(bridge.statistics)
    finally:
        bridge.close()


def _initialize_actor(weights: str, config: dict, deadline: float):
    global _actor_search, _actor_rules, _actor_bridge, _actor_config
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    _actor_config = config
    _actor_search, _actor_rules, _actor_bridge = search_snapshot(
        load_weights(Path(weights)), config, deadline, threaded=False
    )
    atexit.register(_actor_bridge.close)


def _process_game(index):
    before = time.process_time()
    _actor_bridge.reset_statistics()
    game = actor_game(_actor_search, _actor_rules, _actor_config, index)
    statistics = asdict(_actor_bridge.statistics)
    statistics["worker_cpu_seconds"] = time.process_time() - before
    statistics["worker_maxrss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return game, statistics


def collect_process_games(weights: Path, config: dict, indices: range, deadline: float):
    if config["device"] != "cpu" or config["threads"] != 1:
        raise ValueError("spawn actors require CPU and one Torch thread per process")
    with ProcessPoolExecutor(
        max_workers=config["workers"],
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_actor,
        initargs=(str(weights.resolve()), config, deadline),
    ) as pool:
        results = tuple(pool.map(_process_game, indices))
    stats = {
        k: sum(row[k] for _, row in results)
        for k in (
            "positions",
            "batches",
            "backend_seconds",
            "queue_wait_seconds",
            "worker_cpu_seconds",
        )
    }
    stats["largest_batch"] = max(row["largest_batch"] for _, row in results)
    stats["largest_worker_rss_kib"] = max(row["worker_maxrss_kib"] for _, row in results)
    return tuple(game for game, _ in results), stats
