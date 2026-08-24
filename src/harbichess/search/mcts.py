"""PUCT Monte Carlo tree search with deterministic per-game randomness."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove, ChessState, GameOutcome
from harbichess.search.evaluator import PositionEvaluation, SearchEvaluator


@dataclass(frozen=True, slots=True)
class SearchConfig:
    simulations: int = 128
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_fraction: float = 0.25
    claim_draw: bool = True

    def __post_init__(self) -> None:
        if self.simulations <= 0 or self.c_puct <= 0 or self.dirichlet_alpha <= 0:
            raise ValueError("simulations, c_puct, and dirichlet_alpha must be positive")
        if not 0.0 <= self.dirichlet_fraction <= 1.0:
            raise ValueError("dirichlet_fraction must be between zero and one")


@dataclass(slots=True)
class SearchNode:
    prior: float = 1.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[ChessMove, SearchNode] = field(default_factory=dict)

    @property
    def expanded(self) -> bool:
        return bool(self.children)

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0


@dataclass(frozen=True, slots=True)
class MoveStatistics:
    move: ChessMove
    visits: int
    prior: float
    mean_value: float


@dataclass(frozen=True, slots=True)
class SearchResult:
    moves: tuple[MoveStatistics, ...]
    root_value: float
    simulations: int
    outcome: GameOutcome | None = None

    def select_move(self, *, temperature: float, rng: random.Random) -> ChessMove:
        if not self.moves:
            raise ValueError("cannot select a move from a terminal search")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if temperature == 0:
            return sorted(self.moves, key=lambda item: (-item.visits, item.move.uci))[0].move
        log_weights = [
            math.log(statistics.visits) / temperature if statistics.visits else -math.inf
            for statistics in self.moves
        ]
        maximum = max(log_weights)
        if maximum == -math.inf:
            weights = [statistics.prior for statistics in self.moves]
        else:
            weights = [math.exp(weight - maximum) for weight in log_weights]
        threshold = rng.random() * sum(weights)
        cumulative = 0.0
        for statistics, weight in zip(self.moves, weights, strict=True):
            cumulative += weight
            if cumulative >= threshold:
                return statistics.move
        return self.moves[-1].move


class MCTS:
    def __init__(
        self,
        evaluator: SearchEvaluator,
        *,
        rules: PythonChessRules | None = None,
        config: SearchConfig | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.rules = rules or PythonChessRules()
        self.config = config or SearchConfig()

    def search(
        self,
        state: ChessState,
        *,
        rng: random.Random,
        add_root_noise: bool = False,
    ) -> SearchResult:
        root_outcome = self.rules.outcome(state, claim_draw=self.config.claim_draw)
        if root_outcome is not None:
            side_to_move = self.rules.view(state).side_to_move
            root_value = float(root_outcome.value_for(side_to_move))
            return SearchResult((), root_value, 0, root_outcome)

        root = SearchNode()
        self._expand(root, self.evaluator.evaluate(state))
        if add_root_noise:
            self._add_root_noise(root, rng)

        for _ in range(self.config.simulations):
            node = root
            simulation_state = state
            path = [root]
            while node.expanded:
                move, node = self._select_child(node)
                simulation_state = self.rules.apply(simulation_state, move)
                path.append(node)

            outcome = self.rules.outcome(
                simulation_state,
                claim_draw=self.config.claim_draw,
            )
            if outcome is None:
                evaluation = self.evaluator.evaluate(simulation_state)
                self._expand(node, evaluation)
                value = evaluation.value
            else:
                side_to_move = self.rules.view(simulation_state).side_to_move
                value = float(outcome.value_for(side_to_move))
            self._backpropagate(path, value)

        moves = tuple(
            sorted(
                (
                    MoveStatistics(move, child.visit_count, child.prior, -child.mean_value)
                    for move, child in root.children.items()
                ),
                key=lambda item: (-item.visits, item.move.uci),
            )
        )
        return SearchResult(moves, root.mean_value, self.config.simulations)

    def _select_child(self, parent: SearchNode) -> tuple[ChessMove, SearchNode]:
        scale = math.sqrt(max(1, parent.visit_count))

        def score(item: tuple[ChessMove, SearchNode]) -> float:
            _, child = item
            exploitation = -child.mean_value
            exploration = self.config.c_puct * child.prior * scale / (1 + child.visit_count)
            return exploitation + exploration

        return max(parent.children.items(), key=score)

    @staticmethod
    def _expand(node: SearchNode, evaluation: PositionEvaluation) -> None:
        if node.expanded:
            raise RuntimeError("cannot expand a node twice")
        if not evaluation.priors or any(prior < 0 for _, prior in evaluation.priors):
            raise ValueError("position priors must be non-negative and non-empty")
        total = sum(prior for _, prior in evaluation.priors)
        if total <= 0:
            raise ValueError("position priors must have positive mass")
        node.children = {
            move: SearchNode(prior=prior / total)
            for move, prior in sorted(evaluation.priors, key=lambda item: item[0].uci)
        }

    def _add_root_noise(self, root: SearchNode, rng: random.Random) -> None:
        noise = [
            rng.gammavariate(self.config.dirichlet_alpha, 1.0)
            for _ in root.children
        ]
        total = sum(noise)
        fraction = self.config.dirichlet_fraction
        for child, sample in zip(root.children.values(), noise, strict=True):
            child.prior = (1.0 - fraction) * child.prior + fraction * sample / total

    @staticmethod
    def _backpropagate(path: list[SearchNode], value: float) -> None:
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value
