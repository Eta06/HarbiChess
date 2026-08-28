"""Shared-tree Full Gumbel MuZero search for deterministic chess."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove, ChessState
from harbichess.search.evaluator import PositionEvaluation, SearchEvaluator
from harbichess.search.mcts import MoveStatistics, SearchResult


@dataclass(frozen=True, slots=True)
class FullGumbelConfig:
    simulations: int = 128
    max_considered_actions: int = 16
    gumbel_scale: float = 0.0
    value_scale: float = 0.1
    maxvisit_init: float = 50.0
    claim_draw: bool = True

    def __post_init__(self) -> None:
        if self.simulations <= 0 or self.max_considered_actions <= 0:
            raise ValueError("Full Gumbel counts must be positive")
        if any(
            not math.isfinite(value) or value < 0
            for value in (self.gumbel_scale, self.value_scale, self.maxvisit_init)
        ):
            raise ValueError("Full Gumbel scales must be finite and non-negative")


@dataclass(slots=True)
class _Node:
    prior: float = 1.0
    raw_value: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[ChessMove, _Node] = field(default_factory=dict)

    @property
    def expanded(self) -> bool:
        return bool(self.children)

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0


@dataclass(frozen=True, slots=True)
class FullGumbelSearchResult(SearchResult):
    """Search result retaining the Full Gumbel action and soft policy target."""

    selected_action: ChessMove | None = None
    action_weights: tuple[tuple[ChessMove, float], ...] = ()

    def select_move(self, *, temperature: float, rng: random.Random) -> ChessMove:
        if not self.moves or self.selected_action is None:
            raise ValueError("cannot select a move from a terminal search")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if temperature == 0:
            return self.selected_action
        log_weights = tuple(
            math.log(max(weight, 1e-300)) / temperature
            for _, weight in self.action_weights
        )
        maximum = max(log_weights)
        weights = tuple(math.exp(weight - maximum) for weight in log_weights)
        threshold = rng.random() * sum(weights)
        cumulative = 0.0
        for (move, _), weight in zip(self.action_weights, weights, strict=True):
            cumulative += weight
            if cumulative >= threshold:
                return move
        return self.action_weights[-1][0]


def considered_visit_sequence(
    max_considered_actions: int, simulations: int
) -> tuple[int, ...]:
    """Return Mctx's root Sequential Halving considered-visit schedule."""

    if max_considered_actions <= 0 or simulations <= 0:
        raise ValueError("Sequential Halving counts must be positive")
    if max_considered_actions == 1:
        return tuple(range(simulations))
    log2max = math.ceil(math.log2(max_considered_actions))
    sequence: list[int] = []
    visits = [0] * max_considered_actions
    considered = max_considered_actions
    while len(sequence) < simulations:
        extra = max(1, simulations // (log2max * considered))
        for _ in range(extra):
            sequence.extend(visits[:considered])
            for index in range(considered):
                visits[index] += 1
        considered = max(2, considered // 2)
    return tuple(sequence[:simulations])


def _softmax(scores: tuple[float, ...]) -> tuple[float, ...]:
    maximum = max(scores)
    weights = tuple(math.exp(score - maximum) for score in scores)
    total = sum(weights)
    return tuple(weight / total for weight in weights)


def _gumbel(rng: random.Random) -> float:
    uniform = min(1.0 - 1e-12, max(1e-12, rng.random()))
    return -math.log(-math.log(uniform))


class FullGumbelMCTS:
    """Full Gumbel MuZero allocation over an AlphaZero state transition."""

    def __init__(
        self,
        evaluator: SearchEvaluator,
        *,
        rules: PythonChessRules | None = None,
        config: FullGumbelConfig | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.rules = rules or PythonChessRules()
        self.config = config or FullGumbelConfig()

    def search(
        self,
        state: ChessState,
        *,
        rng: random.Random,
        add_root_noise: bool = False,
    ) -> FullGumbelSearchResult:
        if add_root_noise:
            raise ValueError("Full Gumbel search does not use Dirichlet root noise")
        outcome = self.rules.outcome(state, claim_draw=self.config.claim_draw)
        if outcome is not None:
            side = self.rules.view(state).side_to_move
            return FullGumbelSearchResult(
                (), float(outcome.value_for(side)), 0, outcome
            )

        root = _Node()
        evaluation = self.evaluator.evaluate(state)
        network_priors = tuple(evaluation.priors)
        self._expand(root, evaluation)
        root_moves = tuple(root.children)
        logits = {
            move: math.log(max(root.children[move].prior, 1e-300))
            for move in root_moves
        }
        gumbels = {
            move: self.config.gumbel_scale * _gumbel(rng) for move in root_moves
        }
        considered = min(
            self.config.max_considered_actions,
            self.config.simulations,
            len(root_moves),
        )
        schedule = considered_visit_sequence(considered, self.config.simulations)

        for considered_visit in schedule:
            node = root
            simulation_state = state
            path = [root]
            while node.expanded:
                if node is root:
                    move, node = self._select_root_child(
                        root, logits, gumbels, considered_visit
                    )
                else:
                    move, node = self._select_interior_child(node)
                simulation_state = self.rules.apply(simulation_state, move)
                path.append(node)

            leaf_outcome = self.rules.outcome(
                simulation_state, claim_draw=self.config.claim_draw
            )
            if leaf_outcome is None:
                leaf_evaluation = self.evaluator.evaluate(simulation_state)
                self._expand(node, leaf_evaluation)
                value = leaf_evaluation.value
            else:
                side = self.rules.view(simulation_state).side_to_move
                value = float(leaf_outcome.value_for(side))
                node.raw_value = value
            self._backpropagate(path, value)

        completed = self._completed_q(root)
        max_visits = max(child.visit_count for child in root.children.values())
        finalists = tuple(
            move for move in root_moves if root.children[move].visit_count == max_visits
        )
        selected = min(
            finalists,
            key=lambda move: (
                -(gumbels[move] + logits[move] + completed[move]),
                move.uci,
            ),
        )
        probabilities = _softmax(
            tuple(logits[move] + completed[move] for move in root_moves)
        )
        action_weights = tuple(zip(root_moves, probabilities, strict=True))
        moves = tuple(
            sorted(
                (
                    MoveStatistics(
                        move,
                        child.visit_count,
                        child.prior,
                        -child.mean_value,
                    )
                    for move, child in root.children.items()
                ),
                key=lambda item: (-item.visits, item.move.uci),
            )
        )
        return FullGumbelSearchResult(
            moves=moves,
            root_value=root.mean_value,
            simulations=self.config.simulations,
            network_priors=network_priors,
            selected_action=selected,
            action_weights=action_weights,
        )

    def _select_root_child(
        self,
        root: _Node,
        logits: dict[ChessMove, float],
        gumbels: dict[ChessMove, float],
        considered_visit: int,
    ) -> tuple[ChessMove, _Node]:
        completed = self._completed_q(root)
        eligible = tuple(
            (move, child)
            for move, child in root.children.items()
            if child.visit_count == considered_visit
        )
        if not eligible:
            raise RuntimeError("Sequential Halving schedule has no eligible root action")
        return min(
            eligible,
            key=lambda item: (
                -(gumbels[item[0]] + logits[item[0]] + completed[item[0]]),
                item[0].uci,
            ),
        )

    def _select_interior_child(self, node: _Node) -> tuple[ChessMove, _Node]:
        moves = tuple(node.children)
        completed = self._completed_q(node)
        probabilities = _softmax(
            tuple(
                math.log(max(node.children[move].prior, 1e-300)) + completed[move]
                for move in moves
            )
        )
        total_visits = sum(child.visit_count for child in node.children.values())
        selected = min(
            zip(moves, probabilities, strict=True),
            key=lambda item: (
                -(
                    item[1]
                    - node.children[item[0]].visit_count / (1 + total_visits)
                ),
                item[0].uci,
            ),
        )[0]
        return selected, node.children[selected]

    def _completed_q(self, node: _Node) -> dict[ChessMove, float]:
        moves = tuple(node.children)
        visits = {move: node.children[move].visit_count for move in moves}
        qvalues = {
            move: -node.children[move].mean_value
            for move in moves
            if visits[move] > 0
        }
        total_visits = sum(visits.values())
        visited_prior = sum(
            node.children[move].prior for move in moves if visits[move] > 0
        )
        if visited_prior > 0:
            weighted_q = sum(
                node.children[move].prior * qvalues[move] / visited_prior
                for move in qvalues
            )
        else:
            weighted_q = 0.0
        mixed_value = (node.raw_value + total_visits * weighted_q) / (
            total_visits + 1
        )
        completed = {
            move: qvalues[move] if visits[move] > 0 else mixed_value for move in moves
        }
        minimum = min(completed.values())
        maximum = max(completed.values())
        denominator = max(maximum - minimum, 1e-8)
        scale = (self.config.maxvisit_init + max(visits.values())) * self.config.value_scale
        return {
            move: scale * (value - minimum) / denominator
            for move, value in completed.items()
        }

    @staticmethod
    def _expand(node: _Node, evaluation: PositionEvaluation) -> None:
        if node.expanded:
            raise RuntimeError("cannot expand a node twice")
        if not evaluation.priors or any(prior < 0 for _, prior in evaluation.priors):
            raise ValueError("position priors must be non-negative and non-empty")
        total = sum(prior for _, prior in evaluation.priors)
        if total <= 0:
            raise ValueError("position priors must have positive mass")
        node.raw_value = evaluation.value
        node.children = {
            move: _Node(prior=prior / total)
            for move, prior in sorted(evaluation.priors, key=lambda item: item[0].uci)
        }

    @staticmethod
    def _backpropagate(path: list[_Node], value: float) -> None:
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value
