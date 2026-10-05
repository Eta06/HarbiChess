"""Wavefront batching for own-model FullGumbel targets with per-actor search RNG."""

from __future__ import annotations

import math
from dataclasses import dataclass

from harbichess.chess.actions import move_to_action
from harbichess.chess.encoding import BoardEncoder
from harbichess.core.state import ChessMove
from harbichess.search.evaluator import PositionEvaluation, _softmax
from harbichess.search.full_gumbel import (
    FullGumbelMCTS,
    FullGumbelSearchResult,
    _gumbel,
    _Node,
    considered_visit_sequence,
)
from harbichess.search.mate_certificates import certified_policy, immediate_mating_moves
from harbichess.search.mcts import MoveStatistics


class BatchedPositionEvaluator:
    def __init__(self, backend, rules):
        self.backend, self.rules = backend, rules
        self.encoder = BoardEncoder(rules)
        self.batch_sizes = []

    def evaluate_many(self, states):
        if not states:
            return ()
        boards = [self.rules.inspect(state) for state in states]
        moves = [
            tuple(sorted(board.legal_moves, key=lambda move: move.uci()))
            for board in boards
        ]
        if any(not row for row in moves):
            raise ValueError("terminal states never enter neural batch")
        encoded = [
            self.encoder.encode_state(state, board)
            for state, board in zip(states, boards, strict=True)
        ]
        legal = [
            tuple(move_to_action(board, move) for move in row)
            for board, row in zip(boards, moves, strict=True)
        ]
        outputs = self.backend.evaluate_masked(encoded, legal)
        self.batch_sizes.append(len(states))
        results = []
        for row, output in zip(moves, outputs, strict=True):
            probabilities = _softmax(output.policy_logits)
            win, _, loss = _softmax(output.wdl_logits)
            results.append(
                PositionEvaluation(
                    tuple(
                        (ChessMove(move.uci()), value)
                        for move, value in zip(row, probabilities, strict=True)
                    ),
                    win - loss,
                )
            )
        return tuple(results)


@dataclass
class Root:
    state: object
    node: object
    moves: tuple
    logits: dict
    gumbels: dict
    schedule: tuple
    priors: tuple
    certified_mates: tuple
    solved_action: object | None


class WavefrontGumbel:
    def __init__(self, evaluator, rules, config):
        self.evaluator, self.rules, self.config = evaluator, rules, config
        self.core = FullGumbelMCTS(None, rules=rules, config=config)

    def search_many(self, states, rngs, guard=lambda: None):
        if (
            not states
            or len(states) != len(rngs)
            or len({id(rng) for rng in rngs}) != len(rngs)
        ):
            raise ValueError("each nonempty actor batch requires its own RNG")
        guard()
        outcomes = [
            self.rules.outcome(state, claim_draw=self.config.claim_draw)
            for state in states
        ]
        active = [i for i, outcome in enumerate(outcomes) if outcome is None]
        evaluations = self.evaluator.evaluate_many([states[i] for i in active])
        roots = {}
        for i, evaluation in zip(active, evaluations, strict=True):
            node = _Node()
            self.core._expand(node, evaluation)
            moves = tuple(node.children)
            roots[i] = Root(
                states[i],
                node,
                moves,
                {m: math.log(max(node.children[m].prior, 1e-300)) for m in moves},
                {m: self.config.gumbel_scale * _gumbel(rngs[i]) for m in moves},
                considered_visit_sequence(
                    min(
                        self.config.max_considered_actions,
                        self.config.simulations,
                        len(moves),
                    ),
                    self.config.simulations,
                ),
                tuple(evaluation.priors),
                immediate_mating_moves(
                    self.rules, states[i], claim_draw=self.config.claim_draw
                ),
                None,
            )
            root = roots[i]
            if root.certified_mates:
                scores = {m: root.logits[m] + root.gumbels[m] for m in root.certified_mates}
                root.solved_action = min(
                    root.certified_mates,
                    key=lambda m: (-scores[m], m.uci),
                )
        for simulation in range(self.config.simulations):
            guard()
            pending = []
            for i in active:
                root = roots[i]
                node, state, path = root.node, root.state, [root.node]
                if root.solved_action is not None:
                    move = root.solved_action
                    node = root.node.children[move]
                    state = self.rules.apply(state, move)
                    path.append(node)
                else:
                    while node.expanded:
                        if node is root.node:
                            move, node = self.core._select_root_child(
                                node, root.logits, root.gumbels, root.schedule[simulation]
                            )
                        else:
                            move, node = self.core._select_interior_child(node)
                        state = self.rules.apply(state, move)
                        path.append(node)
                outcome = self.rules.outcome(state, claim_draw=self.config.claim_draw)
                if outcome is None:
                    pending.append((node, state, path))
                else:
                    value = float(
                        outcome.value_for(self.rules.view(state).side_to_move)
                    )
                    node.raw_value = value
                    self.core._backpropagate(path, value)
            guard()
            evaluations = self.evaluator.evaluate_many(
                [state for _, state, _ in pending]
            )
            for (node, _, path), evaluation in zip(pending, evaluations, strict=True):
                self.core._expand(node, evaluation)
                self.core._backpropagate(path, evaluation.value)
        results = []
        for i, outcome in enumerate(outcomes):
            if outcome is not None:
                results.append(
                    FullGumbelSearchResult(
                        (),
                        float(
                            outcome.value_for(self.rules.view(states[i]).side_to_move)
                        ),
                        0,
                        outcome,
                    )
                )
                continue
            root = roots[i]
            completed = self.core._completed_q(root.node)
            if root.solved_action is not None:
                selected = root.solved_action
                probabilities, _ = certified_policy(
                    root.moves,
                    root.certified_mates,
                    {m: root.gumbels[m] + root.logits[m] for m in root.certified_mates},
                )
            else:
                max_visits = max(child.visit_count for child in root.node.children.values())
                finalists = [
                    m for m in root.moves if root.node.children[m].visit_count == max_visits
                ]
                selected = min(
                    finalists,
                    key=lambda m: (
                        -(root.gumbels[m] + root.logits[m] + completed[m]),
                        m.uci,
                    ),
                )
                probabilities = _softmax(
                    tuple(root.logits[m] + completed[m] for m in root.moves)
                )
            moves = tuple(
                sorted(
                    (
                        MoveStatistics(
                            m, child.visit_count, child.prior, -child.mean_value
                            if m not in root.certified_mates else 1.0
                        )
                        for m, child in root.node.children.items()
                    ),
                    key=lambda item: (-item.visits, item.move.uci),
                )
            )
            results.append(
                FullGumbelSearchResult(
                    moves=moves,
                    root_value=root.node.mean_value,
                    simulations=self.config.simulations,
                    network_priors=root.priors,
                    selected_action=selected,
                    action_weights=tuple(zip(root.moves, probabilities, strict=True)),
                    certified_mates=root.certified_mates,
                )
            )
        return tuple(results)
