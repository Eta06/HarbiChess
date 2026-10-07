"""Budgeted all-root alpha-beta/quiescence control, no novelty or strength claim.

Rule oracle is python-chess; no learned rule/dynamics claim. Static values are
mover-perspective win-minus-loss in [-1,1], exact terminal values outside that
range. No transposition cache: full repetition/claim history stays authoritative.
"""

import math
from dataclasses import dataclass

import chess


class BudgetExhausted(Exception):
    pass


@dataclass(frozen=True)
class Result:
    move: chess.Move | None
    value: float
    nodes: int
    evaluations: int
    completed_depth: int
    root_actions: int


class BudgetSearch:
    def __init__(self, evaluator, *, nodes=512, quiescence_plies=2, max_depth=8, guard=None):
        if nodes < 1 or quiescence_plies < 0 or max_depth < 1:
            raise ValueError("positive node/depth budget required")
        self.evaluator = evaluator
        self.node_budget = nodes
        self.qdepth = quiescence_plies
        self.max_depth = max_depth
        self.guard = guard or (lambda: None)
        self.nodes = self.evaluations = 0

    def consume(self):
        if self.nodes % 64 == 0:
            self.guard()
        if self.nodes >= self.node_budget:
            raise BudgetExhausted
        self.nodes += 1

    @staticmethod
    def terminal(board, ply):
        outcome = board.outcome(claim_draw=True)
        if outcome is None:
            return None
        if outcome.winner is None:
            return 0.0
        magnitude = 2.0 - min(ply, 10000) * 0.00001
        return magnitude if outcome.winner == board.turn else -magnitude

    def evaluate(self, board):
        self.evaluations += 1
        value = float(self.evaluator(board))
        if not math.isfinite(value) or not -1 <= value <= 1:
            raise ValueError("static evaluator requires finite mover W-L in [-1,1]")
        return value

    @staticmethod
    def ordered(board, moves):
        # Conventional MVV-LVA only orders search, never supplies value or labels.
        def key(move):
            victim = board.piece_type_at(move.to_square) or (1 if board.is_en_passant(move) else 0)
            attacker = board.piece_type_at(move.from_square) or 0
            return (-(16 * victim - attacker if victim else 0), -(move.promotion or 0), move.uci())

        return sorted(moves, key=key)

    def quiesce(self, board, alpha, beta, remaining, ply):
        self.consume()
        value = self.terminal(board, ply)
        if value is not None:
            return value
        checked = board.is_check()
        static = self.evaluate(board)
        if remaining == 0:
            return static
        best = -math.inf if checked else static
        if not checked:
            if best >= beta:
                return best
            alpha = max(alpha, best)
        moves = list(board.legal_moves)
        if not checked:
            moves = [m for m in moves if board.is_capture(m) or m.promotion]
        for move in self.ordered(board, moves):
            board.push(move)
            try:
                score = -self.quiesce(board, -beta, -alpha, remaining - 1, ply + 1)
            finally:
                board.pop()
            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best

    def negamax(self, board, depth, alpha, beta, ply):
        if depth == 0:
            return self.quiesce(board, alpha, beta, self.qdepth, ply)
        self.consume()
        terminal = self.terminal(board, ply)
        if terminal is not None:
            return terminal
        best = -math.inf
        for move in self.ordered(board, board.legal_moves):
            board.push(move)
            try:
                score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            finally:
                board.pop()
            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best

    def search(self, position):
        board = position.copy(stack=True)
        self.nodes = self.evaluations = 0
        self.consume()
        terminal = self.terminal(board, 0)
        if terminal is not None:
            return Result(None, terminal, self.nodes, 0, 0, 0)
        moves = sorted(board.legal_moves, key=lambda m: m.uci())
        if self.node_budget < len(moves) + 1:
            raise ValueError("node budget cannot cover every legal root action")
        scores = {}
        # Complete shallow coverage is reserved for every legal root action.
        for move in moves:
            board.push(move)
            try:
                self.consume()
                value = self.terminal(board, 1)
                scores[move] = -(self.evaluate(board) if value is None else value)
            finally:
                board.pop()
        winner = min(moves, key=lambda m: (-scores[m], m.uci()))
        best, completed = scores[winner], 1
        for depth in range(1, self.max_depth + 1):
            iteration, alpha = {}, -math.inf
            iteration_winner = None
            try:
                for move in sorted(moves, key=lambda m: (-scores[m], m.uci())):
                    board.push(move)
                    try:
                        value = -self.negamax(board, depth - 1, -math.inf, -alpha, 1)
                    finally:
                        board.pop()
                    iteration[move] = value
                    if iteration_winner is None or value > alpha:
                        iteration_winner = move
                    alpha = max(alpha, value)
            except BudgetExhausted:
                break
            # Only a fully completed root pass may replace the previous result.
            scores = iteration
            winner = iteration_winner
            best, completed = alpha, depth
        return Result(winner, best, self.nodes, self.evaluations, completed, len(moves))
