"""Budgeted PVS/history prototype. Conventional engine methods, no strength claim."""

import enum
import math
from dataclasses import dataclass

import chess


class BudgetExhausted(Exception):
    pass


class Bound(enum.Enum):
    EXACT = "exact"
    LOWER = "lower"
    UPPER = "upper"


@dataclass(frozen=True)
class Entry:
    depth: int
    value: float
    bound: Bound
    move: chess.Move | None


@dataclass(frozen=True)
class Result:
    move: chess.Move | None
    value: float
    nodes: int
    evaluations: int
    completed_depth: int
    root_actions: int
    tt_hits: int = 0
    pvs_researches: int = 0
    q_check_extensions: int = 0
    incheck_horizon_returns: int = 0
    budget_exhausted: bool = False
    principal_variation: tuple[str, ...] = ()
    completed_root_passes: int = 0


class BudgetSearch:
    def __init__(
        self,
        evaluator,
        *,
        nodes=512,
        quiescence_plies=2,
        max_depth=8,
        guard=None,
        incheck_extensions=1,
        use_tt=True,
        use_pvs=True,
    ):
        if (
            type(nodes) is not int
            or nodes < 1
            or type(quiescence_plies) is not int
            or quiescence_plies < 0
            or type(max_depth) is not int
            or max_depth < 1
            or incheck_extensions not in (0, 1)
        ):
            raise ValueError("positive budget/depth; explicit bounded check extension0/1")
        self.evaluator = evaluator
        self.node_budget = nodes
        self.qdepth = quiescence_plies
        self.max_depth = max_depth
        self.guard = guard or (lambda: None)
        self.incheck_extensions = incheck_extensions
        self.use_tt = use_tt
        self.use_pvs = use_pvs
        self.reset()

    def reset(self):
        self.nodes = self.evaluations = self.tt_hits = self.pvs_researches = 0
        self.q_check_extensions = self.incheck_horizon_returns = 0
        self.tt = {}
        self.killers = {}
        self.history = {}
        self.pv_hint = {}
        self._active_board = None
        self._root_context = None
        self._root_stack_length = 0

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
            raise ValueError("finite mover static[-1,1]")
        return value

    @staticmethod
    def key(board, ply, phase):
        # Full root history plus current rule fields, NOT placement/FEN4-only.
        # Same current FEN after different repetitions cannot collide. ply included
        # because terminal magnitude is root-relative. No cross-root/global cache.
        return (
            board.root().fen(),
            tuple(m.uci() for m in board.move_stack),
            board.fen(),
            board.chess960,
            ply,
            phase,
        )

    def _key(self, board, ply, phase):
        # TT is reset for each search. Immutable root ancestry is its context;
        # only the exact continuation needs hashing at each internal node.
        if board is not self._active_board:
            return self.key(board, ply, phase)
        return (
            tuple(m.uci() for m in board.move_stack[self._root_stack_length :]),
            board.fen(),
            board.chess960,
            ply,
            phase,
        )

    def ordered(self, board, moves, ply, preferred=None):
        killers = self.killers.get(ply, ())

        def order(move):
            victim = board.piece_type_at(move.to_square) or (1 if board.is_en_passant(move) else 0)
            attacker = board.piece_type_at(move.from_square) or 0
            tactical = bool(victim or move.promotion)
            return (
                0 if move == preferred else 1,
                0 if tactical else 1,
                -(16 * victim - attacker if victim else 0),
                -(move.promotion or 0),
                killers.index(move) if move in killers else 2,
                -self.history.get((board.turn, move.from_square, move.to_square), 0),
                move.uci(),
            )

        return sorted(moves, key=order)

    def cutoff(self, board, move, depth, ply):
        if board.is_capture(move) or move.promotion:
            return
        old = self.killers.get(ply, ())
        self.killers[ply] = (move, *tuple(m for m in old if m != move)[:1])
        key = (board.turn, move.from_square, move.to_square)
        self.history[key] = min(1_000_000, self.history.get(key, 0) + depth * depth)

    def probe(self, key, depth, alpha, beta):
        entry = self.tt.get(key) if self.use_tt else None
        if entry is None or entry.depth != depth:
            return None, alpha, beta
        self.tt_hits += 1
        if entry.bound is Bound.EXACT:
            return entry.value, alpha, beta
        # Non-cutting cached bounds are used for ordering only. Keeping the
        # original window avoids falsely marking a narrowed fail-low exact.
        if (entry.bound is Bound.LOWER and entry.value >= beta) or (
            entry.bound is Bound.UPPER and entry.value <= alpha
        ):
            return entry.value, alpha, beta
        return None, alpha, beta

    def store(self, key, depth, value, alpha0, beta0, move):
        if not self.use_tt:
            return
        bound = Bound.UPPER if value <= alpha0 else Bound.LOWER if value >= beta0 else Bound.EXACT
        old = self.tt.get(key)
        if old is None or depth >= old.depth:
            self.tt[key] = Entry(depth, value, bound, move)
        if move is not None:
            self.pv_hint[key] = move

    def quiesce(self, board, alpha, beta, remaining, ply, checks_left):
        self.consume()
        terminal = self.terminal(board, ply)
        if terminal is not None:
            return terminal
        checked = board.is_check()
        # Baseline-compatible cap can be selected with incheck_extensions=0.
        # Default: one CHARGED evasion extension after q horizon; no stand-pat in
        # check until that explicit finite extension is exhausted.
        if remaining == 0 and (not checked or checks_left == 0):
            if checked:
                self.incheck_horizon_returns += 1
            return self.evaluate(board)
        extension = remaining == 0
        if extension:
            self.q_check_extensions += 1
        key = self._key(board, ply, ("q", remaining, checks_left))
        alpha0, beta0 = alpha, beta
        hit, alpha, beta = self.probe(key, 0, alpha, beta)
        if hit is not None:
            return hit
        best = -math.inf if checked else self.evaluate(board)
        if not checked:
            if best >= beta:
                self.store(key, 0, best, alpha0, beta0, None)
                return best
            alpha = max(alpha, best)
        moves = list(board.legal_moves)
        if not checked:
            moves = [m for m in moves if board.is_capture(m) or m.promotion]
        entry = self.tt.get(key)
        preferred = entry.move if entry else self.pv_hint.get(key)
        winner = None
        for move in self.ordered(board, moves, ply, preferred):
            board.push(move)
            try:
                score = -self.quiesce(
                    board,
                    -beta,
                    -alpha,
                    max(0, remaining - 1),
                    ply + 1,
                    checks_left - 1 if extension else checks_left,
                )
            finally:
                board.pop()
            if score > best:
                best, winner = score, move
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        self.store(key, 0, best, alpha0, beta0, winner)
        return best

    def negamax(self, board, depth, alpha, beta, ply):
        if depth == 0:
            return self.quiesce(board, alpha, beta, self.qdepth, ply, self.incheck_extensions)
        self.consume()
        terminal = self.terminal(board, ply)
        if terminal is not None:
            return terminal
        key = self._key(board, ply, ("n",))
        alpha0, beta0 = alpha, beta
        hit, alpha, beta = self.probe(key, depth, alpha, beta)
        if hit is not None:
            return hit
        entry = self.tt.get(key)
        preferred = entry.move if entry else self.pv_hint.get(key)
        best = -math.inf
        winner = None
        for index, move in enumerate(self.ordered(board, board.legal_moves, ply, preferred)):
            board.push(move)
            try:
                if index == 0 or not self.use_pvs:
                    score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
                else:
                    narrow = math.nextafter(alpha, math.inf)
                    score = -self.negamax(board, depth - 1, -narrow, -alpha, ply + 1)
                    if alpha < score < beta:
                        self.pvs_researches += 1
                        score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            finally:
                board.pop()
            if score > best:
                best, winner = score, move
            alpha = max(alpha, best)
            if alpha >= beta:
                self.cutoff(board, move, depth, ply)
                break
        self.store(key, depth, best, alpha0, beta0, winner)
        return best

    def search(self, position):
        try:
            return self._search_impl(position)
        finally:
            self._active_board = None
            self._root_context = None
            self._root_stack_length = 0

    def _search_impl(self, position):
        board = position.copy(stack=True)
        self.reset()
        self._active_board = board
        self._root_stack_length = len(board.move_stack)
        self._root_context = (board.root().fen(), tuple(m.uci() for m in board.move_stack))
        self.consume()
        terminal = self.terminal(board, 0)
        if terminal is not None:
            return Result(None, terminal, self.nodes, 0, 0, 0)
        moves = sorted(board.legal_moves, key=lambda m: m.uci())
        if self.node_budget < len(moves) + 1:
            raise ValueError("budget must cover ALL root actions")
        scores = {}
        for move in moves:
            board.push(move)
            try:
                self.consume()
                value = self.terminal(board, 1)
                scores[move] = -(self.evaluate(board) if value is None else value)
            finally:
                board.pop()
        winner = min(moves, key=lambda m: (-scores[m], m.uci()))
        best, completed = scores[winner], 0
        exhausted = False
        passes = 0
        for depth in range(1, self.max_depth + 1):
            iteration = {}
            alpha = -math.inf
            iteration_winner = None
            try:
                ordered = sorted(
                    moves, key=lambda m: (0 if m == winner else 1, -scores[m], m.uci())
                )
                for index, move in enumerate(ordered):
                    board.push(move)
                    try:
                        if index == 0 or not self.use_pvs:
                            value = -self.negamax(board, depth - 1, -math.inf, -alpha, 1)
                        else:
                            narrow = math.nextafter(alpha, math.inf)
                            value = -self.negamax(board, depth - 1, -narrow, -alpha, 1)
                            if value > alpha:
                                self.pvs_researches += 1
                                value = -self.negamax(board, depth - 1, -math.inf, -alpha, 1)
                    finally:
                        board.pop()
                    iteration[move] = value
                    if iteration_winner is None or value > alpha:
                        iteration_winner = move
                    alpha = max(alpha, value)
            except BudgetExhausted:
                exhausted = True
                break
            scores = iteration
            winner = iteration_winner
            best, completed = alpha, depth
            passes += 1
        # Only the legal selected root move is certified as PV here. Do not
        # pretend shallow TT/bound hints form a fully searched multi-ply PV.
        return Result(
            winner,
            best,
            self.nodes,
            self.evaluations,
            completed,
            len(moves),
            self.tt_hits,
            self.pvs_researches,
            self.q_check_extensions,
            self.incheck_horizon_returns,
            exhausted,
            (winner.uci(),),
            passes,
        )
