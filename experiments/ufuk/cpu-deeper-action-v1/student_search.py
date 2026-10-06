"""Standard learned root move ordering; no value/label/weight/budget changes."""

import math


def search_type(base_search, result_type, budget_exhausted, model):
    """Caller must SHA-pin original search, interior ordering and model helpers."""

    class RootQuietSearch(base_search):
        def __init__(self, evaluator, *, ordering_weights=None, **kwargs):
            super().__init__(evaluator, **kwargs)
            self.ordering_weights = model.load_model(
                model.model_dict(
                    [0.0] * model.DIM if ordering_weights is None else ordering_weights
                )
            )

        def root_order(self, board, moves, scores):
            original = sorted(moves, key=lambda m: (-scores[m], m.uci()))
            if not any(self.ordering_weights):
                return original
            # Original score remains dominant; correction difference <=epsilon.
            legal = sorted(moves, key=lambda m: m.uci())
            probs = model.probabilities(
                self.ordering_weights, [model.features(board, m) for m in legal]
            )
            correction = {
                m: model.EPSILON * (prob - 1 / len(legal))
                for m, prob in zip(legal, probs, strict=True)
            }
            slots = [
                i
                for i, m in enumerate(original)
                if not board.is_capture(m) and not m.promotion and abs(scores[m]) <= 1.0
            ]
            alternatives = sorted(
                [original[i] for i in slots], key=lambda m: (-(scores[m] + correction[m]), m.uci())
            )
            for i, move in zip(slots, alternatives, strict=True):
                original[i] = move
            return original

        def search(self, position):
            if not any(self.ordering_weights):
                # Exact original full packet; no feature work/copy/order/RNG changes.
                return super().search(position)
            board = position.copy(stack=True)
            self.nodes = self.evaluations = 0
            self.consume()
            terminal = self.terminal(board, 0)
            if terminal is not None:
                return result_type(None, terminal, self.nodes, 0, 0, 0)
            moves = sorted(board.legal_moves, key=lambda m: m.uci())
            if self.node_budget < len(moves) + 1:
                raise ValueError("node budget cannot cover every legal root action")
            scores = {}
            # Shallow full legal coverage remains original UCI order, not learned.
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
                    for move in self.root_order(board, moves, scores):
                        board.push(move)
                        try:
                            value = -self.negamax(board, depth - 1, -math.inf, -alpha, 1)
                        finally:
                            board.pop()
                        iteration[move] = value
                        if iteration_winner is None or value > alpha:
                            iteration_winner = move
                        alpha = max(alpha, value)
                except budget_exhausted:
                    break
                # Partial root passes cannot overwrite the last complete packet.
                scores = iteration
                winner = iteration_winner
                best, completed = alpha, depth
            return result_type(winner, best, self.nodes, self.evaluations, completed, len(moves))

    return RootQuietSearch
