"""Only quiet tie-group ordering changes. Inherit all original search code."""

from model import features, load_model, model_dict, quiet, score


def search_type(base_search):
    """Caller must SHA-pin original module before passing BudgetSearch."""

    class QuietSearch(base_search):
        def __init__(self, evaluator, *, ordering_weights=None, **kwargs):
            super().__init__(evaluator, **kwargs)
            self.ordering_weights = [0.0] * 268 if ordering_weights is None else ordering_weights
            self.ordering_weights = load_model(model_dict(self.ordering_weights))

        def ordered(self, board, moves):
            original = base_search.ordered(board, moves)
            if not any(self.ordering_weights):
                return original  # exact baseline order and no added feature work
            positions = [i for i, m in enumerate(original) if quiet(board, m)]
            alternatives = sorted(
                [original[i] for i in positions],
                key=lambda m: (-score(self.ordering_weights, features(board, m)), m.uci()),
            )
            for i, move in zip(positions, alternatives, strict=True):
                original[i] = move
            return original

    return QuietSearch
