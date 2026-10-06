"""Endpoint-aware original versus identical learned/zero selective controllers."""

from pathlib import Path

from support import load

ORIGINAL_PATH = Path('/workspace/HarbiChess/experiments/ufuk/cpu-classical-own-v1/arena/search.py')
ORIGINAL_SHA = 'de53c14728a67b7772f18b396ac8ef5c35a144d4e4e616fec40099cd461a6670'
DIRECTORY = Path('/workspace/work/harbichess/cpu-selective-quiescence-proposal')


def BudgetSearch(evaluator, **kwargs):
    original = load(ORIGINAL_PATH, ORIGINAL_SHA, 'selective_original_search')
    if not hasattr(evaluator, 'selective_weights'):
        engine = original.BudgetSearch(evaluator, **kwargs)
        engine.controller_kind = 'original-prior-or-E8-search'
        return engine
    import sys

    sys.path.insert(0, str(DIRECTORY))
    try:
        # SHA constants are filled from the prospectively frozen producer, not new defaults.
        controller = load(DIRECTORY / 'search.py', SELECTIVE_SHA, 'selective_arena_controller')
        if Path(controller.features.__code__.co_filename).resolve() != DIRECTORY / 'model.py':
            raise ValueError('selective feature import origin differs')
        model = load(DIRECTORY / 'model.py', MODEL_SHA, 'selective_arena_model_check')
        if (model.load_model(model.model_dict(evaluator.selective_weights))
                != evaluator.selective_weights):
            raise ValueError('exact effort weights')
    finally:
        sys.path.pop(0)
    selected = controller.search_type(original.BudgetSearch, original.BudgetExhausted)

    class Receipted(selected):
        controller_kind = 'new-selective-Q-learned-or-zero'

        def search(self, board):
            self.extension_receipts.clear()
            return super().search(board)

    return Receipted(evaluator, weights=evaluator.selective_weights, **kwargs)


SELECTIVE_SHA = '993e9c2c6d949b3d8753935405b5c289b35249a108b0ef3f8f984ac55de94293'
MODEL_SHA = '14f5a1790f5e9518170a7d9bd490ba0aa61c001086286798a6d1315f00e0b9d2'
