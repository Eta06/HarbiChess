"""Independent role adapter for prior, classical18, PST residual, or E8 evaluator."""

from frozen_reference.value import load_classical
from pst_value import load_pst


class MixedValue:
    def __init__(self, role, *, model_path=None, e8_evaluator=None):
        self.role = role
        if role == "e8":
            if e8_evaluator is None or model_path is not None:
                raise ValueError("E8 role requires its pinned external evaluator only")
            self.evaluator = e8_evaluator
        elif role == "prior":
            if model_path is None or e8_evaluator is not None:
                raise ValueError("prior role requires a classical model file")
            self.evaluator = load_classical(model_path)
        elif role == "classical18":
            if model_path is None or e8_evaluator is not None:
                raise ValueError("classical18 role requires an 18-term model file")
            self.evaluator = load_classical(model_path)
        elif role == "pst":
            if model_path is None or e8_evaluator is not None:
                raise ValueError("PST role requires a versioned PST model file")
            self.evaluator = load_pst(model_path)
        else:
            raise ValueError("unknown mixed arena role")

    def __call__(self, board):
        return self.evaluator(board)
