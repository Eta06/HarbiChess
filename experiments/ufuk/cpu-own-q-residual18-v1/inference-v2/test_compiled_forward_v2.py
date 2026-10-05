import math

import numpy as np
import pytest
from _ownq_forward18_v2 import forward
from compiled_evaluator_v2 import CompiledEvaluator18


class Model:
    def __init__(self):
        self.params = {
            "w1": np.zeros((18, 32)),
            "b1": np.zeros(32),
            "w2": np.zeros((32, 1)),
            "b2": np.zeros(1),
        }


def make(raw, model=None):
    return CompiledEvaluator18(
        model or Model(),
        classical_features=lambda _: tuple(raw),
        prior_weights=[1.0] * 18,
        feature_scales=[1.0] * 18,
        prior_score_scale=600.0,
    )


def test_python312_cancellation_preserves_original_prior_exactly():
    raw = [1e16, 1.0, -1e16] + [0.0] * 15
    assert sum(raw) == 1.0
    value = make(raw)
    assert value.nonterminal(None).hex() == math.tanh(sum(raw) / 600.0).hex()


def test_inference_snapshot_is_immutable_after_source_model_changes():
    model = Model()
    value = make([1.0] * 18, model)
    before = value.nonterminal(None)
    model.params["b2"][0] = 20.0
    assert value.nonterminal(None).hex() == before.hex()


def test_compiled_boundary_rejects_bad_shape_and_nonfinite_inputs():
    value = make([0.0] * 18)
    with pytest.raises(ValueError):
        forward([0.0] * 17, value.payload, 0.0)
    with pytest.raises(ValueError):
        forward([float("nan")] + [0.0] * 17, value.payload, 0.0)
    with pytest.raises(ValueError):
        forward([0.0] * 18, b"bad", 0.0)
    with pytest.raises(ValueError):
        forward([0.0] * 18, value.payload, float("inf"))
