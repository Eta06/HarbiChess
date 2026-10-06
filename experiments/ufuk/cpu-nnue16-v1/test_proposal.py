"""Synthetic sparse/C/storage tests; no actual chess-label training/inference."""

import copy
import math
from types import SimpleNamespace

import pytest
import torch
from evaluator import AuthoritativePrior, Evaluator
from model import FEATURE_SCHEMA, NNUE16, feature_indices, tensor_batch
from native import MATH, Learner, bits_equal


def contract():
    return dict(
        phase="synthetic-test", updates=8, seed=17, math=MATH, feature_schema=FEATURE_SCHEMA
    )


def synthetic_bits(i):
    bits = [0] * 12
    bits[5], bits[11] = 1 << 4, 1 << 60
    bits[i % 5] = (1 << (8 + i)) | (1 << (32 + i))
    bits[6 + i % 5] = 1 << (48 + i)
    return bits


def test_geometry_and_colour_rank_mapping():
    for i in range(12):
        bits = synthetic_bits(i)
        ids = feature_indices(bits, 4, True)
        flipped = [
            sum(1 << (s ^ 56) for s in range(64) if b & (1 << s)) for b in bits[6:] + bits[:6]
        ]
        assert feature_indices(flipped, 4 ^ 56, False) == ids
        assert ids == sorted(ids)
    assert sum(p.numel() for p in NNUE16().parameters()) == 196625


def test_c_residual_parity_nonzero_and_zero():
    import _kingbucket16

    torch.manual_seed(17)
    model = NNUE16()
    with torch.no_grad():
        model.head.weight.normal_(0, 0.03)
        model.head.bias.fill_(0.012)
    maximum = 0.0
    for i in range(12):
        for mover in [True, False]:
            bits = synthetic_bits(i)
            king = 4 if mover else 60
            ids = feature_indices(bits, king, mover)
            batch = tensor_batch([dict(indices=ids, prior_logit=0.0, target=0.0)])
            expected = model.residual(batch[0], batch[1]).item()
            evaluator = Evaluator(model.state_dict(), prior=None, compiled=_kingbucket16)
            actual = _kingbucket16.forward(bits, king, mover, evaluator.payload)
            maximum = max(maximum, abs(actual - expected))
            assert abs(actual - expected) <= 1e-12
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
    evaluator = Evaluator(model.state_dict(), prior=None, compiled=_kingbucket16)
    assert _kingbucket16.forward(synthetic_bits(0), 4, True, evaluator.payload).hex() == (0.0).hex()
    assert maximum < 1e-12


def test_literal_zero_prior_delegation_no_board_or_c_work():
    expected = -0.0
    calls = []
    prior = SimpleNamespace(nonterminal=lambda b: calls.append(b) or expected)
    compiled = SimpleNamespace(forward=lambda *args: pytest.fail("zero must bypass C"))
    evaluator = Evaluator(NNUE16().state_dict(), prior=prior, compiled=compiled)
    sentinel = object()
    assert evaluator.nonterminal(sentinel).hex() == expected.hex()
    assert calls == [sentinel]


def test_authoritative_prior_uses_python_builtin_sum_exactly():
    weights = tuple([1.0] * 18)
    module = SimpleNamespace(
        PRIOR=weights, SCALE=600, features=lambda _: [1e16, 1.0, -1e16] + [0.0] * 15
    )
    original = SimpleNamespace(
        weights=weights,
        theta=[0.0] * 18,
        nonterminal=lambda _: math.tanh(sum(module.features(None)) / 600),
    )
    prior = AuthoritativePrior(module, original)
    assert prior.logit(None).hex() == (sum(module.features(None)) / 600).hex()
    assert prior.nonterminal(None).hex() == original.nonterminal(None).hex()


@pytest.mark.parametrize("mutation", ["counter", "baseline", "rng", "dtype"])
def test_native_corruption_rejected(mutation):
    learner = Learner(contract())
    state = copy.deepcopy(learner.native())
    if mutation == "counter":
        state["step"] = 9
    elif mutation == "baseline":
        state["baseline"]["head.bias"][0] = 1.0
    elif mutation == "rng":
        state["sampler_rng"] = ("invalid",)
    else:
        state["model"]["head.bias"] = state["model"]["head.bias"].float()
    with pytest.raises((ValueError, TypeError)):
        Learner(contract(), state=state)


def test_storage_comparison_preserves_signed_zero():
    assert not bits_equal(torch.tensor([0.0]), torch.tensor([-0.0]))
    assert bits_equal(torch.tensor([1.0]), torch.tensor([1.0]))


def test_own_phase_anonymous_initializer_rejected():
    own = contract() | dict(phase="own-learning", updates=64)
    with pytest.raises(ValueError):
        Learner(own)


def test_named_weights_bridge_is_fresh_adam_not_native_resume(tmp_path):
    from native import sha

    teacher = Learner(contract())
    teacher_contract = dict(
        phase="teacher-bootstrap",
        updates=256,
        seed=17,
        math=MATH,
        feature_schema=FEATURE_SCHEMA,
        prior_helper_sha256="synthetic-prior",
    )
    # Structural synthetic bootstrap fixture, NOT256updates or teacher fit proof.
    bootstrap = dict(
        schema="own-kingbucket-nnue16-model-v1",
        contract=teacher_contract,
        model=copy.deepcopy(teacher.model.state_dict()),
    )
    bootstrap["model"]["head.bias"][0] = 0.125
    path = tmp_path / "synthetic-bootstrap.pt"
    torch.save(bootstrap, path)
    own_contract = dict(
        phase="own-learning",
        updates=64,
        seed=23,
        math=MATH,
        feature_schema=FEATURE_SCHEMA,
        prior_helper_sha256="synthetic-prior",
        bootstrap_candidate_path=str(path),
        bootstrap_candidate_sha256=sha(path),
    )
    own = Learner(own_contract, weights_initializer=bootstrap["model"])
    assert own.step == 0 and not own.optimizer.state
    assert bits_equal(own.baseline, bootstrap["model"])
    state = own.native()
    restored = Learner(own_contract, state=state)
    assert bits_equal(restored.native(), state)
    corrupt = copy.deepcopy(state)
    corrupt["baseline"]["head.bias"][0] += 0.01
    with pytest.raises(ValueError):
        Learner(own_contract, state=corrupt)
