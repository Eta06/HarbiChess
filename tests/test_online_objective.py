from dataclasses import fields, replace

import mlx.core as mx
import numpy as np
import pytest
import torch

from harbichess.training.mlx_online_objective import online_loss as mlx_loss
from harbichess.training.online_objective import (
    OnlineObjectiveConfig,
    OnlineObjectiveTargets,
    reference_online_loss,
)
from harbichess.training.torch_online_objective import online_loss as torch_loss


def targets():
    return OnlineObjectiveTargets(
        legal_masks=np.array([[True, True, False, True], [False, True, True, True]]),
        actions=np.array([0, 1]),
        advantages=np.array([0.6, -0.4]),
        importance=np.array([1.0, 0.5]),
        target_wdl=np.array([[1.0, 0.0, 0.0], [0.1, 0.3, 0.6]]),
        base_policy=np.array([[0.2, 0.5, 0.0, 0.3], [0.0, 0.1, 0.0, 0.9]]),
        base_wdl=np.array([[0.6, 0.0, 0.4], [0.2, 0.7, 0.1]]),
    )


def logits():
    return (
        np.array([[0.3, -0.2, 1000.0, 0.9], [1000.0, 0.2, 0.7, -0.1]]),
        np.array([[0.7, -0.1, -0.3], [0.2, 0.9, -0.7]]),
    )


def config():
    # Test constants, not a registered training hyperparameter choice.
    return OnlineObjectiveConfig(0.03, 0.2, 0.02)


def test_torch_and_actual_mlx_cpu_loss_and_gradients_match_independent_oracle():
    torch.set_num_threads(1)
    mx.set_default_device(mx.cpu)
    data = targets()
    p, v = logits()
    expected = reference_online_loss(p, v, data, config())
    pl = torch.tensor(p, dtype=torch.float32, requires_grad=True)
    vl = torch.tensor(v, dtype=torch.float32, requires_grad=True)
    actual = torch_loss(pl, vl, data, config())
    actual.total.backward()
    mp, mv = mx.array(p, dtype=mx.float32), mx.array(v, dtype=mx.float32)
    result = mlx_loss(mp, mv, data, config())
    value, gradients = mx.value_and_grad(
        lambda policy, wdl: mlx_loss(policy, wdl, data, config()).total, argnums=(0, 1)
    )(mp, mv)
    mx.eval(result, value, gradients)
    np.testing.assert_allclose([float(x.detach()) for x in actual], expected, atol=2e-6)
    np.testing.assert_allclose([float(x) for x in result], expected, atol=2e-6)
    assert float(value) == pytest.approx(expected.total, abs=2e-6)
    for actual_grad, expected_grad in zip(gradients, (pl.grad, vl.grad), strict=True):
        np.testing.assert_allclose(np.array(actual_grad), expected_grad.numpy(), atol=2e-6)
    assert pl.grad[0, 2] == pl.grad[1, 0] == 0


def test_float64_torch_gradients_match_finite_differences_of_reference():
    data = targets()
    arrays = logits()
    tensors = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in arrays]
    torch_loss(*tensors, data, config()).total.backward()
    delta = 1e-6
    for which, array in enumerate(arrays):
        expected = np.zeros_like(array)
        for index in np.ndindex(array.shape):
            plus, minus = [x.copy() for x in arrays], [x.copy() for x in arrays]
            plus[which][index] += delta
            minus[which][index] -= delta
            expected[index] = (
                reference_online_loss(*plus, data, config()).total
                - reference_online_loss(*minus, data, config()).total
            ) / (2 * delta)
        np.testing.assert_allclose(tensors[which].grad.numpy(), expected, atol=1e-9, rtol=1e-7)


def test_actor_gradient_rewards_positive_and_penalizes_negative_advantage():
    p, v = logits()
    policy = torch.tensor(p, dtype=torch.float64, requires_grad=True)
    value = torch.tensor(v, dtype=torch.float64, requires_grad=True)
    loss = torch_loss(policy, value, targets(), OnlineObjectiveConfig(0, 0, 0))
    loss.total.backward()
    assert policy.grad[0, 0] < 0  # Gradient descent raises the rewarded action.
    assert policy.grad[1, 1] > 0  # Gradient descent lowers the penalized action.
    assert torch.count_nonzero(value.grad) == 0


def test_zero_importance_removes_actor_and_target_value_but_preserves_base_anchors():
    data = replace(targets(), importance=np.zeros(2))
    result = reference_online_loss(*logits(), data, config())
    assert result.actor == result.value_target == 0
    assert result.policy_anchor > 0 and result.value_anchor > 0
    assert result.total == pytest.approx(
        config().policy_anchor_weight * result.policy_anchor
        + config().value_anchor_weight * result.value_anchor
    )


def test_chunk_losses_and_gradients_are_weighted_by_rows_not_number_of_chunks():
    original = targets()
    indices = [0, 1, 0]
    data = OnlineObjectiveTargets(
        **{field.name: getattr(original, field.name)[indices] for field in fields(original)}
    )
    p, v = (array[indices] for array in logits())
    complete = reference_online_loss(p, v, data, config())
    complete_p = torch.tensor(p, requires_grad=True)
    complete_v = torch.tensor(v, requires_grad=True)
    torch_loss(complete_p, complete_v, data, config()).total.backward()
    chunk_p = torch.tensor(p, requires_grad=True)
    chunk_v = torch.tensor(v, requires_grad=True)
    weighted = np.zeros(5)
    for start, stop in [(0, 2), (2, 3)]:
        subset = OnlineObjectiveTargets(
            **{field.name: getattr(data, field.name)[start:stop] for field in fields(data)}
        )
        weight = (stop - start) / len(indices)
        weighted += weight * np.array(
            reference_online_loss(p[start:stop], v[start:stop], subset, config())
        )
        (
            weight * torch_loss(chunk_p[start:stop], chunk_v[start:stop], subset, config()).total
        ).backward()
    np.testing.assert_allclose(weighted, complete, atol=1e-15)
    np.testing.assert_allclose(chunk_p.grad.numpy(), complete_p.grad.numpy(), atol=1e-15)
    np.testing.assert_allclose(chunk_v.grad.numpy(), complete_v.grad.numpy(), atol=1e-15)


def test_owned_constants_reject_mutation_and_have_no_external_array_alias():
    original = np.array([0.6, -0.4])
    data = replace(targets(), advantages=original)
    original[:] = 0
    np.testing.assert_array_equal(data.advantages, [0.6, -0.4])
    with pytest.raises(ValueError, match="read-only"):
        data.advantages[0] = 0


@pytest.mark.parametrize(
    "changes",
    [
        {"legal_masks": np.ones((2, 4), dtype=np.float32)},
        {"legal_masks": np.zeros((2, 4), dtype=np.bool_)},
        {"actions": np.array([2, 1])},
        {"actions": np.array([0.0, 1.0])},
        {"actions": np.array([-1, 1])},
        {"actions": np.array([0, 4])},
        {"advantages": np.array([0.6, np.nan])},
        {"advantages": np.array([1.1, 0.4])},
        {"importance": np.array([1.0, -0.2])},
        {"base_policy": np.array([[0.2, 0.4, 0.1, 0.3], [0.0, 0.1, 0.0, 0.9]])},
        {"base_wdl": np.array([[0.6, 0.0, 0.4], [0.2, 0.7, 0.2]])},
        {"target_wdl": np.ones((2, 4), dtype=np.float32) / 4},
    ],
)
def test_reject_invalid_constants_and_illegal_support(changes):
    with pytest.raises(ValueError):
        replace(targets(), **changes)


@pytest.mark.parametrize("coefficients", [(-0.1, 0.2, 0.3), (0.1, np.inf, 0.3)])
def test_reject_invalid_coefficients(coefficients):
    with pytest.raises(ValueError):
        OnlineObjectiveConfig(*coefficients)


def test_reject_nonfinite_logits_and_mismatched_shapes_in_every_implementation():
    mx.set_default_device(mx.cpu)
    p, v = logits()
    for bad in (np.full_like(p, np.nan), p[:, :3]):
        with pytest.raises(ValueError):
            reference_online_loss(bad, v, targets(), config())
        with pytest.raises(ValueError):
            torch_loss(torch.tensor(bad), torch.tensor(v), targets(), config())
        with pytest.raises(ValueError):
            mlx_loss(
                mx.array(bad, dtype=mx.float32), mx.array(v, dtype=mx.float32), targets(), config()
            )
