"""Contracts distinguishing per-case, per-field, and pooled relative errors."""

import math

import pytest
import torch

from chem_operator.experiments import StreamingRegressionMetrics
from chem_operator.models import relative_l2_loss


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("eps", [None, 1e-8, 1e-12])
def test_relative_l2_denominator_reduction_and_gradient(dtype, eps):
    floor = torch.finfo(dtype).eps if eps is None else eps
    target = torch.tensor([[0., 0.], [floor / 2, 0.], [3., 4.]], dtype=dtype)
    prediction = (target + torch.tensor([3., 4.], dtype=dtype)).requires_grad_()
    errors = relative_l2_loss(prediction, target, reduction="none", eps=eps)
    torch.testing.assert_close(errors, torch.tensor([5 / floor, 5 / floor, 1.], dtype=dtype))
    mean = relative_l2_loss(prediction, target, eps=eps)
    torch.testing.assert_close(mean, errors.mean())
    mean.backward()
    expected_gradient = torch.tensor([
        [1 / (5 * floor), 4 / (15 * floor)],
        [1 / (5 * floor), 4 / (15 * floor)],
        [1 / 25, 4 / 75],
    ], dtype=dtype)
    torch.testing.assert_close(prediction.grad, expected_gradient)


def test_mean_relative_l2_is_distinct_from_pooled_relative_l2():
    target = torch.tensor([[1.], [10.]])
    prediction = torch.tensor([[0.], [9.]])
    assert relative_l2_loss(prediction, target).item() == pytest.approx(0.55)
    pooled = StreamingRegressionMetrics()
    for predicted, reference in zip(prediction, target):
        pooled.update(predicted, reference)
    assert pooled.compute()["relative_l2"] == pytest.approx(math.sqrt(2 / 101))


def test_relative_l2_rejects_mismatched_shapes_and_unknown_reduction():
    with pytest.raises(ValueError, match="identical shapes"):
        relative_l2_loss(torch.ones(2, 3), torch.ones(2, 1))
    with pytest.raises(ValueError, match="reduction"):
        relative_l2_loss(torch.ones(2, 3), torch.ones(2, 3), reduction="pooled")
