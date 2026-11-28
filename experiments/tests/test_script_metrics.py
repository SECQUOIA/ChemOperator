"""The scripts' relative-L2 metrics keep case and field axes."""

import math

import pytest
import torch

from experiments.cstr.fno import relative_l2
from experiments.q2d.common import relative_l2_per_case


def test_script_metrics_preserve_case_and_field_axes():
    # Transposed grids also exercise non-contiguous input tensors.
    target = torch.zeros(2, 2, 2, 3, dtype=torch.float64).transpose(-1, -2)
    target[:, 0] = 1.
    target[1, 0] = 10.
    prediction = target.clone()
    prediction[:, 0] -= 1.
    prediction[:, 1] = 1e-10
    cases, fields = relative_l2_per_case(prediction, target)
    torch.testing.assert_close(fields, torch.tensor([
        [1., math.sqrt(6) / 100], [0.1, math.sqrt(6) / 100],
    ], dtype=torch.float64))
    torch.testing.assert_close(cases, torch.tensor([1., 0.1], dtype=torch.float64))
    torch.testing.assert_close(relative_l2(prediction, target, {}), cases.mean())
    # The CSTR adapter must retain its 1e-8 safeguard, not the dtype default.
    zeros = torch.zeros(1, 1, dtype=torch.float64)
    assert relative_l2(zeros + 1e-8, zeros, {}).item() == pytest.approx(1.)
