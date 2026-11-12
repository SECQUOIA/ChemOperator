"""POD checkpoint compatibility and trainer prediction round trips."""

import numpy as np
import pytest
import torch

from chem_operator.experiments import DeepONetTrainer, RunContext
from chem_operator.models import DeepONetTrainingConfig, PODTransform


@pytest.fixture
def legacy_pod_state():
    # The keys, tensors, and list shape written by both previous serializers.
    return {
        "mean": torch.tensor([1., 2., 3.]),
        "basis": torch.tensor([[1., 0.], [0., 1.], [0., 0.]]),
        "explained_variance_ratio": torch.tensor([0.75, 0.25]),
        "cumulative_explained_variance": 1.,
        "output_shape": [3, 1],
    }


@pytest.mark.parametrize("arrays", ["tensor", "numpy"])
def test_pod_legacy_state_and_weights_only_round_trip(tmp_path, legacy_pod_state, arrays):
    state = {
        key: value.numpy() if arrays == "numpy" and isinstance(value, torch.Tensor) else value
        for key, value in legacy_pod_state.items()
    }
    pod = PODTransform.from_state_dict(state)
    coefficients = np.array([[2., -1.], [-1., 3.]], dtype=np.float32)
    expected = np.array([[[3.], [1.], [3.]], [[0.], [5.], [3.]]], dtype=np.float32)
    np.testing.assert_array_equal(pod.decode(coefficients), expected)
    np.testing.assert_array_equal(pod.encode(expected), coefficients)

    checkpoint = tmp_path / "pod.pt"
    torch.save(pod.state_dict(), checkpoint)
    saved = torch.load(checkpoint, weights_only=True)
    assert saved.keys() == legacy_pod_state.keys()
    for key, value in legacy_pod_state.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(saved[key], value, rtol=0, atol=0)
        else:
            assert saved[key] == value
    restored = PODTransform.from_state_dict(saved)
    torch.testing.assert_close(restored.decode_tensor(torch.from_numpy(coefficients)), torch.from_numpy(expected))
    snapshot = pod.state_dict()
    pod.mean[:] = 100
    torch.testing.assert_close(snapshot["mean"], torch.tensor([1., 2., 3.]))


@pytest.mark.parametrize("use_pod", [False, True], ids=["direct", "pod"])
def test_deeponet_checkpoint_preserves_predictions(tmp_path, legacy_pod_state, use_pod):
    class Trajectories(list):
        format = "cartesian_product"

    trunk = torch.linspace(0., 1., 3)[:, None]
    data = Trajectories([
        {"branch": torch.tensor([float(index), 1.]), "trunk": trunk,
         "target": torch.tensor([[1. + index], [2. - index], [3.]]),
         "coordinate": trunk, "labels": ("u",)}
        for index in range(4)
    ])
    context = RunContext.create(tmp_path, problem="synthetic", model="deeponet", run_id="pod", seed=7, device="cpu")
    config = DeepONetTrainingConfig(epochs=1, width=8, latent_width=2, batch_size=2)
    pod = PODTransform.from_state_dict(legacy_pod_state) if use_pod else None
    trainer = DeepONetTrainer(config, pod=pod)
    outcome = trainer.fit(data, data, context)
    assert outcome.metadata["checkpoint_reload_verified"]
    expected = trainer.predict(data, context)
    assert expected.shape == (4, 3, 1)
    # Restore the previous POD payload format inside the full trainer checkpoint.
    payload = torch.load(outcome.checkpoint, weights_only=True)
    payload["pod"] = legacy_pod_state if use_pod else None
    torch.save(payload, outcome.checkpoint)
    restored = DeepONetTrainer(config)
    restored.load_checkpoint(outcome.checkpoint, context)
    torch.testing.assert_close(restored.predict(data, context), expected)
