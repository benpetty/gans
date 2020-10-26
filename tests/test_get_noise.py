import pytest

import torch
from torch import nn

from gans.core.generator import get_noise


@pytest.mark.parametrize(
    "n_samples,z_dim,device",
    [
        (1000, 100, "cpu"),
        # (1000, 32, "cuda")
    ],
)
def test_get_noise(n_samples, z_dim, device):
    # Verify the noise vector function
    noise = get_noise(n_samples, z_dim, device)

    # Make sure a normal distribution was used
    assert tuple(noise.shape) == (n_samples, z_dim)
    assert torch.abs(noise.std() - torch.tensor(1.0)) < 0.01
    assert str(noise.device).startswith(device)
