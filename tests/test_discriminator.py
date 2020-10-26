import pytest

import torch

from gans.core.discriminator import Discriminator, get_disc_loss
from gans.core.generator import Generator, get_noise


@pytest.mark.parametrize("z_dim,hidden_dim", [(5, 10), (20, 8)])
def test_discriminator(z_dim, hidden_dim, num_test=100):
    # Verify the discriminator class

    disc = Discriminator(z_dim, hidden_dim).get_disc()

    # Check there are three parts
    assert len(disc) == 4

    # Check the linear layer is correct
    test_input = torch.randn(num_test, z_dim)
    test_output = disc(test_input)
    assert tuple(test_output.shape) == (num_test, 1)

    # Make sure there's no sigmoid
    assert test_input.max() > 1
    assert test_input.min() < -1

