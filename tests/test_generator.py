import pytest

import torch
from torch import nn

from gans.core.generator import Generator


@pytest.mark.parametrize("z_dim,im_dim,hidden_dim", [(5, 10, 20), (20, 8, 24)])
def test_generator(z_dim, im_dim, hidden_dim, num_test=10000):
    # Verify the generator class
    gen = Generator(z_dim, im_dim, hidden_dim).get_gen()

    # Check there are six modules in the sequential part
    assert len(gen) == 6
    test_input = torch.randn(num_test, z_dim)
    test_output = gen(test_input)

    # Check that the output shape is correct
    assert tuple(test_output.shape) == (num_test, im_dim)
    assert test_output.max() < 1, "Make sure to use a sigmoid"
    assert test_output.min() > 0, "Make sure to use a sigmoid"
    assert test_output.std() > 0.05, "Don't use batchnorm here"
    assert test_output.std() < 0.15, "Don't use batchnorm here"
