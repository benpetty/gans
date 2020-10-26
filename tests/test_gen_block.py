import pytest

import torch
from torch import nn

from gans.core.generator import get_generator_block


@pytest.mark.parametrize("in_features,out_features", [(25, 12), (15, 28)])
def test_gen_block(in_features, out_features, num_test=1000):
    """
    Verify the generator block function.
    """
    block = get_generator_block(in_features, out_features)

    # Check the three parts
    assert len(block) == 3
    assert type(block[0]) == nn.Linear
    assert type(block[1]) == nn.BatchNorm1d
    assert type(block[2]) == nn.ReLU

    # Check the output shape
    test_input = torch.randn(num_test, in_features)
    test_output = block(test_input)
    assert tuple(test_output.shape) == (num_test, out_features)
    assert test_output.std() > 0.55
    assert test_output.std() < 0.65
