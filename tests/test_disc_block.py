import pytest
import torch


from gans.core.discriminator import get_discriminator_block


@pytest.mark.parametrize("in_features,out_features", [(25, 12), (15, 28)])
def test_disc_block(in_features, out_features, num_test=10000):
    # Verify the discriminator block function
    block = get_discriminator_block(in_features, out_features)

    # Check there are two parts
    assert len(block) == 2
    test_input = torch.randn(num_test, in_features)
    test_output = block(test_input)

    # Check that the shape is right
    assert tuple(test_output.shape) == (num_test, out_features)

    # Check that the LeakyReLU slope is about 0.2
    assert -test_output.min() / test_output.max() > 0.1
    assert -test_output.min() / test_output.max() < 0.3
    assert test_output.std() > 0.3
    assert test_output.std() < 0.5
