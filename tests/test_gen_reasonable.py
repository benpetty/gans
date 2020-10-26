import torch
from torch import nn

from gans.core.generator import Generator, get_gen_loss
from gans.core.discriminator import Discriminator


def test_gen_reasonable(num_images=10):
    # Don't use explicit casts to cuda - use the device argument
    import inspect, re

    lines = inspect.getsource(get_gen_loss)
    assert (re.search(r"to\(.cuda.\)", lines)) is None
    assert (re.search(r"\.cuda\(\)", lines)) is None

    z_dim = 64
    gen = torch.zeros_like
    disc = nn.Identity()
    criterion = torch.mul  # Multiply
    gen_loss_tensor = get_gen_loss(gen, disc, criterion, num_images, z_dim, "cpu")

    assert torch.all(torch.abs(gen_loss_tensor) < 1e-5)
    # Verify shape. Related to gen_noise parametrization
    assert tuple(gen_loss_tensor.shape) == (num_images, z_dim)

    gen = torch.ones_like
    disc = nn.Identity()
    criterion = torch.mul  # Multiply
    real = torch.zeros(num_images, 1)
    gen_loss_tensor = get_gen_loss(gen, disc, criterion, num_images, z_dim, "cpu")
    assert torch.all(torch.abs(gen_loss_tensor - 1) < 1e-5)
    # Verify shape. Related to gen_noise parametrization
    assert tuple(gen_loss_tensor.shape) == (num_images, z_dim)
