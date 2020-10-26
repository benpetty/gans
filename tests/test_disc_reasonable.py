import torch
from torch import nn

from gans.core.discriminator import Discriminator, get_disc_loss
from gans.core.generator import Generator


def test_disc_reasonable(num_images=10):

    n_epochs = 200
    z_dim = 64
    display_step = 500
    batch_size = 128
    lr = 0.00001
    device = "cpu"

    z_dim = 64
    gen = Generator(z_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc = Discriminator().to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
    num_steps = 0

    # Don't use explicit casts to cuda - use the device argument
    import inspect, re

    lines = inspect.getsource(get_disc_loss)
    assert (re.search(r"to\(.cuda.\)", lines)) is None
    assert (re.search(r"\.cuda\(\)", lines)) is None

    z_dim = 64
    gen = torch.zeros_like
    disc = lambda x: x.mean(1)[:, None]
    criterion = torch.mul  # Multiply
    real = torch.ones(num_images, z_dim)
    disc_loss = get_disc_loss(gen, disc, criterion, real, num_images, z_dim, "cpu")
    assert torch.all(torch.abs(disc_loss.mean() - 0.5) < 1e-5)

    gen = torch.ones_like
    criterion = torch.mul  # Multiply
    real = torch.zeros(num_images, z_dim)
    assert torch.all(
        torch.abs(get_disc_loss(gen, disc, criterion, real, num_images, z_dim, "cpu"))
        < 1e-5
    )

    gen = lambda x: torch.ones(num_images, 10)
    disc = lambda x: x.mean(1)[:, None] + 10
    criterion = torch.mul  # Multiply
    real = torch.zeros(num_images, 10)
    assert torch.all(
        torch.abs(
            get_disc_loss(gen, disc, criterion, real, num_images, z_dim, "cpu").mean()
            - 5
        )
        < 1e-5
    )

    gen = torch.ones_like
    disc = nn.Linear(64, 1, bias=False)
    real = torch.ones(num_images, 64) * 0.5
    disc.weight.data = torch.ones_like(disc.weight.data) * 0.5
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
    criterion = lambda x, y: torch.sum(x) + torch.sum(y)
    disc_loss = get_disc_loss(
        gen, disc, criterion, real, num_images, z_dim, "cpu"
    ).mean()
    disc_loss.backward()
    assert torch.isclose(torch.abs(disc.weight.grad.mean() - 11.25), torch.tensor(3.75))
