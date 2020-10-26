import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST  # Training dataset
from torchvision import transforms


from gans.core.generator import Generator
from gans.core.discriminator import Discriminator, get_disc_loss


def test_disc_loss(max_tests=10):

    criterion = nn.BCEWithLogitsLoss()
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

    # Load MNIST dataset as tensors
    dataloader = DataLoader(
        MNIST(".", download=True, transform=transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=True,
    )

    for real, _ in dataloader:
        cur_batch_size = len(real)
        real = real.view(cur_batch_size, -1).to(device)

        ### Update discriminator ###
        # Zero out the gradient before backpropagation
        disc_opt.zero_grad()

        # Calculate discriminator loss
        disc_loss = get_disc_loss(
            gen, disc, criterion, real, cur_batch_size, z_dim, device
        )
        assert (disc_loss - 0.68).abs() < 0.05

        # Update gradients
        disc_loss.backward(retain_graph=True)

        # Check that they detached correctly
        assert gen.gen[0][0].weight.grad is None

        # Update optimizer
        old_weight = disc.disc[0][0].weight.data.clone()
        disc_opt.step()
        new_weight = disc.disc[0][0].weight.data

        # Check that some discriminator weights changed
        assert not torch.all(torch.eq(old_weight, new_weight))
        num_steps += 1
        if num_steps >= max_tests:
            break
