"""
Your First GAN
https://www.coursera.org/learn/build-basic-generative-adversarial-networks-gans/programming/RmA3D/lab


-*- Goal -*-
In this notebook, you're going to create your first generative adversarial
network (GAN) for this course! Specifically, you will build and train a GAN
that can generate hand-written images of digits (0-9). You will be using
PyTorch in this specialization, so if you're not familiar with this framework,
you may find the PyTorch documentation useful. The hints will also often
include links to relevant documentation.

-*- Learning Objectives -*-
Build the generator and discriminator components of a GAN from scratch.
Create generator and discriminator loss functions.
Train your GAN and visualize the generated images.

-*- Getting Started -*-
You will begin by importing some useful packages and the dataset you will use
to build and train your GAN. You are also provided with a visualizer function
to help you investigate the images your GAN will create.

-*- MNIST Dataset -*-
The training images your discriminator will be using is from a dataset called
MNIST. It contains 60,000 images of handwritten digits, from 0 to 9, like
these:

https://hjhfjdls.coursera-apps.org/notebooks/MnistExamples.png

You may notice that the images are quite pixelated -- this is because they are
all only 28 x 28! The small size of its images makes MNIST ideal for simple
training. Additionally, these images are also in black-and-white so only one
dimension, or "color channel", is needed to represent them (more on this later
in the course).

-*- Tensor -*-
You will represent the data using tensors. Tensors are a generalization of
matrices: for example, a stack of three matrices with the amounts of red,
green, and blue at different locations in a 64 x 64 pixel image is a tensor
with the shape 3 x 64 x 64.

Tensors are easy to manipulate and supported by PyTorch, the machine learning
library you will be using. Feel free to explore them more, but you can imagine
these as multi-dimensional matrices or vectors!

-*- Batches -*-
While you could train your model after generating one image, it is extremely
inefficient and leads to less stable training. In GANs, and in machine learning
in general, you will process multiple images per training step. These are
called batches.

This means that your generator will generate an entire batch of images and
receive the discriminator's feedback on each before updating the model. The
same goes for the discriminator, it will calculate its loss on the entire batch
of generated images as well as on the reals before the model is updated.

"""
import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

from torchvision import transforms
from torchvision.datasets import MNIST  # Training dataset
from torchvision.utils import make_grid

import matplotlib.pyplot as plt

torch.manual_seed(0)  # Set for testing purposes, please do not change!

from gans.models.gan import GAN
from gans.core.generator import Generator, get_generator_block, get_gen_loss, get_noise
from gans.core.discriminator import Discriminator, get_disc_loss


class YourFirstGan(GAN):
    def train(self):
        """
        Training
        Now you can put it all together! First, you will set your parameters:

        - criterion: the loss function
        - n_epochs: the number of times you iterate through the entire dataset
          when training
        - z_dim: the dimension of the noise vector
        - display_step: how often to display/visualize the images
        - batch_size: the number of images per forward/backward pass
        - lr: the learning rate
        - device: the device type, here using a GPU (which runs CUDA), not CPU
        """

        # Next, you will load the MNIST dataset as tensors using a dataloader.
        # Set your parameters
        criterion = nn.BCEWithLogitsLoss()
        n_epochs = 200
        z_dim = 64
        display_step = 500
        batch_size = 128
        lr = 0.00001

        # Load MNIST dataset as tensors
        dataloader = DataLoader(
            MNIST("./datasets/.", download=True, transform=transforms.ToTensor()),
            batch_size=batch_size,
            shuffle=True,
        )

        ### DO NOT EDIT ###
        device = "cpu"

        """
        Now, you can initialize your generator, discriminator, and optimizers.
        Note that each optimizer only takes the parameters of one particular
        model, since we want each optimizer to optimize only one of the models.
        """
        gen = Generator(z_dim).to(device)
        gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
        disc = Discriminator().to(device)
        disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

        """
        Before you train your GAN, you will need to create functions to
        calculate the discriminator's loss and the generator's loss. This is
        how the discriminator and generator will know how they are doing and
        improve themselves. Since the generator is needed when calculating the
        discriminator's loss, you will need to call .detach() on the generator
        result to ensure that only the discriminator is updated!

        Remember that you have already defined a loss function earlier
        (criterion) and you are encouraged to use torch.ones_like and
        torch.zeros_like instead of torch.ones or torch.zeros. If you use
        torch.ones or torch.zeros, you'll need to pass device=device to them.
        """
        # UNQ_C8 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)

        # GRADED FUNCTION:

        cur_step = 0
        mean_generator_loss = 0
        mean_discriminator_loss = 0
        test_generator = True  # Whether the generator should be tested
        gen_loss = False
        error = False
        for epoch in range(n_epochs):

            # Dataloader returns the batches
            for real, _ in tqdm(dataloader):
                cur_batch_size = len(real)

                # Flatten the batch of real images from the dataset
                real = real.view(cur_batch_size, -1).to(device)

                ### Update discriminator ###
                # Zero out the gradients before backpropagation
                disc_opt.zero_grad()

                # Calculate discriminator loss
                disc_loss = get_disc_loss(
                    gen, disc, criterion, real, cur_batch_size, z_dim, device
                )

                # Update gradients
                disc_loss.backward(retain_graph=True)

                # Update optimizer
                disc_opt.step()

                # For testing purposes, to keep track of the generator weights
                if test_generator:
                    old_generator_weights = gen.gen[0][0].weight.detach().clone()

                ### Update generator ###
                #     Hint: This code will look a lot like the discriminator updates!
                #     These are the steps you will need to complete:
                #       1) Zero out the gradients.
                #       2) Calculate the generator loss, assigning it to gen_loss.
                #       3) Backprop through the generator: update the gradients and optimizer.
                #### START CODE HERE ####
                gen_opt.zero_grad()
                gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim)
                gen_loss.backward()
                gen_opt.step()
                #### END CODE HERE ####

                # For testing purposes, to check that your code changes the generator weights
                if test_generator:
                    try:
                        assert lr > 0.0000002 or (
                            gen.gen[0][0].weight.grad.abs().max() < 0.0005
                            and epoch == 0
                        )
                        assert torch.any(
                            gen.gen[0][0].weight.detach().clone()
                            != old_generator_weights
                        )
                    except:
                        error = True
                        print("Runtime tests have failed")

                # Keep track of the average discriminator loss
                mean_discriminator_loss += disc_loss.item() / display_step

                # Keep track of the average generator loss
                mean_generator_loss += gen_loss.item() / display_step

                ### Visualization code ###
                if cur_step % display_step == 0 and cur_step > 0:
                    print(
                        f"Epoch {epoch}, step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}"
                    )
                    fake_noise = get_noise(cur_batch_size, z_dim, device=device)
                    fake = gen(fake_noise)
                    self.show_tensor_images(fake)
                    self.show_tensor_images(real)
                    mean_generator_loss = 0
                    mean_discriminator_loss = 0
                cur_step += 1

    def show_tensor_images(
        self, image_tensor, num_images: int = 25, size: tuple = (1, 28, 28)
    ):
        """
        Function for visualizing images: Given a tensor of images, number of
        images, and size per image, plots and prints the images in a uniform
        grid.
        """
        image_unflat = image_tensor.detach().cpu().view(-1, *size)
        image_grid = make_grid(image_unflat[:num_images], nrow=5)
        plt.imshow(image_grid.permute(1, 2, 0).squeeze())
        plt.show()
