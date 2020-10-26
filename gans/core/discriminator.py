"""Discriminator
The second component that you need to construct is the discriminator. As with
the generator component, you will start by creating a function that builds a
neural network block for the discriminator.

Note: You use leaky ReLUs to prevent the "dying ReLU" problem, which refers to
the phenomenon where the parameters stop changing due to consistently negative
values passed to a ReLU, which result in a zero gradient. You will learn more
about this in the following lectures!

[https://hjhfjdls.coursera-apps.org/notebooks/relu-graph.png]
[https://hjhfjdls.coursera-apps.org/notebooks/lrelu-graph.png]

"""

import torch
from torch import nn
from gans.core.generator import Generator, get_noise


# UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_discriminator_block
def get_discriminator_block(input_dim, output_dim):
    """
    Discriminator Block
    Function for returning a neural network of the discriminator given input and output dimensions.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a discriminator neural network layer, with a linear transformation
          followed by an nn.LeakyReLU activation with negative slope of 0.2
          (https://pytorch.org/docs/master/generated/torch.nn.LeakyReLU.html)
    """
    return nn.Sequential(
        #### START CODE HERE ####
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(0.2),
        #### END CODE HERE ####
    )


"""
Now you can use these blocks to make a discriminator! The discriminator class
holds 2 values:

- The image dimension
- The hidden dimension

The discriminator will build a neural network with 4 layers. It will start with
the image tensor and transform it until it returns a single number (1-dimension
tensor) output. This output classifies whether an image is fake or real. Note
that you do not need a sigmoid after the output layer since it is included in
the loss function. Finally, to use your discrimator's neural network you are
given a forward pass function that takes in an image tensor to be classified.
"""

# UNQ_C5 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: Discriminator
class Discriminator(nn.Module):
    """
    Discriminator Class
    Values:
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
            (MNIST images are 28x28 = 784 so that is your default)
        hidden_dim: the inner dimension, a scalar
    """

    def __init__(self, im_dim: int = 784, hidden_dim: int = 128) -> None:
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            get_discriminator_block(im_dim, hidden_dim * 4),
            get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            get_discriminator_block(hidden_dim * 2, hidden_dim),
            # Hint: You want to transform the final output into a single value,
            #       so add one more linear map.
            #### START CODE HERE ####
            nn.Linear(hidden_dim, 1),
            #### END CODE HERE ####
        )

    def forward(self, image):
        """
        Function for completing a forward pass of the discriminator: Given an image tensor,
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_dim)
        """
        return self.disc(image)

    # Needed for grading
    def get_disc(self):
        """
        Returns:
            the sequential model
        """
        return self.disc


# UNQ_C6 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_disc_loss
def get_disc_loss(
    gen: Generator,
    disc: Discriminator,
    criterion,
    real,
    num_images: int,
    z_dim: int,
    device: str = "cpu",
):
    """
    Return the loss of the discriminator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare
               the discriminator's predictions to the ground truth reality of the images
               (e.g. fake = 0, real = 1)
        real: a batch of real images
        num_images: the number of images the generator should produce,
                which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        disc_loss: a torch scalar loss value for the current batch
    """
    #     These are the steps you will need to complete:
    #       1) Create noise vectors and generate a batch (num_images) of fake images.
    #            Make sure to pass the device argument to the noise.
    #       2) Get the discriminator's prediction of the fake image
    #            and calculate the loss. Don't forget to detach the generator!
    #            (Remember the loss function you set earlier -- criterion. You need a
    #            'ground truth' tensor in order to calculate the loss.
    #            For example, a ground truth tensor for a fake image is all zeros.)
    #       3) Get the discriminator's prediction of the real image and calculate the loss.
    #       4) Calculate the discriminator's loss by averaging the real and fake loss
    #            and set it to disc_loss.
    #     Note: Please do not use concatenation in your solution. The tests are being updated to
    #           support this, but for now, average the two losses as described in step (4).
    #     *Important*: You should NOT write your own loss function here - use criterion(pred, true)!

    #### START CODE HERE ####
    noise = get_noise(num_images, z_dim, device)
    fake_image = gen(noise).detach()
    fake_pred = disc(fake_image)
    real_pred = disc(real)
    loss_gen = criterion(fake_pred, torch.zeros((num_images, 1)).to(device))
    loss_real = criterion(real_pred, torch.ones((num_images, 1)).to(device))
    disc_loss = (loss_gen + loss_real) / 2
    return disc_loss
    #### END CODE HERE ####
