"""
## Generator
The first step is to build the generator component.

You will start by creating a function to make a single layer/block for the
generator's neural network. Each block should include a

[linear transformation](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
to map to another shape,

a [batch normalization](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)
for stabilization,

and finally a non-linear activation function
(you use a [ReLU here](https://pytorch.org/docs/master/generated/torch.nn.ReLU.html))
so the output can be transformed in complex ways.

You will learn more about activations and batch normalization later in the course.
"""

import torch
from torch import nn, randn


# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_generator_block
def get_generator_block(input_dim, output_dim) -> nn.Sequential:
    """
    Function for returning a block of the generator's neural network
    given input and output dimensions.

    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a generator neural network layer, with a linear transformation
        followed by a batch normalization and then a relu activation
    """
    return nn.Sequential(
        # Hint: Replace all of the "None" with the appropriate dimensions.
        # The documentation may be useful if you're less familiar with PyTorch:
        # https://pytorch.org/docs/stable/nn.html.
        #### START CODE HERE ####
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True),
        #### END CODE HERE ####
    )


"""
Now you can build the generator class. It will take 3 values:

- The noise vector dimension
- The image dimension
- The initial hidden dimension

Using these values, the generator will build a neural network with 5
layers/blocks. Beginning with the noise vector, the generator will apply
non-linear transformations via the block function until the tensor is mapped to
the size of the image to be outputted (the same size as the real images from
MNIST). You will need to fill in the code for final layer since it is different
than the others. The final layer does not need a normalization or activation
function, but does need to be scaled with a sigmoid function.

Finally, you are given a forward pass function that takes in a noise vector and
generates an image of the output dimension using your neural network.

-*- Optional hints for Generator -*-
The output size of the final linear transformation should be im_dim, but
remember you need to scale the outputs between 0 and 1 using the sigmoid
function.

nn.Linear and nn.Sigmoid will be useful here.
"""


# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: Generator
class Generator(nn.Module):
    """
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
          (MNIST images are 28 x 28 = 784 so that is your default)
        hidden_dim: the inner dimension, a scalar
    """

    def __init__(
        self, z_dim: int = 10, im_dim: int = 784, hidden_dim: int = 128
    ) -> None:
        super().__init__()

        # Build the neural network
        self.gen = nn.Sequential(
            get_generator_block(z_dim, hidden_dim),
            get_generator_block(hidden_dim, hidden_dim * 2),
            get_generator_block(hidden_dim * 2, hidden_dim * 4),
            get_generator_block(hidden_dim * 4, hidden_dim * 8),
            #### START CODE HERE ####
            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid(),
            #### END CODE HERE ####
        )

    def forward(self, noise):
        """
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        """
        return self.gen(noise)

    # Needed for grading
    def get_gen(self):
        """
        Returns:
            the sequential model
        """
        return self.gen


"""
-*- Noise -*-
To be able to use your generator, you will need to be able to create noise
vectors. The noise vector z has the important role of making sure the images
generated from the same class don't all look the same -- think of it as a
random seed. You will generate it randomly using PyTorch by sampling random
numbers from the normal distribution. Since multiple images will be processed
per pass, you will generate all the noise vectors at once.

Note that whenever you create a new tensor using torch.ones, torch.zeros, or
torch.randn, you either need to create it on the target device, e.g.
torch.ones(3, 3, device=device), or move it onto the target device using
torch.ones(3, 3).to(device). You do not need to do this if you're creating a
tensor by manipulating another tensor or by using a variation that defaults the
device to the input, such as torch.ones_like. In general, use torch.ones_like
and torch.zeros_like instead of torch.ones or torch.zeros where possible.

-*- Optional hint for get_noise -*-
You will probably find torch.randn useful here.
https://pytorch.org/docs/master/generated/torch.randn.html
"""

# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_noise
def get_noise(n_samples: int, z_dim: int, device="cpu"):
    """
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim),
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    """
    # NOTE: To use this on GPU with device='cuda', make sure to pass the device
    # argument to the function you use to generate the noise.
    #### START CODE HERE ####
    return randn((n_samples, z_dim), device=device)
    #### END CODE HERE ####


# UNQ_C7 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_gen_loss
def get_gen_loss(
    gen: Generator, disc, criterion, num_images: int, z_dim: int, device: str = "cpu",
):
    """
    Return the loss of the generator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare
               the discriminator's predictions to the ground truth reality of the images
               (e.g. fake = 0, real = 1)
        num_images: the number of images the generator should produce,
                which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        gen_loss: a torch scalar loss value for the current batch
    """
    #     These are the steps you will need to complete:
    #       1) Create noise vectors and generate a batch of fake images.
    #           Remember to pass the device argument to the get_noise function.
    #       2) Get the discriminator's prediction of the fake image.
    #       3) Calculate the generator's loss. Remember the generator wants
    #          the discriminator to think that its fake images are real
    #     *Important*: You should NOT write your own loss function here - use criterion(pred, true)!

    #### START CODE HERE ####
    noise = get_noise(num_images, z_dim).to(device)
    fake_images = gen(noise)
    fake_prediction = disc(fake_images)
    gen_loss = criterion(fake_prediction, torch.ones((num_images, 1)).to(device))
    #### END CODE HERE ####
    return gen_loss
