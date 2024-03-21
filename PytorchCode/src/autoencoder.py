"""File to create the autoencoder model in PyTorch. Based on the code for Sindy Autoencoder by Kathleen Champion."""

# Libraries
import torch


# Full Network
def full_network(params):
    input_dim = params['input_dim']
    latent_dim = params['latent_dim']
    activation = params['activation']
    poly_order = params['poly_order']
    if 'include_sine' in params.keys():
        include_sine = params['include_sine']
    else:
        include_sine = False
    library_dim = params['library_dim']
    model_order = params['model_order']


# Define Loss
def define_loss(network, params):
    """Creates a loss function.
    Args:
        network: The neural network.
        params: The parameters for the neural network.
    Returns:
        loss: The loss function.
        losses:
        loss_refinement:
    """


# Linear Autoencoder
def linear_autoencoder(x, input_dim, latent_dim):


# Nonlinear Autoencoder
def nonlinear_autoencoder(x, )


# Build Network








