"""File to create the autoencoder model in PyTorch. Based on the code for Sindy Autoencoder by Kathleen Champion."""

# Libraries
import torch
from itertools import combinations
import numpy as np


class Autoencoder(torch.nn.Module):
    def __init__(self, params):
        """Autoencoder Initializer
        Args:
            params (dict): Parameters for the model
        """
        super(Autoencoder, self).__init__()

        # Parameters
        self.params = params
        # Basic Ones
        self.input_dim = params['input_dim']
        self.hidden_dims = params['widths']
        self.latent_dim = params['latent_dim']
        self.activation = params['activation']
        # Loss params
        self.lambda_0 = params['loss_weight_decoder']
        self.lambda_1 = params['loss_weight_sindy_x']
        self.lambda_2 = params['loss_weight_sindy_z']
        self.lambda_3 = params['loss_weight_sindy_regularization']
        # Sindy params
        self.poly_order = params['poly_order']
        self.include_sine = params['include_sine']
        self.coeff_init = params['coefficient_initialization']
        self.library_dim = library_size(self.latent_dim, self.poly_order, self.include_sine)
        self.sindy_coefficients = None

        # Encoder
        encoder_layers = [torch.nn.Linear(self.input_dim, self.hidden_dims[0])]
        for i in range(len(self.hidden_dims)-1):
            if self.activation != "linear":
                encoder_layers.append(self.activation)
            encoder_layers.append(torch.nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))
        encoder_layers.append(torch.nn.Linear(self.hidden_dims[-1], self.latent_dim))

        self.encoder = torch.nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = [torch.nn.Linear(self.latent_dim, self.hidden_dims[-1])]
        for i in range(len(self.hidden_dims)-1, 0, -1):
            if self.activation != "linear":
                decoder_layers.append(self.activation)
            decoder_layers.append(torch.nn.Linear(self.hidden_dims[i], self.hidden_dims[i-1]))

        decoder_layers.append(torch.nn.Linear(self.hidden_dims[0], self.input_dim))

        self.decoder = torch.nn.Sequential(*decoder_layers)

    def forward(self, state):
        """Forward pass for the autoencoder
        Args:
            state (torch.Tensor): Input data (x, dx, ddx=None)
        Returns:
            torch.Tensor: Reconstructed input data"""
        # Get initial data
        if self.params['model_order'] == 2:
            self.x, self.dx, self.ddx = state
        else:
            self.x, self.dx = state
            self.ddx = None

        # Autoencode x
        # Encode to latent space
        self.z = self.encoder(self.x)
        self.x_hat = self.decoder(self.z)

        # Encode derivatives
        self.dz, self.ddz = self.compute_derivatives(self.x, self.dx, self.ddx)
        
        # Transform using sindy and decode
        if self.ddz is not None:
            self.sindy_ddz = self.sindy(secondOrder=True)
            self.dx_hat, self.ddx_hat = self.compute_derivatives_z2x(self.z, self.dz, self.sindy_ddz)
            return self.x_hat, self.dx_hat, self.ddx_hat
        else:
            self.sindy_dz = self.sindy()
            self.dx_hat, self.ddx_hat = self.compute_derivatives_z2x(self.z, self.sindy_dz)
            return self.x_hat, self.dx_hat
        

    def compute_derivatives_x2z(self, x, dx, ddx=None):
        """Compute the derivatives of the latent state
        Args:
            x (torch.Tensor): Input data
            dx (torch.Tensor): First derivative of the input data
            ddx (torch.Tensor): Second derivative of the input data
        Returns:
            torch.Tensor: Derivatives of the latent state"""

        first_grads = torch.autograd.grad(self.encode(x), x, grad_outputs=torch.ones_like(self.encode(x)))[0]
        dz = first_grads * dx
        if ddx is not None:
            second_grads = torch.autograd.grad(first_grads, x, grad_outputs=torch.ones_like(first_grads))[0]
            ddz = second_grads * ddx
        else:
            ddz = None
        return dz, ddz

    def compute_derivatives_z2x(self, z, dz, ddz=None):
        """Compute the derivatives of the input data
        Args:
            z (torch.Tensor): Latent state
            dz (torch.Tensor): First derivative of the latent state
            ddz (torch.Tensor): Second derivative of the latent state
        Returns:
            torch.Tensor: Derivatives of the input data"""

        first_grads = torch.autograd.grad(self.decode(z), z, grad_outputs=torch.ones_like(self.decode(z)))[0]
        dx = first_grads * dz
        if ddz is not None:
            second_grads = torch.autograd.grad(first_grads, z, grad_outputs=torch.ones_like(first_grads))[0]
            ddx = second_grads * ddz
        else:
            ddx = None
        return dx, ddx

    def sindy(self, secondOrder=False):
        """Compute the SINDy model. Here what we do is have z, dz and possibly ddz. What it will do is try to find a function that takes z, (sometimes dz) to predict ddz/dz
        Returns:
            torch.Tensor: SINDy model prediction"""
        
        # Get Theta
        if secondOrder:
            Theta = sindy_library_torch_order2(self.z, self.dz, self.latent_dim, self.params['poly_order'], self.include_sine)
        # Initialize coefficients

        # Check if we need coefficient mask

        # Predict sindy


    # Loss
    def custom_loss(self):
        """Custom loss function for the autoencoder
        Returns:
            torch.Tensor: Loss value
        """
        # Decoder loss
        loss_decoder = torch.nn.functional.mse_loss(self.x_hat, self.x)

        # SINDy loss x
        


        # Total loss
        loss = self.lambda_0 * loss_decoder \
               + self.lambda_1 * losses_sindy_x \
               + self.lambda_2 * losses_sindy_z \
               + self.lambda_3 * losses_sindy_regularization

        return loss





# Extra functions
def sindy_library_torch(z, latent_dim, poly_order, include_sine=False):
    """
    Build the SINDy library.

    Arguments:
        z - 2D torch tensor of the snapshots on which to build the library. Shape is number of
        time points by the number of state variables.
        latent_dim - Integer, number of state variable in z.
        poly_order - Integer, polynomial order to which to build the library. Max value is 5.
        include_sine - Boolean, whether or not to include sine terms in the library. Default False.

    Returns:
        2D torch tensor containing the constructed library. Shape is number of time points by
        number of library functions. The number of library functions is determined by the number
        of state variables of the input, the polynomial order, and whether or not sines are included.
    """
    library= [torch.ones(z.size(0))]

    for i in range(latent_dim):
        library.append(z[:,i])

    if poly_order > 1:
        for indices in combinations(range(latent_dim), 2):
            product = z[:, indices[0]] * z[:, indices[1]]
            library.append(product)

    if poly_order > 2:
        for indices in combinations(range(latent_dim), 3):
            product = z[:, indices[0]] * z[:, indices[1]] * z[:, indices[2]]
            library.append(product)

    if poly_order > 3:
        for indices in combinations(range(latent_dim), 4):
            product = z[:, indices[0]] * z[:, indices[1]] * z[:, indices[2]] * z[:, indices[3]]
            library.append(product)

    if poly_order > 4:
        for indices in combinations(range(latent_dim), 5):
            product = z[:, indices[0]] * z[:, indices[1]] * z[:, indices[2]] * z[:, indices[3]]* z[:, indices[4]]
            library.append(product)

    if include_sine:
        for i in range(latent_dim):
            library.append(torch.sin(z[:,i]))

    return torch.stack(library, dim=1)

def library_size():
    def library_size(latent_dim, poly_order, use_sine=False, include_constant=True):
    l = 0
    for k in range(poly_order+1):
        l += int(binom(latent_d+k-1,k))
    if use_sine:
        l += latent_dim
    if not include_constant:
        l -= 1
    return l


def sindy_library_torch_order2(z, dz, latent_dim, poly_order, include_sine=False):
    """
    Build the SINDy library for a second order system. This is essentially the same as for a first
    order system, but library terms are also built for the derivatives.
    """
    library = [torch.ones(z.size(0))]

    z_combined = torch.cat([z, dz], 1)

    for i in range(2*latent_dim):
        library.append(z_combined[:,i])

    if poly_order > 1:
        for i in range(2*latent_dim):
            for j in range(i,2*latent_dim):
                library.append(z_combined[:,i] * z_combined[:,j])

    if poly_order > 2:
        for i in range(2*latent_dim):
            for j in range(i,2*latent_dim):
                for k in range(j,2*latent_dim):
                    library.append(z_combined[:,i] * z_combined[:,j] * z_combined[:,k])

    if poly_order > 3:
        for i in range(2*latent_dim):
            for j in range(i,2*latent_dim):
                for k in range(j,2*latent_dim):
                    for p in range(k,2*latent_dim):
                        library.append(z_combined[:,i] * z_combined[:,j] * z_combined[:,k] * z_combined[:,p])

    if poly_order > 4:
        for i in range(2*latent_dim):
            for j in range(i,2*latent_dim):
                for k in range(j,2*latent_dim):
                    for p in range(k,2*latent_dim):
                        for q in range(p,2*latent_dim):
                            library.append(z_combined[:,i] * z_combined[:,j] * z_combined[:,k] * z_combined[:,p] * z_combined[:,q])

    if include_sine:
        for i in range(2*latent_dim):
            library.append(torch.sin(z_combined[:,i]))

    return torch.stack(library,dim=1)