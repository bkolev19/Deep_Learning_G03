"""File to create the autoencoder model in PyTorch. Based on the code for Sindy Autoencoder by Kathleen Champion."""

# Libraries
import torch


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
        input_dim = params['input_dim']
        hidden_dims = params['widths']
        latent_dim = params['latent_dim']
        activation = params['activation']

        # Encoder
        encoder_layers = [torch.nn.Linear(input_dim, hidden_dims[0])]
        for i in range(len(hidden_dims)-1):
            if activation != "linear":
                encoder_layers.append(activation)
            encoder_layers.append(torch.nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        encoder_layers.append(torch.nn.Linear(hidden_dims[-1], latent_dim))

        self.encoder = torch.nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = [torch.nn.Linear(latent_dim, hidden_dims[-1])]
        for i in range(len(hidden_dims)-1, 0, -1):
            if activation != "linear":
                decoder_layers.append(activation)
            decoder_layers.append(torch.nn.Linear(hidden_dims[i], hidden_dims[i-1]))

        decoder_layers.append(torch.nn.Linear(hidden_dims[0], input_dim))

        self.decoder = torch.nn.Sequential(*decoder_layers)

    def forward(self, state):
        """Forward pass for the autoencoder
        Args:
            state (torch.Tensor): Input data (x, dx, ddx=None)
        Returns:
            torch.Tensor: Reconstructed input data"""
        self.x, self.dx, self.ddx = state
        self.z = self.encoder(self.x)
        self.dz, self.ddz = self.compute_derivatives(self.x, self.dx, self.ddx)


        self.x_hat = self.decoder(self.z)
        return self.x_hat

    def compute_derivatives(self, x, dx, ddx=None):
        """Compute the derivatives of the latent state
        Args:
            state (torch.Tensor): Input data (x, dx, ddx=None)
        Returns:
            torch.Tensor: Derivatives of the latent state"""

        if self.params['model_order'] == 1:
            dz = torch.autograd.grad(self.encode(x), x, grad_outputs=torch.ones_like(self.encode(x)))[0]

# Loss
def custom_loss(x, x_hat, z, z_hat, params):
    """Custom loss function for the autoencoder
    Args:
        x (torch.Tensor): Input data
        x_hat (torch.Tensor): Reconstructed input data
        z (torch.Tensor): Latent data
        z_hat (torch.Tensor): Reconstructed latent data
        params (dict): Parameters for the model
    Returns:
        torch.Tensor: Loss value
    """


    # Total loss
    loss = loss_weight_decoder * loss_decoder \
           + loss_weight_sindy_z * losses_sindy_z \
           + loss_weight_sindy_x * losses_sindy_x \
           + loss_weight_sindy_regularization * losses_sindy_regularization

    return loss