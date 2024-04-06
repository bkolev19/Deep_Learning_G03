"""File to create the autoencoder model in PyTorch. Based on the code for Sindy Autoencoder by Kathleen Champion."""

# Libraries
import torch
from scipy.special import binom


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
        self.model_order = params["model_order"]

        if self.activation == "sigmoid":
                self.act_func = torch.nn.Sigmoid()
        elif self.activation == "relu":
                self.act_func = torch.nn.ReLU()
        elif self.activation == "elu":
                self.act_func = torch.nn.ELU()
                
        # Loss params
        self.lambda_0 = params['loss_weight_decoder']
        self.lambda_1 = params['loss_weight_sindy_x']
        self.lambda_2 = params['loss_weight_sindy_z']
        self.lambda_3 = params['loss_weight_sindy_regularization']
        # Sindy params
        self.poly_order = params['poly_order']
        self.include_sine = params['include_sine']
        # self.sindy_coefficients = torch.nn.Parameter(torch.Tensor(self.library_dim, self.latent_dim)) # ADDED
        
        self.coeff_init = params['coefficient_initialization']
        self.library_dim = library_size(self.latent_dim * self.model_order, self.poly_order, self.include_sine)
        

        self.sindy_coefficients = torch.nn.Parameter(torch.Tensor(self.library_dim, self.latent_dim))
        # Initialize coefficients
        if self.coeff_init == "xavier":
            torch.nn.init.xavier_uniform_(self.sindy_coefficients)
        elif self.coeff_init == "normal":
            torch.nn.init.normal_(self.sindy_coefficients)
        elif self.coeff_init == "constant":
            torch.nn.init.constant_(self.sindy_coefficients, 1.0)
        elif self.coeff_init == "specified":
            torch.nn.init.constant_(self.sindy_coefficients, params['specified_coefficient'])

        # Sequential thresholding
        self.seq_thresholding = params['sequential_thresholding']
        if self.seq_thresholding:
            self.mask = torch.tensor(params['coefficient_mask'])
        else:
            self.mask = None

        # Loss 
        self.loss_func = self.custom_loss_unrefined

        # Encoder
        encoder_layers = [torch.nn.Linear(self.input_dim, self.hidden_dims[0])]
        for i in range(len(self.hidden_dims)-1):
            if self.activation != "linear":
                encoder_layers.append(self.act_func)
            encoder_layers.append(torch.nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))
        if self.activation != "linear":
                encoder_layers.append(self.act_func)
        encoder_layers.append(torch.nn.Linear(self.hidden_dims[-1], self.latent_dim))

        self.encoder = torch.nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = [torch.nn.Linear(self.latent_dim, self.hidden_dims[-1])]
        for i in range(len(self.hidden_dims)-1, 0, -1):
            if self.activation != "linear":
                decoder_layers.append(self.act_func)
            decoder_layers.append(torch.nn.Linear(self.hidden_dims[i], self.hidden_dims[i-1]))
        if self.activation != "linear":
                decoder_layers.append(self.act_func)
        decoder_layers.append(torch.nn.Linear(self.hidden_dims[0], self.input_dim))

        self.decoder = torch.nn.Sequential(*decoder_layers)

    def forward(self, state, mask=None):
        """Forward pass for the autoencoder
        Args:
            state (torch.Tensor): Input data (x, dx, ddx=None)
            mask (torch.Tensor): Coefficient mask for SINDy
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
        self.dz, self.ddz = self.compute_derivatives_x2z(self.x, self.dx, self.ddx)
        
        # Transform using sindy and decode
        if self.ddz is not None:
            self.sindy_ddz = self.sindy(mask, secondOrder=True)
            self.sindy_dz = None
            self.dx_hat, self.ddx_hat = self.compute_derivatives_z2x(self.z, self.dz, self.sindy_ddz)
            return self.x_hat, self.dx_hat, self.ddx_hat
        else:
            self.sindy_ddz = None
            self.sindy_dz = self.sindy(mask)
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
        layers = list(self.encoder.children())
        if self.params['model_order'] == 1:
            dz = derivative(x, dx, layers, self.activation, self.act_func)
            ddz = None
        else:
            dz, ddz = derivative_order2(x, dx, ddx, layers, self.activation, self.act_func)

        return dz, ddz

    def compute_derivatives_z2x(self, z, dz, ddz=None):
        """Compute the derivatives of the input data
        Args:
            z (torch.Tensor): Latent state
            dz (torch.Tensor): First derivative of the latent state
            ddz (torch.Tensor): Second derivative of the latent state
        Returns:
            torch.Tensor: Derivatives of the input data"""
        layers = list(self.decoder.children())
        if self.params['model_order'] == 1:
            dx = derivative(z, dz, layers, self.activation, self.act_func)
            ddx = None
        else:
            dx, ddx = derivative_order2(z, dz, ddz, layers, self.activation, self.act_func)

        return dx, ddx

    def sindy(self, mask, secondOrder=False):
        """Compute the SINDy model. Here what we do is have z, dz and possibly ddz. What it will do is try to find a function that takes z, (sometimes dz) to predict ddz/dz
        Returns:
            torch.Tensor: SINDy model prediction"""
        
        # Get Theta
        if secondOrder:
            Theta = sindy_library_torch_order2(self.z, self.dz, self.latent_dim, self.params['poly_order'], self.include_sine)
        else:
            Theta = sindy_library_torch(self.z, self.latent_dim, self.params['poly_order'], self.include_sine)

        # Check if we need coefficient mask
        if self.seq_thresholding:
            if mask is not None:
                self.mask = torch.tensor(mask)
            return torch.matmul(Theta, (self.sindy_coefficients * self.mask).float())
        else:
            return torch.matmul(Theta, self.sindy_coefficients.float())

    # Loss
    # Call self.loss_func() to get the loss, rather than the specific loss function
    def custom_loss_unrefined(self):
        """Custom loss function for the autoencoder
        Returns:
            torch.Tensor: Loss value
        """
        # Decoder loss
        loss_decoder = torch.nn.functional.mse_loss(self.x_hat, self.x)

        # SINDy loss x and z
        if self.ddx is not None:
            losses_sindy_x = torch.nn.functional.mse_loss(self.ddx_hat, self.ddx)
            losses_sindy_z = torch.nn.functional.mse_loss(self.ddz, self.sindy_ddz)
        else:
            losses_sindy_x = torch.nn.functional.mse_loss(self.dx_hat, self.dx)
            losses_sindy_z = torch.nn.functional.mse_loss(self.dz, self.sindy_dz)

        # Total loss
        loss = self.lambda_0 * loss_decoder \
               + self.lambda_1 * losses_sindy_x \
               + self.lambda_2 * losses_sindy_z \

        return loss
    
    def custom_loss_refined(self):
        """Custom loss function for the autoencoder
        Returns:
            torch.Tensor: Loss value
        """
        # Decoder loss
        loss_decoder = torch.nn.functional.mse_loss(self.x_hat, self.x)

        # SINDy loss x and z
        if self.ddx is not None:
            losses_sindy_x = torch.nn.functional.mse_loss(self.ddx_hat, self.ddx)
            losses_sindy_z = torch.nn.functional.mse_loss(self.ddz, self.sindy_ddz)
        else:
            losses_sindy_x = torch.nn.functional.mse_loss(self.dx_hat, self.dx)
            losses_sindy_z = torch.nn.functional.mse_loss(self.dz, self.sindy_dz)
        losses_sindy_regularization = torch.norm(torch.abs(self.sindy_coefficients))

        # Total loss
        loss = self.lambda_0 * loss_decoder \
               + self.lambda_1 * losses_sindy_x \
               + self.lambda_2 * losses_sindy_z \
               + self.lambda_3 * losses_sindy_regularization

        return loss
                    
# Extra derivative functins
def derivative_order2(input, dx, ddx, layers, activation, act_func):
    dz = dx
    ddz = ddx
    for i, layer in enumerate(layers):
        if i < len(layers) - 1:
            input = layer(input)

            if activation == 'elu':
                # ELU derivative: exp(input) if input < 0 else 1
                elu_derivative = torch.where(input < 0, torch.exp(input), torch.tensor(1.0))
                elu_double_derivative = torch.where(input < 0, torch.exp(input), torch.tensor(0.0))
                
                dz_prev = layer(dz)
                ddz_prev = layer(ddz)

                # Apply the chain rule for the first and second derivative
                dz = elu_derivative * dz_prev
                ddz = elu_double_derivative * dz_prev * dz_prev + elu_derivative * ddz_prev

                input = act_func(input)

            elif activation == 'relu':
                # ReLU derivative: 1 if input > 0 else 0
                relu_derivative = (input > 0).float()
                relu_double_derivative = torch.zeros_like(input) # second derivative of ReLU is 0

                dz_prev = layer(dz)
                ddz_prev = layer(ddz)

                # Apply the chain rule for the first and second derivative
                dz = relu_derivative * dz_prev
                ddz = relu_double_derivative * dz_prev * dz_prev + relu_derivative * ddz_prev

                input = act_func(input)

            elif activation == 'sigmoid':
                # Sigmoid derivative: sigmoid(input) * (1 - sigmoid(input))
                sigmoid_derivative = act_func(input) * (1 - act_func(input))
                sigmoid_double_derivative = act_func(input) * (1 - act_func(input)) * (1 - 2 * act_func(input))

                dz_prev = layer(dz)
                ddz_prev = layer(ddz)

                # Apply the chain rule for the first and second derivative
                dz = sigmoid_derivative * dz_prev
                ddz = sigmoid_double_derivative * dz_prev * dz_prev + sigmoid_derivative * ddz_prev

                input = act_func(input)

            # Add other activation conditions here
        else:
            dz = layer(dz)
            ddz = layer(ddz)

    return dz, ddz

def derivative(input, dx, layers, activation, act_func):
    dz = dx
    for i, layer in enumerate(layers):
        if i < len(layers) - 1:
            input = layer(input)
            if activation == 'elu':
                dz = torch.where(input < 0, torch.exp(input), torch.ones_like(input)) * layer(dz)
            elif activation == 'relu':
                dz = (input > 0).float() * layer(dz)
            elif activation == 'sigmoid':
                dz = act_func(input) * (1 - torch.sigmoid(input)) * layer(dz)
            input = act_func(input)
        else:
            dz = layer(dz)
    return dz


# Extra functions (SINDY)
def library_size(latent_dim, poly_order, use_sine=False, include_constant=True):
    l = 0
    for k in range(poly_order+1):
        l += int(binom(latent_dim+k-1,k))
    if use_sine:
        l += latent_dim
    if not include_constant:
        l -= 1
    return l

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
    library = [torch.ones(z.size(0))]

    for i in range(latent_dim):
        library.append(z[:,i])

    if poly_order > 1:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                library.append(z[:,i] * z[:,j])

    if poly_order > 2:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    library.append(z[:,i] * z[:,j] * z[:,k])

    if poly_order > 3:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        library.append(z[:,i] * z[:,j] * z[:,k] * z[:,p])

    if poly_order > 4:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        for q in range(p,latent_dim):
                            library.append(z[:,i] * z[:,j] * z[:,k] * z[:,p] * z[:,q])

    if include_sine:
        for i in range(latent_dim):
            library.append(torch.sin(z[:,i]))

    return torch.stack(library, dim=1)


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

    return torch.stack(library, dim=1)

