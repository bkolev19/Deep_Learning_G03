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
        
        self.coeff_init = params['coefficient_initialization']
        self.library_dim = library_size(self.latent_dim, self.poly_order, self.include_sine)
        
        # Initialize coefficients
        if self.coeff_init == "xavier":
            self.sindy_coefficients = torch.nn.init.xavier_uniform( torch.empty(self.library_dim, self.latent_dim))
        elif self.coeff_init == "normal":
            self.sindy_coefficients = torch.nn.init.normal_( torch.empty(self.library_dim, self.latent_dim))
        elif self.coeff_init == "constant":
            self.sindy_coefficients = torch.nn.init.constant_( torch.empty(self.library_dim, self.latent_dim), 1.0)
        elif self.coeff_init == "specified":
            self.sindy_coefficients = torch.tensor(params['init_coefficients'])

        # Sequential thresholding
        self.seq_thresholding = params['sequential_thresholding']
        if self.seq_thresholding:
            self.mask = params['coefficient_mask']
            self.sindy_w_mask = (self.sindy_coefficients * self.mask).float()
            self.loss_func = self.custom_loss_refined
        else:
            self.mask = None
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
        #first_grads = torch.autograd.grad(self.encoder(x), x, grad_outputs=torch.ones(self.encoder(x).shape), create_graph=True)[0]

        if x.shape[0] == 1:
            grads_list = []
            x_temp = x
            x_temp.requires_grad = True
            z_temp = self.encoder(x_temp)
            # Compute gradient for each output element
            for i in range(z_temp.size(1)):
                grad_outputs = torch.zeros_like(z_temp)
                grad_outputs[:,i] = 1
                grads = torch.autograd.grad(outputs=z_temp, inputs=x_temp,
                                            grad_outputs=grad_outputs,
                                            create_graph=True, retain_graph=True)[0]
                grads_list.append(grads)
            first_grads = torch.cat(grads_list, dim=0)
            dz = torch.matmul(first_grads, dx.T).squeeze().unsqueeze(0)

            if ddx is not None:
                x_temp = x
                x_temp.requires_grad = True
                grads2_list = []
                for i in range(dz.size(1)):
                    grad_outputs = torch.zeros_like(dz)
                    grad_outputs[:,i] = 1
                    grads = torch.autograd.grad(outputs=dz, inputs=x_temp,
                                                grad_outputs=grad_outputs,
                                                create_graph=True, retain_graph=True)[0]
                    grads2_list.append(grads)
                second_grads = torch.cat(grads2_list, dim=0)
                ddz = torch.matmul(second_grads, ddx.T).squeeze().unsqueeze(0)
            else:
                ddz = None

        else:
            dz = torch.zeros_like(self.z)
            for i in range(x.shape[0]):
                x_temp = x[i].unsqueeze(0)
                x_temp.requires_grad = True
                z_temp = self.encoder(x_temp)
                # Compute gradient for each output element
                grads_list = []
                for j in range(z_temp.size(1)):
                    grad_outputs = torch.zeros_like(z_temp)
                    grad_outputs[:,j] = 1
                    grads = torch.autograd.grad(outputs=z_temp, inputs=x_temp,
                                                grad_outputs=grad_outputs,
                                                create_graph=True, retain_graph=True)[0]
                    grads_list.append(grads)
                first_grads = torch.cat(grads_list, dim=0)
                dz[i] = torch.matmul(first_grads, dx[i].unsqueeze(0).T).squeeze()
            if ddx is not None:
                for i in range(x.shape[0]):
                    x_temp = x
                    x_temp.requires_grad = True
                    dz_temp = dz[i].unsqueeze(0)
                    z_temp = self.encoder(x_temp)
                    grads2_list = []
                    for i in range(dz_temp.size(1)):
                        grad_outputs = torch.zeros_like(dz_temp)
                        grad_outputs[:,i] = 1
                        grads = torch.autograd.grad(outputs=dz_temp, inputs=x_temp,
                                                    grad_outputs=grad_outputs,
                                                    create_graph=True, retain_graph=True)[0]
                        grads2_list.append(grads)
                    second_grads = torch.cat(grads2_list, dim=0)
                    ddz[i] = torch.matmul(second_grads, ddx[i].unsqueeze(0).T).squeeze()
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
        if z.shape[0] == 1:
            grads_list = []
            z_temp = z
            x_temp = self.decoder(z_temp)
            # Compute gradient for each output element
            for i in range(x_temp.size(1)):
                grad_outputs = torch.zeros_like(x_temp)
                grad_outputs[:,i] = 1
                grads = torch.autograd.grad(outputs=x_temp, inputs=z_temp,
                                            grad_outputs=grad_outputs,
                                            create_graph=True, retain_graph=True)[0]
                grads_list.append(grads)
            first_grads = torch.cat(grads_list, dim=0)
            dx = torch.matmul(first_grads, dz.T).squeeze().unsqueeze(0)

            if ddz is not None:
                z_temp = z
                z_temp.requires_grad = True
                grads2_list = []
                for i in range(dx.size(1)):
                    grad_outputs = torch.zeros_like(dx)
                    grad_outputs[:,i] = 1
                    grads = torch.autograd.grad(outputs=dx, inputs=z_temp,
                                                grad_outputs=grad_outputs,
                                                create_graph=True, retain_graph=True)[0]
                    grads2_list.append(grads)
                second_grads = torch.cat(grads2_list, dim=0)
                ddx = torch.matmul(second_grads, ddz.T).squeeze().unsqueeze(0)
            else:
                ddx = None

        else:
            dx = torch.zeros_like(self.x)
            for i in range(z.shape[0]):
                z_temp = z[i].unsqueeze(0)
                x_temp = self.decoder(z_temp)
                # Compute gradient for each output element
                grads_list = []
                for j in range(x_temp.size(1)):
                    grad_outputs = torch.zeros_like(x_temp)
                    grad_outputs[:,j] = 1
                    grads = torch.autograd.grad(outputs=x_temp, inputs=z_temp,
                                                grad_outputs=grad_outputs,
                                                create_graph=True, retain_graph=True)[0]
                    grads_list.append(grads)
                first_grads = torch.cat(grads_list, dim=0)
                dx[i] = torch.matmul(first_grads, dz[i].unsqueeze(0).T).squeeze()
            if ddz is not None:
                for i in range(z.shape[0]):
                    z_temp = z
                    z_temp.requires_grad = True
                    dx_temp = dx[i].unsqueeze(0)
                    x_temp = self.decoder(z_temp)
                    grads2_list = []
                    for i in range(dx_temp.size(1)):
                        grad_outputs = torch.zeros_like(dx_temp)
                        grad_outputs[:,i] = 1
                        grads = torch.autograd.grad(outputs=dx_temp, inputs=z_temp,
                                                    grad_outputs=grad_outputs,
                                                    create_graph=True, retain_graph=True)[0]
                        grads2_list.append(grads)
                    second_grads = torch.cat(grads2_list, dim=0)
                    ddx[i] = torch.matmul(second_grads, ddz[i].unsqueeze(0).T).squeeze()
            else:
                ddx = None
        
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
                self.mask = mask
                self.sindy_w_mask = (self.sindy_coefficients * self.mask).float()
            return torch.matmul(Theta, self.sindy_w_mask)
        else:
            return torch.matmul(Theta, self.sindy_coefficients)

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
