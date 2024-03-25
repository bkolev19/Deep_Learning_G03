import torch

def full_network(params):
    """
    Define the full network architecture.

    Arguments:
        params - Dictionary object containing the parameters that specify the training.
        See README file for a description of the parameters.

    Returns:
        network - Dictionary containing the tensorflow objects that make up the network.
    """
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

    network = {}

    # x = torch.tensor(torch.float32, shape=[None, input_dim], name='x')
    # dx = torch.tensor(torch.float32, shape=[None, input_dim], name='dx')
    x = torch.zeros((params['batch_size'], input_dim), dtype=torch.float32, requires_grad=False)
    dx = torch.zeros((params['batch_size'], input_dim), dtype=torch.float32, requires_grad=False)
    if model_order == 2:
        # ddx = torch.tensor(torch.float32, shape=[None, input_dim], name='ddx')
        ddx = torch.zeros((params['batch_size'], input_dim), dtype=torch.float32, requires_grad=False)

    if activation == 'linear':
        z, x_decode, encoder_weights, encoder_biases, decoder_weights, decoder_biases = linear_autoencoder(x, input_dim, latent_dim)
    else:
        z, x_decode, encoder_weights, encoder_biases, decoder_weights, decoder_biases = nonlinear_autoencoder(x, input_dim, latent_dim, params['widths'], activation=activation)
    
    if model_order == 1:
        dz = z_derivative(x, dx, encoder_weights, encoder_biases, activation=activation)
        Theta = sindy_library_torch(z, latent_dim, poly_order, include_sine)
    else:
        dz,ddz = z_derivative_order2(x, dx, ddx, encoder_weights, encoder_biases, activation=activation)
        Theta = sindy_library_torch_order2(z, dz, latent_dim, poly_order, include_sine)

    if params['coefficient_initialization'] == 'xavier':
        sindy_coefficients = torch.nn.Parameter(torch.empty(library_dim, latent_dim))
        torch.nn.init.xavier_uniform_(sindy_coefficients)
    elif params['coefficient_initialization'] == 'specified':
        sindy_coefficients = torch.nn.Parameter(params['init_coefficients'])
    elif params['coefficient_initialization'] == 'constant':
        sindy_coefficients = torch.nn.Parameter(torch.ones(library_dim, latent_dim))
    elif params['coefficient_initialization'] == 'normal':
        sindy_coefficients = torch.nn.Parameter(torch.randn(library_dim, latent_dim))
    
    if params['sequential_thresholding']:
        coefficient_mask = torch.zeros((library_dim, latent_dim), dtype=torch.float32, requires_grad=False)
        sindy_predict = torch.matmul(Theta, coefficient_mask*sindy_coefficients)
        network['coefficient_mask'] = coefficient_mask
    else:
        sindy_predict = torch.matmul(Theta, sindy_coefficients)

    if model_order == 1:
        dx_decode = z_derivative(z, sindy_predict, decoder_weights, decoder_biases, activation=activation)
    else:
        dx_decode,ddx_decode = z_derivative_order2(z, dz, sindy_predict, decoder_weights, decoder_biases,
                                             activation=activation)

    network['x'] = x
    network['dx'] = dx
    network['z'] = z
    network['dz'] = dz
    network['x_decode'] = x_decode
    network['dx_decode'] = dx_decode
    network['encoder_weights'] = encoder_weights
    network['encoder_biases'] = encoder_biases
    network['decoder_weights'] = decoder_weights
    network['decoder_biases'] = decoder_biases
    network['Theta'] = Theta
    network['sindy_coefficients'] = sindy_coefficients

    if model_order == 1:
        network['dz_predict'] = sindy_predict
    else:
        network['ddz'] = ddz
        network['ddz_predict'] = sindy_predict
        network['ddx'] = ddx
        network['ddx_decode'] = ddx_decode

    return network


def define_loss(network, params):
    """
    Create the loss functions.

    Arguments:
        network - Dictionary object containing the elements of the network architecture.
        This will be the output of the full_network() function.
    """
    x = network['x']
    x_decode = network['x_decode']
    if params['model_order'] == 1:
        dz = network['dz']
        dz_predict = network['dz_predict']
        dx = network['dx']
        dx_decode = network['dx_decode']
    else:
        ddz = network['ddz']
        ddz_predict = network['ddz_predict']
        ddx = network['ddx']
        ddx_decode = network['ddx_decode']
        
    # Convert params['coefficient_mask'] to a PyTorch tensor
    coefficient_mask = torch.tensor(params['coefficient_mask'], dtype=torch.float32)

# Extract the value of network['sindy_coefficients'] (a Parameter) as a tensor
    sindy_coefficients = network['sindy_coefficients'].data

# Perform element-wise multiplication
    sindy_coefficients = coefficient_mask * sindy_coefficients

    losses = {}
    losses['decoder'] = torch.mean((x - x_decode)**2)
    if params['model_order'] == 1:
        losses['sindy_z'] = torch.mean((dz - dz_predict)**2)
        losses['sindy_x'] = torch.mean((dx - dx_decode)**2)
    else:
        losses['sindy_z'] = torch.mean((ddz - ddz_predict)**2)
        losses['sindy_x'] = torch.mean((ddx - ddx_decode)**2)
    losses['sindy_regularization'] = torch.mean(torch.abs(sindy_coefficients))
    loss = params['loss_weight_decoder'] * losses['decoder'] \
           + params['loss_weight_sindy_z'] * losses['sindy_z'] \
           + params['loss_weight_sindy_x'] * losses['sindy_x'] \
           + params['loss_weight_sindy_regularization'] * losses['sindy_regularization']
    print(f'{type(loss)}')

    loss_refinement = params['loss_weight_decoder'] * losses['decoder'] \
                      + params['loss_weight_sindy_z'] * losses['sindy_z'] \
                      + params['loss_weight_sindy_x'] * losses['sindy_x']
    print(f'{type(loss_refinement)}')
    return loss, losses, loss_refinement


def linear_autoencoder(x, input_dim, d):
    # z,encoder_weights,encoder_biases = encoder(x, input_dim, latent_dim, [], None, 'encoder')
    # x_decode,decoder_weights,decoder_biases = decoder(z, input_dim, latent_dim, [], None, 'decoder')
    z,encoder_weights,encoder_biases = build_network_layers(x, input_dim, latent_dim, [], None, 'encoder')
    x_decode,decoder_weights,decoder_biases = build_network_layers(z, latent_dim, input_dim, [], None, 'decoder')

    return z, x_decode, encoder_weights, encoder_biases,decoder_weights,decoder_biases


def nonlinear_autoencoder(x, input_dim, latent_dim, widths, activation='elu'):
    """
    Construct a nonlinear autoencoder.

    Arguments:

    Returns:
        z -
        x_decode -
        encoder_weights - List of tensorflow arrays containing the encoder weights
        encoder_biases - List of tensorflow arrays containing the encoder biases
        decoder_weights - List of tensorflow arrays containing the decoder weights
        decoder_biases - List of tensorflow arrays containing the decoder biases
    """
    if activation == 'relu':
        activation_function = torch.relu
    elif activation == 'elu':
        activation_function = torch.nn.functional.elu
    elif activation == 'sigmoid':
        activation_function = torch.sigmoid
    else:
        raise ValueError('invalid activation function')
    # z,encoder_weights,encoder_biases = encoder(x, input_dim, latent_dim, widths, activation_function, 'encoder')
    # x_decode,decoder_weights,decoder_biases = decoder(z, input_dim, latent_dim, widths[::-1], activation_function, 'decoder')
    z,encoder_weights,encoder_biases = build_network_layers(x, input_dim, latent_dim, widths, activation_function, 'encoder')
    x_decode,decoder_weights,decoder_biases = build_network_layers(z, latent_dim, input_dim, widths[::-1], activation_function, 'decoder')

    return z, x_decode, encoder_weights, encoder_biases, decoder_weights, decoder_biases


def build_network_layers(input, input_dim, output_dim, widths, activation, name):
    """
    Construct one portion of the network (either encoder or decoder).

    Arguments:
        input - 2D tensorflow array, input to the network (shape is [?,input_dim])
        input_dim - Integer, number of state variables in the input to the first layer
        output_dim - Integer, number of state variables to output from the final layer
        widths - List of integers representing how many units are in each network layer
        activation - Tensorflow function to be used as the activation function at each layer
        name - String, prefix to be used in naming the tensorflow variables

    Returns:
        input - Tensorflow array, output of the network layers (shape is [?,output_dim])
        weights - List of tensorflow arrays containing the network weights
        biases - List of tensorflow arrays containing the network biases
    """
    weights = []
    biases = []
    last_width = input_dim
    for i, n_units in enumerate(widths):
        W = torch.nn.Parameter(torch.empty(last_width, n_units))
        torch.nn.init.xavier_uniform_(W)
        b = torch.nn.Parameter(torch.zeros(n_units))
        input = torch.matmul(input, W) + b
        if activation is not None:
            input = activation(input)
        last_width = n_units
        weights.append(W)
        biases.append(b)
    W = torch.nn.Parameter(torch.empty(last_width, output_dim))
    torch.nn.init.xavier_uniform_(W)
    b = torch.nn.Parameter(torch.zeros(output_dim))
    input = torch.matmul(input, W) + b
    weights.append(W)
    biases.append(b)
    return input, weights, biases




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


def z_derivative(input, dx, weights, biases, activation='elu'):
    """
    Compute the first order time derivatives by propagating through the network.

    Arguments:
        input - 2D torch tensor, input to the network. Dimensions are number of time points
        by number of state variables.
        dx - First order time derivatives of the input to the network.
        weights - List of torch tensors containing the network weights
        biases - List of torch tensors containing the network biases
        activation - String specifying which activation function to use. Options are
        'elu' (exponential linear unit), 'relu' (rectified linear unit), 'sigmoid',
        or linear.

    Returns:
        dz - Torch tensor, first order time derivatives of the network output.
    """
    dz = dx
    if activation == 'elu':
        for i in range(len(weights)-1):
            input = torch.matmul(input, weights[i]) + biases[i]
            dz = torch.mul(torch.minimum(torch.exp(input),1.0),
                                  torch.matmul(dz, weights[i]))
            input = torch.nn.functional.elu(input)
        dz = torch.matmul(dz, weights[-1])
    elif activation == 'relu':
        for i in range(len(weights)-1):
            input = torch.matmul(input, weights[i]) + biases[i]
            dz = torch.mul(torch.tensor(input>0, dtype=torch.float32), torch.matmul(dz, weights[i]))
            input = torch.nn.functional.relu(input)
        dz = torch.matmul(dz, weights[-1])
    elif activation == 'sigmoid':
        for i in range(len(weights)-1):
            input = torch.matmul(input, weights[i]) + biases[i]
            input = torch.sigmoid(input)
            dz = torch.mul(torch.mul(input, 1-input), torch.matmul(dz, weights[i]))
        dz = torch.matmul(dz, weights[-1])
    else:
        for i in range(len(weights)-1):
            dz = torch.matmul(dz, weights[i])
        dz = torch.matmul(dz, weights[-1])
    return dz


def z_derivative_order2(input, dx, ddx, weights, biases, activation='elu'):
    """
    Compute the first and second order time derivatives by propagating through the network.

    Arguments:
        input - 2D torch tensor, input to the network. Dimensions are number of time points
        by number of state variables.
        dx - First order time derivatives of the input to the network.
        ddx - Second order time derivatives of the input to the network.
        weights - List of torch tensors containing the network weights
        biases - List of torch tensors containing the network biases
        activation - String specifying which activation function to use. Options are
        'elu' (exponential linear unit), 'relu' (rectified linear unit), 'sigmoid',
        or linear.

    Returns:
        dz - Torch tensor, first order time derivatives of the network output.
        ddz - Torch tensor, second order time derivatives of the network output.
    """
    dz = dx
    ddz = ddx
    if activation == 'elu':
        for i in range(len(weights)-1):
            input = torch.matmul(input, weights[i]) + biases[i]
            dz_prev = torch.matmul(dz, weights[i])
            elu_derivative = torch.minimum(torch.exp(input),1.0)
            elu_derivative2 = torch.mul(torch.exp(input), torch.tensor(input<0, dtype=torch.float32))
            dz = torch.mul(elu_derivative, dz_prev)
            ddz = torch.mul(elu_derivative2, torch.square(dz_prev)) \
                  + torch.mul(elu_derivative, torch.matmul(ddz, weights[i]))
            input = torch.nn.functional.elu(input)
        dz = torch.matmul(dz, weights[-1])
        ddz = torch.matmul(ddz, weights[-1])
    elif activation == 'relu':
        # NOTE: currently having trouble assessing accuracy of 2nd derivative due to discontinuity
        for i in range(len(weights)-1):
            input = torch.matmul(input, weights[i]) + biases[i]
            relu_derivative = torch.tensor(input>0, dtype=torch.float32)
            dz = torch.mul(relu_derivative, torch.matmul(dz, weights[i]))
            ddz = torch.mul(relu_derivative, torch.matmul(ddz, weights[i]))
            input = torch.nn.functional.relu(input)
        dz = torch.matmul(dz, weights[-1])
        ddz = torch.matmul(ddz, weights[-1])
    elif activation == 'sigmoid':
        for i in range(len(weights)-1):
            input = torch.matmul(input, weights[i]) + biases[i]
            input = torch.sigmoid(input)
            dz_prev = torch.matmul(dz, weights[i])
            sigmoid_derivative = torch.mul(input, 1-input)
            sigmoid_derivative2 = torch.mul(sigmoid_derivative, 1 - 2*input)
            dz = torch.mul(sigmoid_derivative, dz_prev)
            ddz = torch.mul(sigmoid_derivative2, torch.square(dz_prev)) \
                  + torch.mul(sigmoid_derivative, torch.matmul(ddz, weights[i]))
        dz = torch.matmul(dz, weights[-1])
        ddz = torch.matmul(ddz, weights[-1])
    else:
        for i in range(len(weights)-1):
            dz = torch.matmul(dz, weights[i])
            ddz = torch.matmul(ddz, weights[i])
        dz = torch.matmul(dz, weights[-1])
        ddz = torch.matmul(ddz, weights[-1])
    return dz,ddz
