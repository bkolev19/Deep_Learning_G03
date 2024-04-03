import numpy as np
import torch

from autoencoder import Autoencoder


# Params
params = {'activation': 'sigmoid',
        'batch_size': 1,
        'coefficient_initialization': 'constant',
        'coefficient_mask': np.array([[False,  True],
            [False, False],
            [False,  True],
            [ True, False],
            [False,  True],
            [ True, False]]),
        'coefficient_threshold': 0.1,
        'epoch_size': 512000,
        'include_sine': False,
        'input_dim': 8,
        'latent_dim': 2,
        'learning_rate': 0.001,
        'library_dim': 20,
        'loss_weight_decoder': 1.0,
        'loss_weight_sindy_regularization': 1e-05,
        'loss_weight_sindy_x': 0.0001,
        'loss_weight_sindy_z': 0.0,
        'max_epochs': 10001,
        'model_order': 1,
        'poly_order': 2,
        'print_frequency': 100,
        'print_progress': True,
        'refinement_epochs': 1001,
        'sequential_thresholding': True,
        'threshold_frequency': 500,
        'widths': [4, 3]}

model = Autoencoder(params)

# DATA
x = torch.rand(3, 8)
dx = torch.rand(3, 8)


epochs = 10
# optim = torch.optim.Adam(loss, 0.001)
optim = torch.optim.Adam(model.parameters(), 0.001)
for epoch in range(
    # Forward pass
    outputs = model.forward([x, dx])
    loss = model.loss_func()
    print(loss)
    # Backward pass
    optim.zero_grad()
    loss.backward()
    # Update Weights
    optim.step()