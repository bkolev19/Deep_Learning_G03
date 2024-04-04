import numpy as np
import torch

from autoencoder import Autoencoder

def test_order2():
    # Params
    params = {'activation': 'sigmoid',
            'batch_size': 1,
            'coefficient_initialization': 'constant',
            'coefficient_mask': np.array([[False,  True, False],
                [False, False, False],
                [False,  True, False],
                [ True, False, False],
                [False,  True, False],
                [ True, False, False],
                [ True, False, False],
                [ True, False, False],
                [ True, False, False],
                [ True, False, False]]),
            'coefficient_threshold': 0.1,
            'epoch_size': 512000,
            'include_sine': False,
            'input_dim': 8,
            'latent_dim': 3,
            'learning_rate': 0.001,
            'library_dim': 20,
            'loss_weight_decoder': 1.0,
            'loss_weight_sindy_regularization': 1e-05,
            'loss_weight_sindy_x': 0.0001,
            'loss_weight_sindy_z': 0.0,
            'max_epochs': 10001,
            'model_order': 2,
            'poly_order': 2,
            'print_frequency': 100,
            'print_progress': True,
            'refinement_epochs': 1001,
            'sequential_thresholding': True,
            'threshold_frequency': 500,
            'widths': [6, 4]}

    model = Autoencoder(params)

    # DATA
    x = torch.rand(1, 8)
    dx = torch.rand(1, 8)
    ddx = torch.rand(1, 8)


    epochs = 10
    # optim = torch.optim.Adam(loss, 0.001)
    optim = torch.optim.Adam(model.parameters(), 0.001)

    for epoch in range(epochs):
        # Forward pass
        outputs = model.forward([x, dx, ddx])
        loss = model.loss_func()
        print(loss)
        # qBackward pass
        optim.zero_grad()
        loss.backward()
        # Update Weights
        optim.step()

def test_order1():
    # Params
    params = {'activation': 'sigmoid',
            'batch_size': 1,
            'coefficient_initialization': 'constant',
            'coefficient_mask': np.array([[False,  True, False],
                [False, False, False],
                [False,  True, False],
                [ True, False, False],
                [False,  True, False],
                [ True, False, False],
                [ True, False, False],
                [ True, False, False],
                [ True, False, False],
                [ True, False, False]]),
            'coefficient_threshold': 0.1,
            'epoch_size': 512000,
            'include_sine': False,
            'input_dim': 8,
            'latent_dim': 3,
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
            'widths': [6, 4]}

    model = Autoencoder(params)

    # DATA
    x = torch.rand(1, 8)
    dx = torch.rand(1, 8)


    epochs = 10
    # optim = torch.optim.Adam(loss, 0.001)
    optim = torch.optim.Adam(model.parameters(), 0.001)

    for epoch in range(epochs):
        # Forward pass
        outputs = model.forward([x, dx])
        loss = model.loss_func()
        print(loss)
        # qBackward pass
        optim.zero_grad()
        loss.backward()
        # Update Weights
        optim.step()

if __name__ == '__main__':
    test_order1()