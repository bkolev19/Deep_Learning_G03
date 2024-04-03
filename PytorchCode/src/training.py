# import sys
# sys.path.append('..')
# import os
# import torch
# import torch.nn as nn 
# import pickle
# import numpy as np
# from tqdm import tqdm
# from autoencoder import full_network, define_loss

# # USING GPU
# # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
# # os.environ["CUDA_VISIBLE_DEVICES"]="0" # GPU index

# # print("Available GPUs:", torch.cuda.device_count())
# # for i in range(torch.cuda.device_count()):
# #     print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# # torch.cuda.set_device(device)
# # print(device)




# def train_network(training_data, val_data, params, loss_fn):
#     # TODO: functions bellow from autoencoder.py are not implemented yet.
#     autoencoder_network = full_network(params)
#     loss, losses, loss_refinement = define_loss(autoencoder_network, params)  
#     # TODO: makie sure the pars to be passed to the function are defined by Niko & Ana
#     train_op = torch.optim.Adam(autoencoder_network.parameters(), lr=params['learning_rate'])
#     train_op_refinement = torch.optim.Adam(autoencoder_network.parameters(), lr=params['learning_rate']).minimize(loss_refinement)

#     print('TRAINING')
#     for i in tqdm(range(params['max_epochs']), desc='Training'):
#         for batch in range(params['epoch_size']//params['batch_size']):
#             batch_idxs = torch.arange(batch*params['batch_size'], (batch+1)*params['batch_size'])
#             train_dict = create_feed_dictionary(training_data, params, idxs=batch_idxs)
#             train_op.zero_grad()
#             loss = loss_fn(autoencoder_network, train_dict)
#             loss.backward()
#             train_op.step()

#     for _ in tqdm(range(params['refinement_epochs']), desc='Refinement'):
#         for batch in range(params['epoch_size']//params['batch_size']):
#             batch_idxs = torch.arange(batch*params['batch_size'], (batch+1)*params['batch_size'])
#             train_dict = create_feed_dictionary(training_data, params, idxs=batch_idxs)
#             train_op_refinement.zero_grad()
#             loss = loss_fn(autoencoder_network, train_dict)
#             loss.backward()
#             train_op_refinement.step()
    
#     torch.save(autoencoder_network.state_dict(), params['data_path'] + params['save_name'])
#     pickle.dump(params, open(params['data_path'] + params['save_name'] + '_params.pkl', 'wb'))
#     with torch.no_grad():
#         final_losses = (losses['decoder'](autoencoder_network, validation_dict),
#                         losses['sindy_x'](autoencoder_network, validation_dict),
#                         losses['sindy_z'](autoencoder_network, validation_dict),
#                         losses['sindy_regularization'](autoencoder_network, validation_dict))
#         if params['model_order'] == 1:
#             sindy_predict_norm_z = torch.mean(autoencoder_network['dz'](validation_dict['x'])**2)
#         else:
#             sindy_predict_norm_z = torch.mean(autoencoder_network['ddz'](validation_dict['x'])**2)
#         sindy_coefficients = autoencoder_network['sindy_coefficients'](torch.empty(0))



#     results = {}
#     results['num_epochs'] = i
#     results['x_norm'] = x_norm
#     results['sindy_predict_norm_x'] = sindy_predict_norm_x
#     results['sindy_predict_norm_z'] = sindy_predict_norm_z
#     results['sindy_coefficients'] = sindy_coefficients
#     results['loss_decoder'] = final_losses[0]
#     results['loss_decoder_sindy'] = final_losses[1]
#     results['loss_sindy'] = final_losses[2]
#     results['loss_sindy_regularization'] = final_losses[3]
#     results['validation_losses'] = np.array(validation_losses)
#     results['sindy_model_terms'] = np.array(sindy_model_terms)


# def create_feed_dictionary(data, params, idxs=None):
#     """
#     Create the feed dictionary for passing into tensorflow.

#     Arguments:
#         data - Dictionary object containing the data to be passed in. Must contain input data x,
#         along the first (and possibly second) order time derivatives dx (ddx).
#         params - Dictionary object containing model and training parameters. The relevant
#         parameters are model_order (which determines whether the SINDy model predicts first or
#         second order time derivatives), sequential_thresholding (which indicates whether or not
#         coefficient thresholding is performed), coefficient_mask (optional if sequential
#         thresholding is performed; 0/1 mask that selects the relevant coefficients in the SINDy
#         model), and learning rate (float that determines the learning rate).
#         idxs - Optional array of indices that selects which examples from the dataset are passed
#         in to tensorflow. If None, all examples are used.

#     Returns:
#         feed_dict - Dictionary object containing the relevant data to pass to tensorflow.
#     """
#     if idxs is None:
#         idxs = np.arange(data['x'].shape[0])
#     feed_dict = {}
#     feed_dict['x:0'] = data['x'][idxs]
#     feed_dict['dx:0'] = data['dx'][idxs]
#     if params['model_order'] == 2:
#         feed_dict['ddx:0'] = data['ddx'][idxs]
#     if params['sequential_thresholding']:
#         feed_dict['coefficient_mask:0'] = params['coefficient_mask']
#     feed_dict['learning_rate:0'] = params['learning_rate']
#     return feed_dict

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pickle
import numpy as np
from autoencoder import full_network, define_loss

def train_batch()







def train_network(training_data, val_data, params):
    # SET UP NETWORK
    autoencoder_network = full_network(params)
    loss, losses, loss_refinement = define_loss(autoencoder_network, params)
    # learning_rate = torch.tensor(0.001, dtype=torch.float32, requires_grad=False) # Placeholder for learning rate
    train_op = optim.Adam(loss, 0.001)
    train_op_ref = optim.Adam(loss_refinement,0.001)
    
    validation_dict = create_feed_dictionary(val_data, params, idxs=None)

    x_norm = np.mean(val_data['x']**2)
    if params['model_order'] == 1:
        sindy_predict_norm_x = np.mean(val_data['dx']**2)
    else:
        sindy_predict_norm_x = np.mean(val_data['ddx']**2)

    validation_losses = []
    sindy_model_terms = [np.sum(params['coefficient_mask'])]

    print('TRAINING_PART2')
    for i in tqdm(range(params['max_epochs']), desc='Training'):
        for j in range(params['epoch_size']//params['batch_size']):
            batch_idxs = np.arange(j*params['batch_size'], (j+1)*params['batch_size'])
            train_dict = create_feed_dictionary(training_data, params, idxs=batch_idxs)
            train_op.zero_grad()
            outputs = autoencoder_network(train_dict['x:0'])
            loss_value = loss(outputs, train_dict['x:0'])
            loss_value.backward()
            train_op.step()

        if params['print_progress'] and (i % params['print_frequency'] == 0):
            validation_losses.append(print_progress(autoencoder_network, i, loss, losses, train_dict, validation_dict, x_norm, sindy_predict_norm_x))

        if params['sequential_thresholding'] and (i % params['threshold_frequency'] == 0) and (i > 0):
            params['coefficient_mask'] = torch.abs(autoencoder_network['sindy_coefficients']) > params['coefficient_threshold']
            validation_dict['coefficient_mask:0'] = params['coefficient_mask']
            print('THRESHOLDING: %d active coefficients' % torch.sum(params['coefficient_mask']))
            sindy_model_terms.append(torch.sum(params['coefficient_mask']))

    print('REFINEMENT')
    for i_refinement in range(params['refinement_epochs']):
        for j in range(params['epoch_size']//params['batch_size']):
            batch_idxs = np.arange(j*params['batch_size'], (j+1)*params['batch_size'])
            train_dict = create_feed_dictionary(training_data, params, idxs=batch_idxs)
            train_op_ref.zero_grad()
            outputs = autoencoder_network(train_dict['x:0'])
            loss_value = loss_refinement(outputs, train_dict['x:0'])
            loss_value.backward()
            train_op_ref.step()

        if params['print_progress'] and (i_refinement % params['print_frequency'] == 0):
            validation_losses.append(print_progress(autoencoder_network, i_refinement, loss_refinement, losses, train_dict, validation_dict, x_norm, sindy_predict_norm_x))

    torch.save(autoencoder_network.state_dict(), params['data_path'] + params['save_name'])
    pickle.dump(params, open(params['data_path'] + params['save_name'] + '_params.pkl', 'wb'))
    final_losses = loss(autoencoder_network(validation_dict['x:0']), validation_dict['x:0'])
    if params['model_order'] == 1:
        sindy_predict_norm_z = np.mean(autoencoder_network(validation_dict['x:0'])**2)
    else:
        sindy_predict_norm_z = np.mean(autoencoder_network(validation_dict['x:0'])**2)
    sindy_coefficients = autoencoder_network['sindy_coefficients'].detach().numpy()

    results_dict = {}
    results_dict['num_epochs'] = i
    results_dict['x_norm'] = x_norm
    results_dict['sindy_predict_norm_x'] = sindy_predict_norm_x
    results_dict['sindy_predict_norm_z'] = sindy_predict_norm_z
    results_dict['sindy_coefficients'] = sindy_coefficients
    results_dict['loss_decoder'] = final_losses
    results_dict['validation_losses'] = np.array(validation_losses)
    results_dict['sindy_model_terms'] = np.array(sindy_model_terms)

    return results_dict

def print_progress(network, i, loss_fn, train_dict, validation_dict, x_norm, sindy_predict_norm):
    with torch.no_grad():
        training_loss_vals = loss_fn(network(train_dict['x:0']), train_dict['x:0'])
        validation_loss_vals = loss_fn(network(validation_dict['x:0']), validation_dict['x:0'])
        print("Epoch %d" % i)
        print("   training loss {0}, {1}".format(training_loss_vals[0], training_loss_vals[1:]))
        print("   validation loss {0}, {1}".format(validation_loss_vals[0], validation_loss_vals[1:]))
        decoder_losses = loss_fn(network(validation_dict['x:0']), validation_dict['x:0'])
        loss_ratios = (decoder_losses[0]/x_norm, decoder_losses[1]/sindy_predict_norm)
        print("decoder loss ratio: %f, decoder SINDy loss  ratio: %f" % loss_ratios)
        return validation_loss_vals

def create_feed_dictionary(data, params, idxs=None):
    if idxs is None:
        idxs = np.arange(data['x'].shape[0])
    feed_dict = {}
    feed_dict['x:0'] = torch.tensor(data['x'][idxs], dtype=torch.float32)
    feed_dict['dx:0'] = torch.tensor(data['dx'][idxs], dtype=torch.float32)
    if params['model_order'] == 2:
        feed_dict['ddx:0'] = torch.tensor(data['ddx'][idxs], dtype=torch.float32)
    if params['sequential_thresholding']:
        feed_dict['coefficient_mask:0'] = torch.tensor(params['coefficient_mask'], dtype=torch.float32)
    feed_dict['learning_rate:0'] = torch.tensor(params['learning_rate'], dtype=torch.float32)
    return feed_dict
