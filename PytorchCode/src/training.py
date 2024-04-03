import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pickle
import numpy as np
from autoencoder import Autoencoder
    
batch_size = 32

def train_network(training_data, val_data, params):
    model       = Autoencoder(params)
    optimizer   = torch.optim.Adam(model.parameters(), 0.001)
    batch_iter  = params['epoch_size']//params['batch_size']

    validation_dict   = create_feed_dictionary(val_data, params, idxs=None)
    x_norm            = np.mean(val_data['x']**2)
    sindy_model_terms = [np.sum(params['coefficient_mask'])]
    if params['model_order'] == 1:
        sindy_predict_norm_x = np.mean(val_data['dx']**2)
    else:
        sindy_predict_norm_x = np.mean(val_data['ddx']**2)

    
    for i in tqdm(params['max_epochs'], desc='Epochs_Loop'):
        for j in range(batch_iter):
            batch_idxs = np.arange(j*params['batch_size'], (j+1)*params['batch_size'])
            train_dict = create_feed_dictionary(training_data, params, idxs=batch_idxs)
            x          = train_dict['x:0']
            dx         = train_dict['dx:0']

            model.forward([x, dx])
            loss = model.loss_func()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if params['sequential_thresholding'] and (i % params['threshold_frequency'] == 0) and (i > 0):
            params['coefficient_mask'] = np.abs(model.sindy_coefficients) > params['coefficient_threshold']
            validation_dict['coefficient_mask:0'] = params['coefficient_mask']
            print('THRESHOLDING: %d active coefficients' % np.sum(params['coefficient_mask']))
            sindy_model_terms.append(np.sum(params['coefficient_mask']))
    
    # Save the model
    MODEL_PATH = params['data_path'] + params['save_name']
    torch.save(model.state_dict(), MODEL_PATH)

    if params['model_order'] == 1:
        sindy_predict_norm_z = np.mean(model.dz, feed_dict=validation_dict)**2)
    else:
        sindy_predict_norm_z = np.mean(model.ddz, feed_dict=validation_dict)**2)
    sindy_coefficients = model.sindy_coefficients, feed_dict={})
    

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

# def train_network(training_data, val_data, params):
#     # SET UP NETWORK
#     autoencoder_network = full_network(params)
#     loss, losses, loss_refinement = define_loss(autoencoder_network, params)
#     # learning_rate = torch.tensor(0.001, dtype=torch.float32, requires_grad=False) # Placeholder for learning rate
#     train_op = optim.Adam(loss, 0.001)
#     train_op_ref = optim.Adam(loss_refinement,0.001)
    
#     validation_dict = create_feed_dictionary(val_data, params, idxs=None)

#     x_norm = np.mean(val_data['x']**2)
#     if params['model_order'] == 1:
#         sindy_predict_norm_x = np.mean(val_data['dx']**2)
#     else:
#         sindy_predict_norm_x = np.mean(val_data['ddx']**2)

#     validation_losses = []
#     sindy_model_terms = [np.sum(params['coefficient_mask'])]

#     print('TRAINING_PART2')
#     for i in tqdm(range(params['max_epochs']), desc='Training'):
#         for j in range(params['epoch_size']//params['batch_size']):
#             batch_idxs = np.arange(j*params['batch_size'], (j+1)*params['batch_size'])
#             train_dict = create_feed_dictionary(training_data, params, idxs=batch_idxs)
#             train_op.zero_grad()
#             outputs = autoencoder_network(train_dict['x:0'])
#             loss_value = loss(outputs, train_dict['x:0'])
#             loss_value.backward()
#             train_op.step()

#         if params['print_progress'] and (i % params['print_frequency'] == 0):
#             validation_losses.append(print_progress(autoencoder_network, i, loss, losses, train_dict, validation_dict, x_norm, sindy_predict_norm_x))

#         if params['sequential_thresholding'] and (i % params['threshold_frequency'] == 0) and (i > 0):
#             params['coefficient_mask'] = torch.abs(autoencoder_network['sindy_coefficients']) > params['coefficient_threshold']
#             validation_dict['coefficient_mask:0'] = params['coefficient_mask']
#             print('THRESHOLDING: %d active coefficients' % torch.sum(params['coefficient_mask']))
#             sindy_model_terms.append(torch.sum(params['coefficient_mask']))

#     print('REFINEMENT')
#     for i_refinement in range(params['refinement_epochs']):
#         for j in range(params['epoch_size']//params['batch_size']):
#             batch_idxs = np.arange(j*params['batch_size'], (j+1)*params['batch_size'])
#             train_dict = create_feed_dictionary(training_data, params, idxs=batch_idxs)
#             train_op_ref.zero_grad()
#             outputs = autoencoder_network(train_dict['x:0'])
#             loss_value = loss_refinement(outputs, train_dict['x:0'])
#             loss_value.backward()
#             train_op_ref.step()

#         if params['print_progress'] and (i_refinement % params['print_frequency'] == 0):
#             validation_losses.append(print_progress(autoencoder_network, i_refinement, loss_refinement, losses, train_dict, validation_dict, x_norm, sindy_predict_norm_x))

#     torch.save(autoencoder_network.state_dict(), params['data_path'] + params['save_name'])
#     pickle.dump(params, open(params['data_path'] + params['save_name'] + '_params.pkl', 'wb'))
#     final_losses = loss(autoencoder_network(validation_dict['x:0']), validation_dict['x:0'])
#     if params['model_order'] == 1:
#         sindy_predict_norm_z = np.mean(autoencoder_network(validation_dict['x:0'])**2)
#     else:
#         sindy_predict_norm_z = np.mean(autoencoder_network(validation_dict['x:0'])**2)
#     sindy_coefficients = autoencoder_network['sindy_coefficients'].detach().numpy()

#     results_dict = {}
#     results_dict['num_epochs'] = i
#     results_dict['x_norm'] = x_norm
#     results_dict['sindy_predict_norm_x'] = sindy_predict_norm_x
#     results_dict['sindy_predict_norm_z'] = sindy_predict_norm_z
#     results_dict['sindy_coefficients'] = sindy_coefficients
#     results_dict['loss_decoder'] = final_losses
#     results_dict['validation_losses'] = np.array(validation_losses)
#     results_dict['sindy_model_terms'] = np.array(sindy_model_terms)

#     return results_dict

# def print_progress(network, i, loss_fn, train_dict, validation_dict, x_norm, sindy_predict_norm):
#     with torch.no_grad():
#         training_loss_vals = loss_fn(network(train_dict['x:0']), train_dict['x:0'])
#         validation_loss_vals = loss_fn(network(validation_dict['x:0']), validation_dict['x:0'])
#         print("Epoch %d" % i)
#         print("   training loss {0}, {1}".format(training_loss_vals[0], training_loss_vals[1:]))
#         print("   validation loss {0}, {1}".format(validation_loss_vals[0], validation_loss_vals[1:]))
#         decoder_losses = loss_fn(network(validation_dict['x:0']), validation_dict['x:0'])
#         loss_ratios = (decoder_losses[0]/x_norm, decoder_losses[1]/sindy_predict_norm)
#         print("decoder loss ratio: %f, decoder SINDy loss  ratio: %f" % loss_ratios)
#         return validation_loss_vals

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
