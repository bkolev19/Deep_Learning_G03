import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import odeint
from tqdm import tqdm
import numpy as np
from autoencoder import Autoencoder, sindy_library_torch, sindy_library_torch_order2
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0" # GPU index

print("Available GPUs:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

cuda_id = torch.cuda.current_device()
print(f"ID of current CUDA device:{torch.cuda.current_device()}")
print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)

torch.manual_seed(seed=42)
torch.cuda.manual_seed(seed=42)



def move_tensors_to_device(data_dict, device):
    """
    Move all tensors within a dictionary to the specified device.
    """
    for key in data_dict:
        if isinstance(data_dict[key], torch.Tensor):
            data_dict[key] = data_dict[key].to(device)
    return data_dict


def create_feed_dictionary(data, params, idxs=None):
    if idxs is None:
        idxs = np.arange(data['x'].shape[0])
    feed_dict = {
        'x': torch.tensor(data['x'][idxs], dtype=torch.float32),
        'dx': torch.tensor(data['dx'][idxs], dtype=torch.float32)
        }
    if params['model_order'] == 2:
        feed_dict['ddx'] = torch.tensor(data['ddx'][idxs], dtype=torch.float32)
    if params['sequential_thresholding']:
        feed_dict['coefficient_mask'] = torch.tensor(params['coefficient_mask'], dtype=torch.float32)
    feed_dict['learning_rate'] = torch.tensor(params['learning_rate'], dtype=torch.float32)
    return feed_dict



def train_network(training_data, val_data, params, device: torch.device = device):
    training_data   = move_tensors_to_device(training_data ,device)
    val_data        = move_tensors_to_device(val_data, device)
    params          = move_tensors_to_device(params, device)

    model       = Autoencoder(params)
    # model.train()
    optimizer   = torch.optim.Adam(model.parameters(), params['learning_rate'])
    batch_iter  = params['epoch_size']//params['batch_size']

    validation_dict   = create_feed_dictionary(val_data, params, idxs=None)
    x_norm            = np.mean(val_data['x']**2)
    
    if params['model_order'] == 1:
        sindy_predict_norm_x = np.mean(val_data['dx']**2)
    else:
        sindy_predict_norm_x = np.mean(val_data['ddx']**2)

    validation_losses = []
    final_losses      = []
    refined_losses    = []
    ref_val_loss      = []
    sindy_model_terms = [np.sum(params['coefficient_mask'])]



    for i in tqdm(range(params['max_epochs']), desc='Epochs_Loop'):
        for j in (range(batch_iter)):
            batch_idxs = np.arange(j*params['batch_size'], (j+1)*params['batch_size'])
            train_dict = create_feed_dictionary(training_data, params, idxs=batch_idxs)
            x          = train_dict['x']
            dx         = train_dict['dx']

            model.forward([x, dx])
            loss = model.loss_func()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if params['print_progress'] and (i % params['print_frequency'] == 0):
            validation_losses.append(loss.detach().cpu().numpy())

        # HERE ALL THE MODEL.MASK USED TO BE PARAMS['COEFFICIENT_MASK'], BUT THAT DOES NOT UPDATE THE MODEL
        if params['sequential_thresholding'] and (i % params['threshold_frequency'] == 0) and (i > 0):
            # Get numpy copy from tensor model.sindy_coefficients and save different stuff. DONT TOUCH
            sindy_coeffs_copy = model.sindy_coefficients.clone().detach()
            temp_mask = np.abs(sindy_coeffs_copy) > params['coefficient_threshold']
            model.mask = temp_mask
            temp_mask = temp_mask.numpy()
            params['coefficient_mask'] = temp_mask
            validation_dict['coefficient_mask:0'] = temp_mask
            print('THRESHOLDING: %d active coefficients' % np.sum(temp_mask))
            sindy_model_terms.append(np.sum(temp_mask))
        
        final_losses.append(loss.detach().cpu().numpy())

    model.loss_func = model.custom_loss_refined ## ADD THIS FOR REFINED!!!
    for i in tqdm(range(params['refinement_epochs']), desc='Refined_Epochs_Loop'):
        for j in (range(batch_iter)):
            batch_idxs = np.arange(j*params['batch_size'], (j+1)*params['batch_size'])
            train_dict = create_feed_dictionary(training_data, params, idxs=batch_idxs)
            x          = train_dict['x']
            dx         = train_dict['dx']

            model.forward([x, dx])
            loss = model.loss_func()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if params['print_progress'] and (i % params['print_frequency'] == 0):
            ref_val_loss.append(loss.detach().cpu().numpy())   
        
        refined_losses.append(loss.detach().cpu().numpy())

    # Save the model
    MODEL_PATH = params['data_path'] + params['save_name']
    torch.save(model.state_dict(), (MODEL_PATH + '.pth'))

    if params['model_order'] == 1:
        sindy_predict_norm_z = torch.mean((model.sindy_dz)**2).item()
    else:
        sindy_predict_norm_z = torch.mean((model.sindy_ddz)**2).item()
    sindy_coefficients = model.sindy_coefficients
    

    results_dict = {}
    results_dict['num_epochs']                  = i
    results_dict['x_norm']                      = x_norm
    results_dict['sindy_predict_norm_x']        = sindy_predict_norm_x
    results_dict['sindy_predict_norm_z']        = sindy_predict_norm_z
    results_dict['sindy_coefficients']          = sindy_coefficients
    results_dict['loss_decoder']                = final_losses
    results_dict['validation_losses']           = validation_losses
    results_dict['sindy_model_terms']           = sindy_model_terms
    results_dict['refined_losses']              = refined_losses
    results_dict['refined_validation_losses'] = ref_val_loss

    return results_dict



def sindy_simulate(x0, t, Xi, poly_order, include_sine):
    m = t.size
    n = x0.size
    a_0 = sindy_library_torch(torch.tensor(x0).reshape((1,n)), poly_order, include_sine)
    Xi = torch.tensor(Xi)
    # f = lambda x,t : np.dot(sindy_library_torch(torch.tensor(x).reshape((1,n)), poly_order, include_sine), Xi).reshape((n,))
    f = lambda x, t: torch.matmul(sindy_library_torch(torch.tensor(x).reshape((1, n)), poly_order, include_sine), Xi).reshape((n,))

    x = odeint(f, x0, t)
    return x


def sindy_simulate_order2(x0, dx0, t, Xi, poly_order, include_sine):
    m = t.size
    n = 2*x0.size
    l = Xi.shape[0]

    Xi_order1 = torch.zeros((l,n))
    for i in range(n//2):
        Xi_order1[2*(i+1),i] = 1.
        Xi_order1[:,i+n//2] = Xi[:,i]
    
    x = sindy_simulate(np.concatenate((x0,dx0)), t, Xi_order1, poly_order, include_sine)
    return x