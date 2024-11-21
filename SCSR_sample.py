import yaml
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import joblib
import os 


n_samples = 1
print(f'##### SAMPLING {n_samples} EXAMPLES #####')


def load_config(file_path):
    """ Loads a configuration file in YAML format."""
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def build_model_from_config(config):
    """Builds a PyTorch model from a configuration dictionary."""
    layers = []
    last_out_features = None
    for layer in config['layers']:
        if layer['type'] == 'Linear':
            linear_layer = nn.Linear(layer['in_features'], layer['out_features'])
            layers.append(linear_layer)
            last_out_features = layer['out_features']  # Track the last layer's output features
        elif layer['type'] == 'BatchNorm1d':
            layers.append(nn.BatchNorm1d(last_out_features))  # Apply to the last output features
        elif layer['type'] == 'SiLU':
            layers.append(nn.SiLU())
        elif layer['type'] == 'Dropout':
            layers.append(nn.Dropout(p=layer.get('p', 0.5)))  # Use .get for optional parameters
    return nn.Sequential(*layers)


def corrupt_data(data, corruption_rate, use_region, region_indices, always_mask_indices):
    """Corrupts data by masking a portion of it with zeros."""
    corrupted = data.clone()

    # Either region-wise corruption or random corruption
    if use_region:
        # initialize mask with size of data as binary
        mask = torch.zeros_like(data, dtype=torch.bool, device=data.device)

        #unique_region_indices = np.unique(region_indices)
        num_regions_to_corrupt = int(len(unique_region_indices) * corruption_rate) - len(always_mask_indices)
        possible_indices_to_corrupt = np.setdiff1d(unique_region_indices, always_mask_indices)

        for i in range(data.shape[0]):  # Iterate over rows
            indices_to_corrupt = np.random.choice(possible_indices_to_corrupt, num_regions_to_corrupt, replace=False)
            final_indices_to_corrupt = np.union1d(indices_to_corrupt, always_mask_indices)
            mask[i,] = torch.tensor(np.isin(region_indices, final_indices_to_corrupt)) #.to(device)       
            
    else:
        if len(always_mask_indices) > 0:   # always mask AD regions
            always_mask = torch.tensor(np.isin(region_indices, always_mask_indices)).to(device)
            replicated_mask = always_mask.repeat(data.shape[0], 1)
            mask = (torch.rand(corrupted.shape, device=device) < corruption_rate) | replicated_mask
        else:
            #mask = torch.rand(corrupted.shape) < corruption_rate
            mask = (torch.rand(corrupted.shape, device=device) < corruption_rate)

    corrupted[mask] = 0
   
    return corrupted, mask


def nanstd_torch(input_tensor, axis=0):
    mask = ~torch.isnan(input_tensor)
    mean = torch.sum(input_tensor * mask, axis=axis) / mask.sum(axis=axis)
    variance = torch.sum(((input_tensor - mean.unsqueeze(axis)) * mask) ** 2, axis=axis) / mask.sum(axis=axis)
    std_dev = torch.sqrt(variance)
    return std_dev


def evaluate_with_masking(model, test_loader, scaler, device, num_predictions, region_indices, always_mask_indices, corruption_rate, percentile, use_region, aparc_names, compute_std):
    model.eval()
    final_predictions = []
    actuals = []
    
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            age_sex = inputs[:, :2].to(device)  
            thickness_data = inputs[:, 2:].to(device)  
            
            # Store predictions for each sample across 30 random masks
            sample_predictions_tensors = []
            for _ in range(num_predictions):
                corrupted_thickness, mask = corrupt_data(thickness_data, corruption_rate, use_region, region_indices, always_mask_indices)
                full_corrupted_inputs = torch.cat([age_sex, corrupted_thickness], dim=1)
                outputs = model(full_corrupted_inputs)
                thickness_outputs = outputs[:, 2:]
                thickness_outputs[~mask] = float('nan') 
                sample_predictions_tensors.append(thickness_outputs)
            
            predictions_tensor = torch.stack(sample_predictions_tensors)
            quantile_predictions = torch.nanquantile(predictions_tensor, percentile, dim=0, keepdim=False)
            final_predictions.append(quantile_predictions)
            actuals.append(labels[:, 2:].to(device))


    final_predictions_tensor = torch.cat(final_predictions, dim=0)
    actuals_tensor = torch.cat(actuals, dim=0) 
   # Move the final predictions tensor to CPU and convert to NumPy for inverse transform
    final_predictions_np = final_predictions_tensor.cpu().numpy()
    final_predictions_inv = scaler.inverse_transform(final_predictions_np)

    # Also prepare the actuals for comparison
    actuals_np = actuals_tensor.cpu().numpy()
    actuals_original = scaler.inverse_transform(actuals_np)
    residuals = actuals_original - final_predictions_inv

    mse = np.mean((residuals)**2)
    print(f'Test MSE: {mse}')
    return final_predictions_inv, actuals_original


def restore_array(to_restore, num_cols_restored, index_mapping):
    to_restore_array = to_restore.values
    restored_array = np.zeros((to_restore_array.shape[0], num_cols_restored)) 
    
    # Map the new indices to the original indices
    for new_idx, orig_idx in index_mapping.items():
        restored_array[:, orig_idx] = to_restore_array[:, new_idx]    
    
    print(restored_array.shape)
    return restored_array


def save_restored_array(restored_array, restored_index, output_path='resultsVariance/percentages/'):
    # Get unique combinations of DX and Dataset
    combinations = set(restored_index)
    
    # Save each combination as a separate file
    for (dx, dataset) in combinations:
        # Find the index for the row corresponding to this combination
        idx = restored_index.get_loc((dx, dataset))

        output_filename = f"{output_path}percentages_{dx}_{dataset}.npy"
        np.save(output_filename, restored_array[idx, :])


corruption_rates = [0.8]
percentiles = [0.95]

num_predictions=100

wandb_name = 'toasty-disco-150'
best_model = 'bestLossUKB'

test_on_cth = False

conf_name = wandb_name + '_conf.yaml'
training_config = load_config('checkpoints/' + conf_name)
network_config = load_config('model/' + training_config['network_config_file'])

print(f'Corruption rate {corruption_rates}, percentile {percentiles}, configuration {conf_name}')
AD_ROI = ['entorhinal', 'inferiortemporal', 'middletemporal', 'inferiorparietal', 'fusiform']

harmonization = training_config.get('harmonization', False)

val_matrix_file = 'data/sample_data.feather'
subject_id = None
aparc_names = np.load('tables/aparc_labels_10K.npy', allow_pickle=True)

if harmonization:
    scaler_name = 'UKB_stdScaler_10K_harmonized.joblib'
else:
    scaler_name = 'UKB_stdScaler_10K.joblib'

outputPath = f"samples/{wandb_name}_{val_matrix_file.split('/')[-1].replace('.feather', '')}/"

if not os.path.exists(outputPath):
    os.makedirs(outputPath)

val_matrix = pd.read_feather(val_matrix_file)        

# Remove diagnosis column as we are not using it
val_matrix.drop('DX', axis=1, inplace=True)

print(val_matrix.shape)    
scaler = joblib.load(scaler_name)
sd_pop = np.load('variances/' + f'{wandb_name}_sd_pop.npy')

# regions that are ignored in the reconstruction (entirely removed)
aparc_ignore = ['unknown','isthmuscingulate','rostralanteriorcingulate','caudalanteriorcingulate','posteriorcingulate','corpuscallosum']
unique_regions, region_indices = np.unique(aparc_names, return_inverse=True)
remove_mask = np.array([name in aparc_ignore for name in aparc_names])

original_indices = np.arange(len(aparc_names))
filtered_indices = original_indices[~remove_mask]
index_mapping = {new_idx: orig_idx for new_idx, orig_idx in enumerate(filtered_indices)}

aparc_names = aparc_names[~remove_mask]
region_indices = region_indices[~remove_mask]
unique_region_indices = np.unique(region_indices)
print(f"removed: {remove_mask.sum()}")

# regions that are always masked, ie, never serve as predictors
# NOTE: ensure that corruption_rate is high enough to have enough parcels remaining for random selection
if training_config['mask_AD_ROI'] == True:
    aparc_always_masked = ['entorhinal', 'inferiortemporal', 'middletemporal', 'inferiorparietal', 'fusiform']
else:
    aparc_always_masked = []
print(aparc_always_masked)
always_mask_indices = np.array([np.where(unique_regions == region)[0][0] for region in aparc_always_masked if region in unique_regions])
age_val = val_matrix.iloc[:, 1].values.reshape(-1, 1).astype(np.float32)  
sex_val = val_matrix.iloc[:, 2].values.reshape(-1, 1)  
X_val = val_matrix.iloc[:, 3:].values.astype(np.float32)  
X_val = X_val[:, ~remove_mask]
print(X_val.shape)

if training_config['use_age_sex']:
    as_scaler_name = scaler_name.replace('UKB_', 'UKB_AS_')
    preprocessor = joblib.load(as_scaler_name)
    as_val_normalized = preprocessor.transform(np.hstack((age_val,sex_val)))
else:
    preprocessor = joblib.load(scaler_name)
    as_val_normalized = np.zeros((X_val.shape[0], 2))
X_val_normalized = scaler.transform(X_val)
X_val_normalized = np.hstack((as_val_normalized, X_val_normalized))
X_val_tensor = torch.tensor(X_val_normalized, dtype=torch.float32)

if subject_id is not None:
    X_val_tensor = X_val_tensor[subject_id][None]
val_dataset = TensorDataset(X_val_tensor, X_val_tensor)


val_loader = DataLoader(val_dataset[:n_samples], batch_size=32, shuffle=False)

input_dim = X_val_normalized.shape[1]
del X_val, X_val_normalized, X_val_tensor

print('Using MLP model')
network_config['layers'][0]['in_features'] = input_dim
network_config['layers'][-1]['out_features'] = input_dim
model = build_model_from_config(network_config)
print(model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = wandb_name + '_' + best_model + '.pth'
snapshot = torch.load("checkpoints/" + model_name, map_location=device)
model_state_dict = snapshot['state_dict']
model.load_state_dict(model_state_dict)
model = model.to(device)
model.eval()

use_region = training_config['use_region']

rec_error_all = None
vertex_rec_error_all = None
percent_entorhinal_all = None
stats_all = None # pd.DataFrame()
auc_all = None

for corruption_rate in corruption_rates:
    for percentile in percentiles:
        print(f'Corruption rate {corruption_rate}, percentile {percentile}')
        compute_std = False
        predictions, targets = evaluate_with_masking(model, val_loader, scaler, device, num_predictions, region_indices, always_mask_indices, corruption_rate, percentile, use_region, aparc_names, compute_std)

        zscores = (targets - predictions) / sd_pop

        zscores_output = -np.ones((n_samples, 10242))
        zscores_output[:, ~remove_mask] = zscores

        pred_output = -np.ones((n_samples, 10242))
        pred_output[:, ~remove_mask] = predictions

        target_output = -np.ones((n_samples, 10242))
        target_output[:, ~remove_mask] = targets

        np.save(outputPath + f'zscores_corr{corruption_rate}_perc{percentile}.npy', zscores_output)
        np.save(outputPath + f'predictions_corr{corruption_rate}_perc{percentile}.npy', pred_output)
        np.save(outputPath + f'targets_corr{corruption_rate}_perc{percentile}.npy', target_output)

        print(f'Output saved to {outputPath}')
