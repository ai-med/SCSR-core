import os
import shutil
import yaml
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import wandb
import joblib
import argparse
import time


# Function to load configuration files
def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_config_res_to_csv(config, best_loss_checkpoint, csv_filepath):
    # Select only the relevant training parameters
    training_params = {
        'epoch': best_loss_checkpoint['epoch'],
        'test_loss': best_loss_checkpoint['test_loss']
    }
    
    # Merge the config dictionary and training parameters into a single dictionary
    combined_dict = {**config, **training_params}
    df = pd.DataFrame([combined_dict])
    df.to_csv(csv_filepath, index=False)


# Function to build the model from configuration
def build_model_from_config(config):
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


def add_jitter(inputs, std_dev=0.01):
    """ Adds Gaussian noise (jitter) to the inputs."""
    noise = torch.randn_like(inputs) * std_dev
    return inputs + noise


def train(model, train_loader, test_loader, num_epochs, region_indices, always_mask_indices):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=wandb.config.T_max)
    
    best_auc = 0.0
    best_loss = 10.0
    best_loss_ukb = 10.0
    best_auc_checkpoint = {}
    best_loss_ukb_checkpoint = {}
    model.train()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        train_loss = 0.0
        for data in train_loader:
            full_inputs = data[0].to(device)  
            age_sex = full_inputs[:, :2]      
            thickness_data = full_inputs[:, 2:]  

            jittered_inputs = add_jitter(thickness_data, wandb.config.jitter_sd)
            corrupted_inputs, mask = corrupt_data(jittered_inputs, wandb.config.corruption_rate_train, wandb.config.use_region, region_indices, always_mask_indices)
            final_inputs = torch.cat([age_sex, corrupted_inputs], dim=1)

            optimizer.zero_grad()
            outputs = model(final_inputs)  

            full_mask = torch.cat([torch.ones(age_sex.shape[0], 2).bool().to(device), mask], dim=1)
            #full_mask = torch.cat([torch.zeros(age_sex.shape[0], 2).bool().to(device), mask], dim=1)  # ignores age and sex in loss
            loss = criterion(outputs[full_mask], full_inputs[full_mask])
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        train_loss /= len(train_loader)

        test_loss = evaluate_test_loss(model, criterion, test_loader, device, region_indices, always_mask_indices)
        epoch_duration = time.time() - epoch_start_time

        if test_loss < best_loss_ukb:
            best_loss_ukb = test_loss
            best_loss_ukb_checkpoint = {'epoch': epoch + 1,'state_dict': model.state_dict(),'optimizer': optimizer.state_dict(),'scheduler': scheduler.state_dict(),
                                   'test_loss': test_loss}            
            print(f"Saved new best model with UKB test Loss: {best_loss_ukb:.4f}")


        if (epoch % 50 == 0 and epoch > 1) or (epoch == num_epochs - 1):  # Save every 50 epochs and the last epoch
            torch.save(best_loss_ukb_checkpoint, f'checkpoints/{wandb.run.name}_bestLossUKB.pth')
            save_config_res_to_csv(training_config,best_loss_ukb_checkpoint, f'checkpoints/{wandb.run.name}_bestLossUKB.csv')
            print("Saving best models to disk.")



def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)


def evaluate_test_loss(model, criterion, data_loader, device, region_indices, always_mask_indices):
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for data in data_loader:
            full_inputs = data[0].to(device)  
            age_sex = full_inputs[:, :2]      
            thickness_data = full_inputs[:, 2:]              

            corrupted_inputs, mask = corrupt_data(thickness_data, wandb.config.corruption_rate_test, wandb.config.use_region, region_indices, always_mask_indices)
            final_inputs = torch.cat([age_sex, corrupted_inputs], dim=1)
            outputs = model(final_inputs)
            
            full_mask = torch.cat([torch.ones(age_sex.shape[0], 2).bool().to(device), mask], dim=1)
            loss = criterion(outputs[full_mask], full_inputs[full_mask])

            total_loss += loss.item() * thickness_data.size(0)  
            total_count += thickness_data.size(0)

    average_loss = total_loss / total_count
    return average_loss


def evaluate(model, test_loader, scaler):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for data in test_loader:
            inputs = data[0].to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
            actuals.append(data[1].cpu().numpy())
    
    # Concatenate all batches
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    
    # Inverse transform to original scale
    predictions_original = scaler.inverse_transform(predictions)
    actuals_original = scaler.inverse_transform(actuals)
    
    # Calculate metrics, e.g., MSE
    mse = np.mean((predictions_original - actuals_original)**2)
    print(f'Test MSE: {mse}')



def evaluate_with_masking(model, test_loader, scaler, device, num_predictions, region_indices, always_mask_indices):
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
                corrupted_thickness, mask = corrupt_data(thickness_data, wandb.config.corruption_rate_test, wandb.config.use_region, region_indices, always_mask_indices)
                full_corrupted_inputs = torch.cat([age_sex, corrupted_thickness], dim=1)
                outputs = model(full_corrupted_inputs)
                thickness_outputs = outputs[:, 2:]
                thickness_outputs[~mask] = float('nan') 
                sample_predictions_tensors.append(thickness_outputs)
            
            predictions_tensor = torch.stack(sample_predictions_tensors)
            quantile_predictions = torch.nanquantile(predictions_tensor, wandb.config.percentile, dim=0, keepdim=False)
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
    return final_predictions, mse, residuals




parser = argparse.ArgumentParser(description='Load a YAML configuration file.')
parser.add_argument('config_file', type=str, help='Name of the YAML configuration file.')
parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use. Default is 0.')
args = parser.parse_args()

training_config = load_config('training_configs/' + args.config_file)
print(f"Loaded configuration from {args.config_file}")

# Load training configuration and network configuration
#training_config = load_config('training_configs/' + 'training_config1.yaml')
network_config = load_config('model/' + training_config['network_config_file'])

os.environ['WANDB_MODE'] = 'online'
#os.environ['WANDB_MODE'] = 'disabled'
wandb.init(project="DAE_MLP", config=training_config)
config_copy = f'checkpoints/{wandb.run.name}_conf.yaml'
shutil.copy('training_configs/' + args.config_file, config_copy)
with open(config_copy, 'a') as f:
    f.write(f'orig_conf_file: {args.config_file}')
print(training_config)
harmonization = training_config['harmonization']


if harmonization:
    table_path = 'XXX.feather'
    print('Harmonized data loaded')
    scaler_name = 'UKB_stdScaler_10K_harmonized.joblib'
else:
    table_path = 'XXX.feather'
    scaler_name = 'UKB_stdScaler_10K.joblib'
data_matrix = pd.read_feather(table_path).values
aparc_names = np.load('tables/aparc_labels_10K.npy', allow_pickle=True)


print(data_matrix.shape)    

# regions that are ignored in the reconstruction (entirely removed)
aparc_ignore = ['unknown','isthmuscingulate','rostralanteriorcingulate','caudalanteriorcingulate','posteriorcingulate','corpuscallosum']
#aparc_ignore = []
unique_regions, region_indices = np.unique(aparc_names, return_inverse=True)
remove_mask = np.array([name in aparc_ignore for name in aparc_names])
aparc_names = aparc_names[~remove_mask]
region_indices = region_indices[~remove_mask]
unique_region_indices = np.unique(region_indices)
print(f"removed: {remove_mask.sum()}")

# regions that never serve as predictors
# NOTE: ensure that corruption_rate is high enough to have enough parcels remaining for random selection
if training_config['mask_AD_ROI'] == True:
    aparc_always_masked = ['entorhinal', 'inferiortemporal', 'middletemporal', 'inferiorparietal', 'fusiform']
else:
    aparc_always_masked = []
#aparc_always_masked = training_config['aparc_always_masked']
print(aparc_always_masked)
always_mask_indices = np.array([np.where(unique_regions == region)[0][0] for region in aparc_always_masked if region in unique_regions])
num_remaining_parcels = len(unique_region_indices)
num_parcels_to_mask = int(num_remaining_parcels * wandb.config.corruption_rate_train)
num_rand_parcels = num_parcels_to_mask - len(always_mask_indices)
print(f"Total parcels: {num_remaining_parcels}. Parcels to mask: {num_parcels_to_mask}. Always masked: {len(always_mask_indices)}. Randomly pick {num_rand_parcels} from {num_remaining_parcels-len(always_mask_indices)}.")

if harmonization:
    age = data_matrix[:, 1].reshape(-1, 1).astype(np.float32)
    sex = data_matrix[:, 2].reshape(-1, 1)    
    X_train, X_test, as_train, as_test = train_test_split(data_matrix[:, 4:].astype(np.float32), np.hstack((age, sex)), test_size=0.2, random_state=42)
else:
    age = data_matrix[:, 0].reshape(-1, 1).astype(np.float32)
    sex = data_matrix[:, 1].reshape(-1, 1)
    X_train, X_test, as_train, as_test = train_test_split(data_matrix[:, 5:].astype(np.float32), np.hstack((age, sex)), test_size=0.2, random_state=42)


# Remove regions from aparc_ignore
X_train = X_train[:, ~remove_mask]
X_test = X_test[:, ~remove_mask]

del data_matrix
print(X_train.shape)

# Combine preprocessing for age normalization and sex encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('age', StandardScaler(), [0]),  # Normalize age
        ('sex', OneHotEncoder(drop='if_binary'), [1])  # Convert sex to binary format
    ])
as_train_normalized = preprocessor.fit_transform(as_train)
as_test_normalized = preprocessor.transform(as_test)
as_scaler_name = scaler_name.replace('UKB_', 'UKB_AS_')
joblib.dump(preprocessor, as_scaler_name)

if training_config['use_age_sex'] == False:
    as_train_normalized = np.zeros(as_train_normalized.shape)
    as_test_normalized = np.zeros(as_test_normalized.shape)

# Fit the scaler on the training data and transform both training and testing data
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)
joblib.dump(scaler, scaler_name)
print('scaler saved')
del X_train, X_test

# Combine Age & Sex with thickness data
X_train_normalized = np.hstack((as_train_normalized, X_train_normalized))
X_test_normalized = np.hstack((as_test_normalized, X_test_normalized))

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_normalized, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_normalized, dtype=torch.float32)
input_dim = X_train_normalized.shape[1]
del X_train_normalized, X_test_normalized 

# Create datasets
train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
test_dataset = TensorDataset(X_test_tensor, X_test_tensor)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=wandb.config.batch_size, shuffle=False)

# Define the model based on the loaded network configuration
network_config['layers'][0]['in_features'] = input_dim
network_config['layers'][-1]['out_features'] = input_dim
model = build_model_from_config(network_config)
print(model)
wandb.watch(model, log_freq=2)

# Device configuration for CUDA or CPU
#device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
model.to(device)


train(model, train_loader, test_loader, wandb.config.epochs,region_indices, always_mask_indices)

# Calculate the total number of parameters and trainable params
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Parameters: {total_params}")
print(f"Trainable Parameters: {trainable_params}")

del X_train_tensor, train_dataset, train_loader
torch.cuda.empty_cache()  # Frees up unused memory from PyTorch's allocator (for GPU)

sd_pop = np.nanstd(res_test, axis=0)
sd_pop_name = f'{wandb.run.name}_sd_pop.npy'
np.save('variances/' + sd_pop_name, sd_pop)
mse_sd = np.mean((res_test / sd_pop)**2)

