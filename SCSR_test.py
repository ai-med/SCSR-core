import yaml
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import spearmanr, kendalltau
import joblib
import nibabel.freesurfer.io as fsio
import os 
from itertools import combinations
from scipy.stats import mannwhitneyu


# Function to load configuration files
def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def compute_auc(data, class1, class2, use_AUC_AD_ROI=True):
    if data.empty or data['DX'].nunique() < 2:
        return None  # Skip if dataset is empty or classes are not present
    #data['DX_encoded'] = LabelEncoder().fit_transform(data['DX'] == class1)
    #data['ROI'] = data[AD_ROI].sum(axis=1)
    
    data = data.copy()
    data.loc[:, 'DX_encoded'] = LabelEncoder().fit_transform(data['DX'] == class1)
    if use_AUC_AD_ROI:
        data.loc[:, 'ROI'] = data.loc[:, AD_ROI].sum(axis=1)
    else:
        data.loc[:, 'ROI'] = data.loc[:, np.unique(aparc_names)].sum(axis=1)

    roc_pos = roc_auc_score(data['DX_encoded'], data['ROI'])
    roc_neg = roc_auc_score(data['DX_encoded'], -data['ROI'])
    return max(roc_pos, roc_neg)


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
    all_std = []
    
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
            
            #std_predictions = nanstd_torch(predictions_tensor, axis=0)
            #all_std.append(std_predictions)
            if compute_std:
                predictions_tensor_np = predictions_tensor.cpu().numpy()
                predictions_original = scaler.inverse_transform(predictions_tensor_np)
                predictions_std = np.nanstd(predictions_original, axis=0)
                all_std.append(predictions_std)


    final_predictions_tensor = torch.cat(final_predictions, dim=0)
    actuals_tensor = torch.cat(actuals, dim=0) 
   # Move the final predictions tensor to CPU and convert to NumPy for inverse transform
    final_predictions_np = final_predictions_tensor.cpu().numpy()
    final_predictions_inv = scaler.inverse_transform(final_predictions_np)

    # Also prepare the actuals for comparison
    actuals_np = actuals_tensor.cpu().numpy()
    actuals_original = scaler.inverse_transform(actuals_np)
    residuals = actuals_original - final_predictions_inv

    actuals_original_df = pd.DataFrame(actuals_original, columns=aparc_names)
    actual_original_by_region = actuals_original_df.groupby(by=actuals_original_df.columns, axis=1).mean()
    #actual_original_by_region = actuals_original_df.T.groupby(aparc_names).mean().T
    final_predictions_inv_df = pd.DataFrame(final_predictions_inv, columns=aparc_names)
    final_predictions_inv_by_region = final_predictions_inv_df.groupby(by=final_predictions_inv_df.columns, axis=1).mean()
    #final_predictions_inv_by_region = final_predictions_inv_df.T.groupby(aparc_names).mean().T
    residuals_by_region = actual_original_by_region - final_predictions_inv_by_region

    mse = np.mean((residuals)**2)
    print(f'Test MSE: {mse}')
    return final_predictions_tensor, mse, residuals, residuals_by_region, all_std

def restore_array(to_restore, num_cols_restored, index_mapping):
    if isinstance(to_restore, pd.DataFrame):
        to_restore_array = to_restore.values
    elif isinstance(to_restore, np.ndarray):
        to_restore_array = to_restore
    #to_restore_array = to_restore.values
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

def save_restored_zscores(restored_array, restored_index, output_path, percentile):
    # Get unique combinations of DX and Dataset
    combinations = set(restored_index)
    
    # Save each combination as a separate file
    for (dx, dataset) in combinations:
        # Find the index for the row corresponding to this combination
        idx = restored_index.get_loc((dx, dataset))

        output_filename = f"{output_path}meanZ/zscores_{dx}_{percentile}.npy"
        np.save(output_filename, restored_array[idx, :])   # previously saved negative z-scores. changed



corruption_rates = [0.8]
percentiles = [0.95]

num_predictions=100

wandb_name = 'toasty-disco-150'
best_model = 'bestLossUKB'

test_on_cth = False
use_AUC_AD_ROI = True

outputPath = 'resultsTest/'


conf_name = wandb_name + '_conf.yaml'
training_config = load_config('checkpoints/' + conf_name)
network_config = load_config('model/' + training_config['network_config_file'])
#print(f"Loaded configuration from {conf_name}")
print(f'Corruption rate {corruption_rates}, percentile {percentiles}, configuration {conf_name}')
AD_ROI = ['entorhinal', 'inferiortemporal', 'middletemporal', 'inferiorparietal', 'fusiform']

harmonization = training_config.get('harmonization', False)


if harmonization:
    table_path = 'XXX.feather'
    
    print('Harmonized data')   
else:
    table_path = 'XXX.feather'

val_matrix = pd.read_feather(table_path)     
aparc_names = np.load('tables/aparc_labels_10K.npy', allow_pickle=True)

if harmonization:
    scaler_name = 'UKB_stdScaler_10K_harmonized.joblib'
else:
    scaler_name = 'UKB_stdScaler_10K.joblib'

print(val_matrix.shape)    
scaler = joblib.load(scaler_name)
sd_pop = np.load('variances/' + f'{wandb_name}_sd_pop.npy')

# regions that are ignored in the reconstruction (entirely removed)
aparc_ignore = ['unknown','isthmuscingulate','rostralanteriorcingulate','caudalanteriorcingulate','posteriorcingulate','corpuscallosum']
#aparc_ignore = []
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
X_val = val_matrix.iloc[:, 4:].values.astype(np.float32)  
X_val = X_val[:, ~remove_mask]
print(X_val.shape)

as_scaler_name = scaler_name.replace('UKB_', 'UKB_AS_')
preprocessor = joblib.load(as_scaler_name)
as_val_normalized = preprocessor.transform(np.hstack((age_val,sex_val)))
if training_config['use_age_sex'] == False:
    as_val_normalized = np.zeros(as_val_normalized.shape)
X_val_normalized = scaler.transform(X_val)
X_val_normalized = np.hstack((as_val_normalized, X_val_normalized))
X_val_tensor = torch.tensor(X_val_normalized, dtype=torch.float32)
val_dataset = TensorDataset(X_val_tensor, X_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

input_dim = X_val_normalized.shape[1]
del X_val, X_val_normalized, X_val_tensor

network_config['layers'][0]['in_features'] = input_dim
network_config['layers'][-1]['out_features'] = input_dim
model = build_model_from_config(network_config)
print(model)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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
        compute_std = False
        predictions, mse, residuals, residuals_by_region, stds = evaluate_with_masking(model, val_loader, scaler, device, num_predictions, region_indices, always_mask_indices, corruption_rate, percentile, use_region, aparc_names, compute_std)

        differences = pd.concat([val_matrix.iloc[:, :4], pd.DataFrame(residuals, columns=aparc_names)], axis=1)
        print(f'Corruption rate {corruption_rates}, percentile {percentiles}')

        zscores = pd.concat([val_matrix.iloc[:, :4], pd.DataFrame(residuals/sd_pop, columns=aparc_names)], axis=1)

        # Total MAE 
        residuals_means = residuals_by_region.abs().mean(axis=1)
        residuals_means_df = pd.concat([val_matrix.iloc[:, :4], residuals_means.to_frame(name='Mean Residual')], axis=1)
        rec_error_grouped = residuals_means_df.groupby(['Dataset', 'DX']).agg(
            **{f'rec_error_{corruption_rate}_{percentile}': ('Mean Residual', 'mean')}
        ).reset_index()
        if rec_error_all is None:
            rec_error_all = rec_error_grouped
        else:
            rec_error_all = rec_error_all.merge(rec_error_grouped, on=['Dataset', 'DX'])
        print(rec_error_all)

        # Absolute mean of residuals (no parcels)
        vertex_rec_error = pd.concat([
            differences.iloc[:, :4], 
            pd.DataFrame(np.abs(differences.iloc[:, 4:].values).mean(axis=1), columns=['Abs_Mean'])
        ], axis=1)
        vertex_rec_error_grouped = vertex_rec_error.groupby(['Dataset', 'DX']).agg(
            **{f'vertex_error_mean_{corruption_rate}_{percentile}': ('Abs_Mean', 'mean')}
        ).reset_index()

        if vertex_rec_error_all is None:
            vertex_rec_error_all = vertex_rec_error_grouped
        else:
            vertex_rec_error_all = vertex_rec_error_all.merge(vertex_rec_error_grouped, on=['Dataset', 'DX'])
        print(vertex_rec_error_all)


        # Zscores by parcel
        zscore_means_by_region = pd.concat([
            zscores.iloc[:, :4], 
            zscores.iloc[:, 4:].groupby(by=zscores.columns[4:], axis=1).mean()
        ], axis=1)
        zscore_means_by_region['ROI'] = zscore_means_by_region[AD_ROI].mean(axis=1, skipna=True)


        # Parcel-wise Percentage of z-scores below -2
        zscore_columns = [col for col in zscore_means_by_region.columns if col not in ['DX', 'Dataset', 'AGE', 'PTGENDER' ]]
        counts_below_minus_two = zscore_means_by_region.groupby(['DX', 'Dataset'])[zscore_columns].apply(lambda x: (x < -2).sum())
        group_sizes = zscore_means_by_region.groupby(['DX', 'Dataset']).size()
        percentages_per_feature = counts_below_minus_two.div(group_sizes, axis=0) * 100
        entorhinal_df = percentages_per_feature[['entorhinal']].reset_index()
        entorhinal_df.rename(columns={'entorhinal': f'perc_ento_{corruption_rate}_{percentile}'}, inplace=True)
        if percent_entorhinal_all is None:
            percent_entorhinal_all = entorhinal_df
        else:
            percent_entorhinal_all = percent_entorhinal_all.merge(entorhinal_df, on=['Dataset', 'DX'], how='outer', suffixes=('', '_new'))
        print(percent_entorhinal_all)        
        
        # Vertex-wise Percentage of z-scores below -2
        mask = zscores.iloc[:, 4:] < -2
        counts_below_minus_two = mask.groupby([zscores['DX'], zscores['Dataset']]).sum()
        group_sizes = zscores.groupby(['DX', 'Dataset']).size()
        percentages = counts_below_minus_two.div(group_sizes, axis=0) * 100

        # Mean z-scores by subgroup
        mean_zscores = zscores.iloc[:, 4:].groupby([zscores['DX'], zscores['Dataset']]).mean()
        mean_zscoresDX = zscores.iloc[:, 4:].groupby(zscores['DX']).mean()

        datasets = zscore_means_by_region['Dataset'].unique().tolist() + ['all']
        # Correlation between z-scores and diagnosis
        dx_mapping = {'CN': 1, 'MCI': 2, 'AD': 3}        
        zscore_means_by_region['DX_int'] = zscore_means_by_region['DX'].map(dx_mapping)

        all_corr_data = []
        for dataset in datasets:
            if dataset == 'all':
                data_subset = zscore_means_by_region
            else:
                data_subset = zscore_means_by_region[zscore_means_by_region['Dataset'] == dataset]

            spearman_corr = spearmanr(data_subset['DX_int'], data_subset['ROI'])[0]
            kendall_corr = kendalltau(data_subset['DX_int'], data_subset['ROI'])[0]
            all_corr_data.append({
                    'Dataset': dataset,
                    f'Spearman_{corruption_rate}_{percentile}': spearman_corr,
                    f'Kendall{corruption_rate}_{percentile}': kendall_corr
            })

        corr_res = pd.DataFrame(all_corr_data)
        if stats_all is None:
            stats_all = corr_res
        else:
            stats_all = pd.merge(stats_all, corr_res, on=['Dataset'], how='outer')
        print(stats_all)            


        #AUC
        # comparisons = {
        #     'AD_CN': ('AD', 'CN'),
        #     'AD_MCI': ('AD', 'MCI'),
        #     'CN_MCI': ('CN', 'MCI')
        # }
        diagnostic_groups = val_matrix['DX'].unique().tolist()
        comparisons = {f'{a}_{b}': (a, b) for a, b in combinations(diagnostic_groups, 2)}        

        label_encoder = LabelEncoder()

        all_auc_data = []
        # Iterate over each dataset and each comparison pair        
        for dataset in datasets:
            dataset_auc_scores = [] 

            if dataset == 'all':
                data_subset = zscore_means_by_region
            else:
                data_subset = zscore_means_by_region[zscore_means_by_region['Dataset'] == dataset]

            for name, (class1, class2) in comparisons.items():
                auc_score = compute_auc(data_subset[data_subset['DX'].isin([class1, class2])], class1, class2, use_AUC_AD_ROI)
                if auc_score is not None:
                    all_auc_data.append({
                        'Dataset': dataset,
                        'Comparison': name,
                        f'AUC_Score_{corruption_rate}_{percentile}': auc_score,
                    })
                    dataset_auc_scores.append(auc_score)

            if dataset_auc_scores:
                average_auc = sum(dataset_auc_scores) / len(dataset_auc_scores)
                all_auc_data.append({
                    'Dataset': dataset,
                    'Comparison': 'Pairwise_Average',
                    f'AUC_Score_{corruption_rate}_{percentile}': average_auc,
                })            

        auc_res = pd.DataFrame(all_auc_data)
        #auc_res_dict[(corruption_rate, percentile)] = auc_res
        if auc_all is None:
            auc_all = auc_res
        else:
            auc_all = pd.merge(auc_all, auc_res, on=['Dataset', 'Comparison'], how='outer')
        print(auc_all)

  

print(rec_error_all)
print(vertex_rec_error_all)
print(stats_all)
print(auc_all)
print(percent_entorhinal_all)


stats_all.to_csv(f'{outputPath}Stats_{wandb_name}_{best_model}.csv', index=True)
rec_error_all.to_csv(f'{outputPath}total_mae_{wandb_name}_{best_model}.csv', index=False)
vertex_rec_error_all.to_csv(f'{outputPath}vertex_mae_{wandb_name}_{best_model}.csv', index=False)
auc_all.to_csv(f'{outputPath}AUC_{wandb_name}_{best_model}.csv', index=False)
percent_entorhinal_all.to_csv(f'{outputPath}Perc_Ento_{wandb_name}_{best_model}.csv', index=False)


