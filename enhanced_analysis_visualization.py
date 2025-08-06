#!/usr/bin/env python3
"""
Enhanced Analysis Visualization Script

This script creates enhanced analysis plots for trained models, including:
1. Original AGOP/NFM correlation and train/val loss
2. NEW: Cosine similarities between G_i, H_i, M_i matrices

Key definitions:
- H_i := E[h_i h_i^T] (representation matrix, R)
- G_i := E[g_i g_i^T], g_i := -∇_{h_i} ℓ (gradient matrix, G)  
- M_i := W_i^T W_i (weight matrix, W, same as NFM)

The script computes pairwise cosine similarities:
- G_i vs H_i
- G_i vs M_i
- H_i vs M_i

Usage:
    python enhanced_analysis_visualization.py --models adam_20250803_173338 muon_20250804_041908 sgd_20250803_164335
    
Author: GitHub Copilot
Date: 2025-08-06
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import required modules
import dataset
import neural_model
from verify_deep_NFA import (
    clean_compiled_state_dict, get_layer_output, build_subnetwork, 
    egop, correlate, read_configs, SEED, get_best_device
)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_model_at_epoch(model_dir, epoch):
    """Load model at specific epoch"""
    # Try the original format first
    model_path = os.path.join(model_dir, 'models', f'model_epoch_{epoch}.pt')
    if os.path.exists(model_path):
        return model_path
    
    # Try the new cloud format with pattern: model_epoch_X:dataset:width:W:depth:D:act:A:nn.pth
    # Look for any file in the model_dir that matches the pattern
    import glob
    pattern = os.path.join(model_dir, f'model_epoch_{epoch}:*:nn.pth')
    matching_files = glob.glob(pattern)
    if matching_files:
        return matching_files[0]  # Return the first match
    
    # Also try looking in models subdirectory with new format (colon version)
    pattern = os.path.join(model_dir, 'models', f'model_epoch_{epoch}:*:nn.pth')
    matching_files = glob.glob(pattern)
    if matching_files:
        return matching_files[0]
    
    # Try underscore format: model_epoch_X_dataset_width_W_depth_D_act_A_nn.pth
    pattern = os.path.join(model_dir, 'models', f'model_epoch_{epoch}_*_nn.pth')
    matching_files = glob.glob(pattern)
    if matching_files:
        return matching_files[0]
    
    return None

def compute_representation_matrix(net, trainloader, layer_idx, max_samples=None):
    """
    Compute H_i := E[h_i h_i^T] (representation matrix)
    """
    print(f"Computing representation matrix H_{layer_idx}...")
    
    # Move network to CPU to ensure consistency
    net = net.cpu()
    
    # Get layer outputs (post-activation)
    layer_outputs = get_layer_output(net, trainloader, layer_idx=layer_idx, max_samples=max_samples)
    
    # Ensure outputs are on CPU
    layer_outputs = layer_outputs.cpu()
    
    # Compute H_i = E[h_i h_i^T]
    H = torch.mm(layer_outputs.T, layer_outputs) / layer_outputs.shape[0]
    
    return H

def compute_gradient_matrix(net, trainloader, layer_idx, max_samples=None):
    """
    Compute G_i := E[g_i g_i^T] where g_i := -∇_{h_i} ℓ (gradient matrix)
    """
    print(f"Computing gradient matrix G_{layer_idx}...")
    
    device = get_best_device()
    net = net.to(device)
    net.eval()
    
    criterion = nn.MSELoss()
    
    # Collect gradients w.r.t. layer activations
    gradients = []
    
    # Get dataset samples
    dataset_obj = trainloader.dataset
    if max_samples is not None:
        dataset_obj = torch.utils.data.Subset(dataset_obj, range(min(max_samples, len(dataset_obj))))
    
    batch_size = 128
    for i in range(0, len(dataset_obj), batch_size):
        end_idx = min(i + batch_size, len(dataset_obj))
        batch_data = []
        batch_labels = []
        
        for j in range(i, end_idx):
            data, label = dataset_obj[j]
            batch_data.append(data)
            batch_labels.append(label)
        
        batch_tensor = torch.stack(batch_data).to(device)
        # Handle labels properly - they might already be tensors
        if isinstance(batch_labels[0], torch.Tensor):
            batch_labels = torch.stack(batch_labels).to(device)
        else:
            batch_labels = torch.tensor(batch_labels).to(device)
        batch_size_actual = batch_tensor.shape[0]
        
        # Forward pass to get layer activation
        flattened_data = batch_tensor.view(batch_size_actual, -1)
        
        # Get activation at specified layer
        if layer_idx == 0:
            h_i = flattened_data
        elif layer_idx == 1:
            h_i = neural_model.Nonlinearity()(net.first(flattened_data))
        else:
            h_i = net.first(flattened_data)
            for l_idx, layer in enumerate(net.middle):
                h_i = layer(h_i)
                if l_idx + 1 == layer_idx:
                    h_i = neural_model.Nonlinearity()(h_i)
                    break
        
        # Enable gradient computation for h_i
        h_i.requires_grad_(True)
        
        # Forward pass from h_i to output
        if layer_idx == 0:
            output = net.first(h_i)
            for layer in net.middle:
                output = layer(output)
            output = net.last(output)
        elif layer_idx == 1:
            output = h_i
            for layer in net.middle:
                output = layer(output)
            output = net.last(output)
        else:
            output = h_i
            remaining_layers = net.middle[layer_idx-1:]
            for layer in remaining_layers:
                output = layer(output)
            output = net.last(output)
        
        # Compute loss
        if len(batch_labels.shape) == 1 and output.shape[1] > 1:
            # Classification case
            loss = criterion(output, nn.functional.one_hot(batch_labels, num_classes=output.shape[1]).float())
        else:
            # Regression case or already one-hot
            loss = criterion(output, batch_labels.float().unsqueeze(1) if len(batch_labels.shape) == 1 else batch_labels.float())
        
        # Compute gradient: g_i = -∇_{h_i} ℓ
        grad_h_i = torch.autograd.grad(loss, h_i, retain_graph=False)[0]
        g_i = -grad_h_i  # g_i := -∇_{h_i} ℓ
        
        gradients.append(g_i.detach().cpu())
    
    # Concatenate all gradients (ensure all on CPU)
    all_gradients = torch.cat(gradients, dim=0).cpu()
    
    # Compute G_i = E[g_i g_i^T]
    G = torch.mm(all_gradients.T, all_gradients) / all_gradients.shape[0]
    
    # Move network back to CPU for consistency
    net = net.cpu()
    
    return G

def compute_weight_matrix(net, layer_idx):
    """
    Compute M_i := W_i^T W_i (weight matrix, same as NFM)
    """
    print(f"Computing weight matrix M_{layer_idx}...")
    
    # Ensure network is on CPU
    net = net.cpu()
    
    params = list(net.parameters())
    W_i = params[layer_idx].data.cpu()
    
    # M_i = W_i^T W_i
    M = torch.mm(W_i.T, W_i)
    
    return M

def compute_matrix_similarities(model_path, layer_indices, max_samples=None):
    """
    Compute pairwise cosine similarities between G_i, H_i, M_i matrices
    """
    # Load model and parse config
    filename = os.path.basename(model_path)
    
    # Parse config from filename
    if ':' in filename and filename.endswith('.pth'):
        # New format: model_epoch_X:dataset:width:W:depth:D:act:A:nn.pth
        parts = filename.replace('.pth', '').split(':')
        dataset_name = parts[1] if len(parts) > 1 else 'svhn'
        
        # Extract width, depth, act from the parts
        width = 1024  # default
        depth = 5     # default  
        act_name = 'relu'  # default
        
        for i in range(len(parts)):
            if parts[i] == 'width' and i + 1 < len(parts):
                width = int(parts[i + 1])
            elif parts[i] == 'depth' and i + 1 < len(parts):
                depth = int(parts[i + 1])
            elif parts[i] == 'act' and i + 1 < len(parts):
                act_name = parts[i + 1]
    elif '_' in filename and filename.endswith('.pth'):
        # Underscore format: model_epoch_X_dataset_width_W_depth_D_act_A_nn.pth
        parts = filename.replace('.pth', '').split('_')
        dataset_name = 'svhn'  # default
        width = 1024  # default
        depth = 5     # default  
        act_name = 'relu'  # default
        
        for i in range(len(parts)):
            if parts[i] in ['svhn', 'cifar']:
                dataset_name = parts[i]
            elif parts[i] == 'width' and i + 1 < len(parts):
                width = int(parts[i + 1])
            elif parts[i] == 'depth' and i + 1 < len(parts):
                depth = int(parts[i + 1])
            elif parts[i] == 'act' and i + 1 < len(parts):
                act_name = parts[i + 1]
    else:
        # Fallback to original parsing
        path_parts = model_path.split('/')
        config_str = path_parts[-2] if len(path_parts) > 1 else path_parts[-1]
        width, depth, act_name = read_configs(config_str)
        dataset_name = 'svhn'  # Default dataset
        if 'cifar' in config_str.lower():
            dataset_name = 'cifar'

    # Setup dataset (same as training)
    if dataset_name == 'svhn':
        NUM_CLASSES = 10
        SIZE = 32
        c = 3
        dim = c * SIZE * SIZE
        trainloader, _, _ = dataset.get_svhn()
    elif dataset_name == 'cifar':
        NUM_CLASSES = 10
        SIZE = 32
        c = 3
        dim = c * SIZE * SIZE
        trainloader, _, _ = dataset.get_cifar()
    
    # Load trained model
    net = neural_model.Net(dim, width=width, depth=depth, 
                          num_classes=NUM_CLASSES, act_name=act_name)
    
    d = torch.load(model_path, map_location='cpu')
    net.load_state_dict(clean_compiled_state_dict(d['state_dict']))
    
    similarities = {}
    
    for layer_idx in layer_indices:
        print(f"\n=== Processing Layer {layer_idx} ===")
        
        try:
            # Compute matrices
            H_i = compute_representation_matrix(net, trainloader, layer_idx, max_samples)
            G_i = compute_gradient_matrix(net, trainloader, layer_idx, max_samples)
            M_i = compute_weight_matrix(net, layer_idx)
            
            # Ensure all matrices are on CPU and properly formatted
            H_i = H_i.cpu().float()
            G_i = G_i.cpu().float()
            M_i = M_i.cpu().float()
            
            # Compute pairwise cosine similarities
            sim_GH = correlate(G_i, H_i)  # G_i vs H_i
            sim_GM = correlate(G_i, M_i)  # G_i vs M_i  
            sim_HM = correlate(H_i, M_i)  # H_i vs M_i
            
            # Convert to Python scalars safely - handle different tensor shapes
            def safe_scalar_conversion(tensor_val):
                if isinstance(tensor_val, torch.Tensor):
                    if tensor_val.numel() == 1:
                        return tensor_val.item()
                    else:
                        # If tensor has multiple elements, take the first one or handle appropriately
                        return tensor_val.flatten()[0].item()
                else:
                    return float(tensor_val)
            
            sim_GH_val = safe_scalar_conversion(sim_GH)
            sim_GM_val = safe_scalar_conversion(sim_GM)
            sim_HM_val = safe_scalar_conversion(sim_HM)
            
            similarities[layer_idx] = {
                'G_H': sim_GH_val,
                'G_M': sim_GM_val,
                'H_M': sim_HM_val
            }
            
            print(f"Layer {layer_idx} similarities:")
            print(f"  G_i vs H_i: {sim_GH_val:.6f}")
            print(f"  G_i vs M_i: {sim_GM_val:.6f}")
            print(f"  H_i vs M_i: {sim_HM_val:.6f}")
            
        except Exception as e:
            print(f"Error computing similarities for layer {layer_idx}: {e}")
            similarities[layer_idx] = {
                'G_H': 0.0,
                'G_M': 0.0,
                'H_M': 0.0
            }
    
    return similarities

def create_enhanced_analysis_plot(model_dir, output_dir):
    """
    Create enhanced analysis plot with original metrics + new matrix similarities
    """
    # Load original results
    results_path = os.path.join(model_dir, 'results.json')
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Extract model name
    model_name = os.path.basename(model_dir)
    optimizer_name = model_name.split('_')[0].upper()
    
    # Get epochs from results
    epochs = results['epochs']
    layer_indices = [0, 1, 2, 3, 4]
    
    # Compute matrix similarities for each saved epoch
    print(f"\n=== Computing matrix similarities for {model_name} ===")
    
    epoch_similarities = {}
    for epoch in epochs:
        print(f"\nProcessing epoch {epoch}...")
        model_path = load_model_at_epoch(model_dir, epoch)
        if model_path is None:
            print(f"Model not found for epoch {epoch}")
            continue
        
        similarities = compute_matrix_similarities(model_path, layer_indices, max_samples=None)
        epoch_similarities[epoch] = similarities
    
    # Create enhanced visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Enhanced Training Analysis - {optimizer_name}', fontsize=16, fontweight='bold')
    
    # Plot for each layer (5 subplots)
    plot_positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    
    for i, layer_idx in enumerate(layer_indices):
        row, col = plot_positions[i]
        ax1 = axes[row, col]
        
        # Primary y-axis: Training Loss
        losses = results['train_losses'][layer_idx]
        val_losses = results['val_losses'][layer_idx]
        
        ax1.plot(epochs, losses, 'b-', label='Train Loss', linewidth=2, marker='o', markersize=4)
        ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2, marker='s', markersize=4)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        
        # Secondary y-axis: Correlations and Similarities
        ax2 = ax1.twinx()
        
        # Original AGOP/NFM correlation
        correlations = results['layer_correlations'][layer_idx]
        ax2.plot(epochs, correlations, 'g-', label='AGOP/NFM Corr', linewidth=2, marker='^', markersize=4)
        
        # New matrix similarities
        if epoch_similarities:
            valid_epochs = []
            sim_GH = []
            sim_GM = []
            sim_HM = []
            
            for epoch in epochs:
                if epoch in epoch_similarities and layer_idx in epoch_similarities[epoch]:
                    valid_epochs.append(epoch)
                    sim_GH.append(epoch_similarities[epoch][layer_idx]['G_H'])
                    sim_GM.append(epoch_similarities[epoch][layer_idx]['G_M'])
                    sim_HM.append(epoch_similarities[epoch][layer_idx]['H_M'])
            
            if valid_epochs:
                ax2.plot(valid_epochs, sim_GH, 'm-', label='G vs H', linewidth=2, marker='d', markersize=4)
                ax2.plot(valid_epochs, sim_GM, 'c-', label='G vs M', linewidth=2, marker='v', markersize=4)
                ax2.plot(valid_epochs, sim_HM, 'orange', label='H vs M', linewidth=2, marker='*', markersize=4)
        
        ax2.set_ylabel('Cosine Similarity', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.set_ylim(0, 1)
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
        
        ax1.set_title(f'Layer {layer_idx}', fontweight='bold')
        ax1.grid(True, alpha=0.3)
    
    # Hide the last subplot
    axes[1, 2].axis('off')
    
    # Add text explanation in the last subplot
    axes[1, 2].text(0.1, 0.7, 
                   'Matrix Definitions:\n'
                   '• H_i := E[h_i h_i^T] (Representation)\n'
                   '• G_i := E[g_i g_i^T] (Gradient)\n'
                   '• M_i := W_i^T W_i (Weight/NFM)\n\n'
                   'Similarities:\n'
                   '• G vs H: Gradient-Representation\n'
                   '• G vs M: Gradient-Weight\n'
                   '• H vs M: Representation-Weight',
                   fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{model_name}_enhanced_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Enhanced analysis plot saved: {output_path}")
    
    # Save similarities data
    similarities_path = os.path.join(output_dir, f'{model_name}_similarities.json')
    with open(similarities_path, 'w') as f:
        json.dump(epoch_similarities, f, indent=2)
    print(f"Similarities data saved: {similarities_path}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Enhanced Analysis Visualization')
    parser.add_argument('--models', nargs='+', required=True,
                       help='Model directory names (e.g., adam_20250803_173338 muon_20250804_041908 sgd_20250803_164335)')
    parser.add_argument('--experiments_dir', type=str, default='experiments',
                       help='Experiments directory (default: experiments)')
    parser.add_argument('--output_dir', type=str, default='enhanced_analysis',
                       help='Output directory for enhanced plots (default: enhanced_analysis)')
    
    args = parser.parse_args()
    
    print("=== Enhanced Analysis Visualization ===")
    print(f"Processing models: {args.models}")
    
    # Set random seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    for model_name in args.models:
        model_dir = os.path.join(args.experiments_dir, model_name)
        
        if not os.path.exists(model_dir):
            print(f"Model directory not found: {model_dir}")
            continue
        
        print(f"\n=== Processing {model_name} ===")
        create_enhanced_analysis_plot(model_dir, args.output_dir)
    
    print(f"\n✅ Enhanced analysis completed! Results saved in {args.output_dir}/")

if __name__ == "__main__":
    main()
