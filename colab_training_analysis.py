#!/usr/bin/env python3
"""
Colab-Optimized Training and AGOP Analysis Script

Memory-optimized version for Google Colab with reduced sample sizes
and efficient memory management.
"""

import os
import sys
import subprocess

# Install required packages if not available
def install_packages():
    packages = ['matplotlib', 'seaborn']
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install packages first
install_packages()

import numpy as np
import torch
import random
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse

# Import our modules
import dataset
import trainer
import neural_model
from verify_deep_NFA import (
    clean_compiled_state_dict, get_layer_output, build_subnetwork, 
    egop, correlate, read_configs, SEED
)

def setup_experiment_dir(optimizer_name):
    """Create experiment directory structure"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiments/{optimizer_name}_{timestamp}"
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f"{exp_dir}/models", exist_ok=True)
    os.makedirs(f"{exp_dir}/plots", exist_ok=True)
    
    return exp_dir

def compute_agop_nfm_correlation_optimized(model_path, layer_indices, max_samples=5000):
    """
    Memory-optimized AGOP vs NFM correlation computation
    
    Returns:
        dict: {layer_idx: correlation_value}
    """
    correlations = {}
    
    # Set random seed for consistency
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    
    try:
        # Extract config from model path filename
        # Expected format: model_epoch_X:svhn:width:1024:depth:5:act:relu:nn.pth
        filename = os.path.basename(model_path)
        if ':' in filename:
            config_part = filename.split(':')[1:]  # Skip the epoch part
            config_str = ':'.join(config_part).replace('.pth', '')
        else:
            # Fallback: use default config
            config_str = "svhn:width:1024:depth:5:act:relu:nn"
        
        width, depth, act_name = read_configs(config_str)
        
        # Dataset parameters
        NUM_CLASSES = 10
        SIZE = 32
        c = 3
        dim = c * SIZE * SIZE
        
        # Load dataset
        trainloader, valloader, testloader = dataset.get_svhn()
        
        for layer_idx in layer_indices:
            print(f"  Computing layer {layer_idx} correlation...")
            
            # Load trained model and get NFM
            net = neural_model.Net(dim, width=width, depth=depth,
                                 num_classes=NUM_CLASSES, act_name=act_name)
            d = torch.load(model_path, map_location='cpu')
            net.load_state_dict(clean_compiled_state_dict(d['state_dict']))
            
            # Extract NFM (Neural Feature Matrix) for this layer
            for idx, p in enumerate(net.parameters()):
                if idx == layer_idx:
                    M = p.data.cpu().numpy()
                    break
            
            # Compute NFM
            M = M.T @ M * (1/len(M))
            
            # Build subnetwork for AGOP computation
            subnet = build_subnetwork(net, M.shape[0], width, depth, NUM_CLASSES, 
                                    layer_idx=layer_idx, random_net=False, act_name=act_name)
            
            # Get layer output with memory limit
            out = get_layer_output(net, trainloader, layer_idx=layer_idx, max_samples=max_samples)
            
            # Compute AGOP (uncentered) with smaller batch size
            print(f"    Computing EGOP for {out.shape[0]} samples...")
            G = egop(subnet, out, centering=False)
            
            # Compute correlation
            correlation = correlate(torch.from_numpy(M), G)
            correlations[layer_idx] = correlation.item()
            
            # Clean up GPU memory
            del net, subnet, out, G
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            print(f"    Layer {layer_idx} correlation: {correlation.item():.6f}")
    
    except Exception as e:
        print(f"Error computing AGOP/NFM correlation: {e}")
        import traceback
        traceback.print_exc()
        for layer_idx in layer_indices:
            correlations[layer_idx] = 0.0
    
    return correlations

def train_with_analysis_colab(optimizer_name, lr, num_epochs=500, val_interval=20):
    """
    Colab-optimized training with comprehensive analysis
    """
    
    print(f"\n=== Starting Training with {optimizer_name.upper()} ===")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {num_epochs}")
    print(f"Analysis interval: {val_interval}")
    print(f"Memory-optimized for Colab environment")
    
    # Setup experiment directory
    exp_dir = setup_experiment_dir(optimizer_name)
    print(f"Experiment directory: {exp_dir}")
    
    # Set random seed
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    
    # Load data
    trainloader, valloader, testloader = dataset.get_svhn()
    
    # Get input dimension
    for batch in trainloader:
        inputs, _ = batch
        _, dim = inputs.shape
        break
    
    # Model configuration
    configs = {
        'num_epochs': num_epochs,
        'learning_rate': lr,
        'weight_decay': 0,
        'init': 'default',
        'optimizer': optimizer_name,
        'freeze': False,
        'width': 1024,
        'depth': 5,
        'act': 'relu',
        'val_interval': val_interval
    }
    
    # Create model
    net = neural_model.Net(dim, width=configs['width'],
                          depth=configs['depth'],
                          num_classes=10,
                          act_name=configs['act'])
    
    # Get device and setup
    device = trainer.get_best_device()
    net = net.to(device)
    
    # Setup optimizer
    optimizer = trainer.select_optimizer(optimizer_name, lr, net, configs['weight_decay'])
    
    # Enable torch.compile for muon (if supported)
    if optimizer_name == 'muon':
        try:
            net = torch.compile(net, mode='reduce-overhead')
            print("Torch.compile enabled for Muon")
        except:
            print("Torch.compile not available, using regular mode")
    
    criterion = torch.nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # Save initial model
    net.cpu()
    d = {'state_dict': trainer.get_clean_state_dict(net)}
    torch.save(d, f'{exp_dir}/models/init_model.pth')
    net.to(device)
    
    # Training tracking
    results = {
        'epochs': [],
        'train_losses': [],
        'val_losses': [],
        'layer_correlations': {i: [] for i in range(5)}  # layers 0-4
    }
    
    layer_indices = [0, 1, 2, 3, 4]
    
    print(f"\nStarting training loop...")
    
    for epoch in range(num_epochs + 1):
        # Training step
        if epoch > 0:  # Skip training on epoch 0 (initial state)
            train_loss = trainer.train_step(net, optimizer, trainloader, criterion, device, scaler)
        else:
            train_loss = 0.0
        
        # Analysis every val_interval epochs
        if epoch % val_interval == 0:
            print(f"\n--- Epoch {epoch} Analysis ---")
            
            # Validation loss
            val_loss = trainer.val_step(net, valloader, criterion, device)
            
            # Save model with config info in filename for easier parsing
            net.cpu()
            d = {'state_dict': trainer.get_clean_state_dict(net)}
            model_filename = f'model_epoch_{epoch}:svhn:width:{configs["width"]}:depth:{configs["depth"]}:act:{configs["act"]}:nn.pth'
            model_path = f'{exp_dir}/models/{model_filename}'
            torch.save(d, model_path)
            net.to(device)
            
            # Compute AGOP/NFM correlations with memory optimization
            print("Computing AGOP/NFM correlations...")
            correlations = compute_agop_nfm_correlation_optimized(model_path, layer_indices)
            
            # Store results
            results['epochs'].append(epoch)
            results['train_losses'].append(train_loss)
            results['val_losses'].append(val_loss)
            
            for layer_idx in layer_indices:
                results['layer_correlations'][layer_idx].append(correlations.get(layer_idx, 0.0))
            
            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
            for layer_idx in layer_indices:
                corr = correlations.get(layer_idx, 0.0)
                print(f"  Layer {layer_idx} AGOP/NFM correlation: {corr:.6f}")
        
        else:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}")
    
    # Save results
    with open(f'{exp_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate plots
    generate_plots_colab(results, exp_dir, optimizer_name)
    
    print(f"\nTraining completed! Results saved in: {exp_dir}")
    return exp_dir, results

def generate_plots_colab(results, exp_dir, optimizer_name):
    """Generate comprehensive visualization plots optimized for Colab"""
    
    print("Generating plots...")
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create subplots: 5 layers in a 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'{optimizer_name.upper()} Training Analysis: Loss and AGOP/NFM Correlation', 
                 fontsize=18, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    epochs = results['epochs']
    train_losses = results['train_losses']
    val_losses = results['val_losses']
    
    for layer_idx in range(5):
        ax = axes_flat[layer_idx]
        correlations = results['layer_correlations'][layer_idx]
        
        # Create twin axis for correlation
        ax2 = ax.twinx()
        
        # Plot losses
        line1 = ax.plot(epochs, train_losses, 'b-', linewidth=2.5, label='Train Loss', alpha=0.8)
        line2 = ax.plot(epochs, val_losses, 'r-', linewidth=2.5, label='Val Loss', alpha=0.8)
        
        # Plot correlation
        line3 = ax2.plot(epochs, correlations, 'g-', linewidth=2.5, label='AGOP/NFM Corr', alpha=0.8, marker='o', markersize=4)
        
        # Formatting
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', color='black', fontsize=12)
        ax2.set_ylabel('AGOP/NFM Correlation', color='green', fontsize=12)
        ax.set_title(f'Layer {layer_idx}', fontweight='bold', fontsize=14)
        
        # Color the y-axis labels
        ax.tick_params(axis='y', labelcolor='black')
        ax2.tick_params(axis='y', labelcolor='green')
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        # Legend
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right', fontsize=10)
        
        # Set reasonable y-limits
        if len(train_losses) > 0:
            loss_min = min(min(train_losses), min(val_losses))
            loss_max = max(max(train_losses), max(val_losses))
            loss_range = loss_max - loss_min
            if loss_range > 0:
                ax.set_ylim(loss_min - 0.1 * loss_range, loss_max + 0.1 * loss_range)
        
        if len(correlations) > 0 and max(correlations) > min(correlations):
            corr_min = min(correlations)
            corr_max = max(correlations)
            corr_range = corr_max - corr_min
            if corr_range > 0:
                ax2.set_ylim(corr_min - 0.1 * corr_range, corr_max + 0.1 * corr_range)
    
    # Hide the last (6th) subplot
    axes_flat[5].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plot_path = f'{exp_dir}/plots/training_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(f'{exp_dir}/plots/training_analysis.pdf', bbox_inches='tight')
    
    # Display in Colab
    plt.show()
    plt.close()
    
    print(f"Plots saved in: {exp_dir}/plots/")

def run_all_optimizers():
    """Run training for all three optimizers"""
    
    optimizers = [
        ('sgd', 0.1),
        ('adam', 0.001),
        ('muon', 0.01)
    ]
    
    results_summary = {}
    
    for optimizer_name, lr in optimizers:
        print(f"\n{'='*60}")
        print(f"Running {optimizer_name.upper()} with lr={lr}")
        print(f"{'='*60}")
        
        try:
            exp_dir, results = train_with_analysis_colab(
                optimizer_name=optimizer_name,
                lr=lr,
                num_epochs=500,
                val_interval=20
            )
            
            results_summary[optimizer_name] = {
                'exp_dir': exp_dir,
                'final_train_loss': results['train_losses'][-1] if results['train_losses'] else 0,
                'final_val_loss': results['val_losses'][-1] if results['val_losses'] else 0,
                'final_correlations': {k: v[-1] if v else 0 for k, v in results['layer_correlations'].items()}
            }
            
        except Exception as e:
            print(f"Error training {optimizer_name}: {e}")
            import traceback
            traceback.print_exc()
            results_summary[optimizer_name] = {'error': str(e)}
    
    # Print summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    for optimizer_name in ['sgd', 'adam', 'muon']:
        if optimizer_name in results_summary:
            result = results_summary[optimizer_name]
            if 'error' in result:
                print(f"{optimizer_name.upper()}: FAILED - {result['error']}")
            else:
                print(f"\n{optimizer_name.upper()}:")
                print(f"  Final Train Loss: {result['final_train_loss']:.6f}")
                print(f"  Final Val Loss: {result['final_val_loss']:.6f}")
                print(f"  Final Correlations:")
                for layer, corr in result['final_correlations'].items():
                    print(f"    Layer {layer}: {corr:.6f}")
                print(f"  Results dir: {result['exp_dir']}")
    
    return results_summary

if __name__ == "__main__":
    print("=== Colab-Optimized Training and AGOP Analysis ===")
    print("This script will train models with SGD, Adam, and Muon optimizers")
    print("Each training: 500 epochs, analysis every 20 epochs")
    print("Memory-optimized for Colab environment\n")
    
    # Run all optimizers
    results_summary = run_all_optimizers()
    
    print("\n=== ALL TRAINING COMPLETE ===")
    print("Check the experiments/ directory for detailed results and plots.")
