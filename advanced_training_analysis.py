#!/usr/bin/env python3
"""
Advanced Training and AGOP Analysis Script

This script trains models with different optimizers for 500 epochs,
saves models every 20 epochs, and tracks:
1. Train/Val Loss
2. AGOP (uncentered) vs NFM correlation for each layer

Generates comprehensive visualization plots.
"""

import os
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

def get_dataset_info(dataset_name):
    """Get dataset-specific information including loss axis ranges for consistent plotting"""
    dataset_configs = {
        'svhn': {
            'num_classes': 10,
            'loader_func': dataset.get_svhn,
            'input_size': 32,
            'channels': 3,
            'loss_ylim': (0, 0.1)  # Consistent loss range for SVHN across all optimizers
        },
        'cifar': {
            'num_classes': 10,
            'loader_func': dataset.get_cifar,
            'input_size': 32,
            'channels': 3,
            'loss_ylim': (0, 0.15)  # Consistent loss range for CIFAR across all optimizers
        },
        'cifar_mnist': {
            'num_classes': 10,
            'loader_func': dataset.get_cifar_mnist,
            'input_size': 32,
            'channels': 3,
            'loss_ylim': (0, 0.12)  # Consistent loss range for CIFAR-MNIST across all optimizers
        },
        'celeba': {
            'num_classes': 2,
            'loader_func': lambda: dataset.get_celeba(feature_idx=20),
            'input_size': 96,
            'channels': 3,
            'loss_ylim': (0, 0.25)  # Consistent loss range for CelebA across all optimizers
        },
        'stl_star': {
            'num_classes': 2,
            'loader_func': dataset.get_stl_star,
            'input_size': 96,
            'channels': 3,
            'loss_ylim': (0, 0.25)  # Consistent loss range for STL-Star across all optimizers
        }
    }
    
    if dataset_name not in dataset_configs:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: {list(dataset_configs.keys())}")
    
    return dataset_configs[dataset_name]

def setup_experiment_dir(optimizer_name, dataset_name='svhn'):
    """Create experiment directory structure"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiments/{dataset_name}_{optimizer_name}_{timestamp}"
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f"{exp_dir}/models", exist_ok=True)
    os.makedirs(f"{exp_dir}/plots", exist_ok=True)
    
    return exp_dir

def compute_agop_nfm_correlation(model_path, layer_indices, max_samples=None, init_model_path=None, dataset_name='svhn'):
    """
    AGOP vs NFM correlation computation
    
    Args:
        model_path: Path to the saved model
        layer_indices: List of layer indices to analyze
        max_samples: limit samples for AGOP computation (memory management)
        init_model_path: Path to initial model (for remove_init operation)
        dataset_name: dataset to use for AGOP computation
    
    Returns:
        dict: {layer_idx: correlation_value}
    """
    correlations = {}
    
    # Get dataset configuration
    dataset_info = get_dataset_info(dataset_name)
    
    # Set random seed for consistency
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    
    try:
        # Read model config from path
        width, depth, act_name = read_configs(model_path)
        
        # Get dataset-specific parameters
        NUM_CLASSES = dataset_info['num_classes']
        SIZE = dataset_info['input_size']
        c = dataset_info['channels']
        dim = c * SIZE * SIZE
        
        # Load correct dataset based on dataset_name
        if dataset_name == 'svhn':
            trainloader, valloader, testloader = dataset.get_svhn()
        elif dataset_name == 'cifar':
            trainloader, valloader, testloader = dataset.get_cifar()
        elif dataset_name == 'cifar_mnist':
            trainloader, valloader, testloader = dataset.get_cifar_mnist()
        elif dataset_name == 'celeba':
            trainloader, valloader, testloader = dataset.get_celeba()
        elif dataset_name == 'stl_star':
            trainloader, valloader, testloader = dataset.get_stl_star()
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        # Load initial model if provided for remove_init operation
        init_params = None
        if init_model_path is not None:
            init_net = neural_model.Net(dim, width=width, depth=depth,
                                      num_classes=NUM_CLASSES, act_name=act_name)
            init_d = torch.load(init_model_path, map_location='cpu')
            init_net.load_state_dict(clean_compiled_state_dict(init_d['state_dict']))
            init_params = [p.data.cpu().numpy() for p in init_net.parameters()]
        
        for layer_idx in layer_indices:
            print(f"Computing AGOP/NFM correlation for layer {layer_idx}...")
            
            # Load trained model and get NFM
            net = neural_model.Net(dim, width=width, depth=depth,
                                 num_classes=NUM_CLASSES, act_name=act_name)
            d = torch.load(model_path)
            net.load_state_dict(clean_compiled_state_dict(d['state_dict']))
            
            # Extract NFM (Neural Feature Matrix) for this layer
            for idx, p in enumerate(net.parameters()):
                if idx == layer_idx:
                    M = p.data.cpu().numpy()
                    
                    # Apply remove_init operation (same as verify_deep_NFA.py)
                    if init_params is not None:
                        M0 = init_params[idx]
                        M = M - M0  # Remove initial parameters
                        print(f"    Applied remove_init: M shape after init removal: {M.shape}")
                    
                    break
            
            # Compute NFM
            M = M.T @ M * (1/len(M))
            
            # Build subnetwork for AGOP computation
            subnet = build_subnetwork(net, M.shape[0], width, depth, NUM_CLASSES, 
                                    layer_idx=layer_idx, random_net=False, act_name=act_name)
            
            # Get layer output
            out = get_layer_output(net, trainloader, layer_idx=layer_idx, max_samples=max_samples)
            
            # Compute AGOP (uncentered)
            G = egop(subnet, out, centering=False)
            
            # Compute correlation
            correlation = correlate(torch.from_numpy(M), G)
            correlations[layer_idx] = correlation.item()
            
            print(f"Layer {layer_idx} AGOP/NFM correlation: {correlation.item():.6f}")
    
    except Exception as e:
        print(f"Error computing AGOP/NFM correlation: {e}")
        for layer_idx in layer_indices:
            correlations[layer_idx] = 0.0
    
    return correlations

def train_with_analysis(optimizer_name, lr, num_epochs=500, val_interval=20, max_samples=None, weight_decay=0, dataset_name='svhn'):
    """
    Train model with comprehensive analysis
    
    Args:
        optimizer_name: 'sgd', 'adam', or 'muon'
        lr: learning rate
        num_epochs: total epochs
        val_interval: interval for saving and analysis
        max_samples: limit samples for AGOP computation (memory management)
        dataset_name: dataset to use for training
    """
    
    print(f"\n=== Starting Training with {optimizer_name.upper()} on {dataset_name.upper()} ===")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {num_epochs}")
    print(f"Analysis interval: {val_interval}")
    
    # Setup experiment directory
    exp_dir = setup_experiment_dir(optimizer_name, dataset_name)
    print(f"Experiment directory: {exp_dir}")
    
    # Set random seed
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    
    # Get dataset info and load data
    dataset_info = get_dataset_info(dataset_name)
    if dataset_name == 'svhn':
        trainloader, valloader, testloader = dataset.get_svhn()
    elif dataset_name == 'cifar':
        trainloader, valloader, testloader = dataset.get_cifar()
    elif dataset_name == 'cifar_mnist':
        trainloader, valloader, testloader = dataset.get_cifar_mnist()
    elif dataset_name == 'celeba':
        trainloader, valloader, testloader = dataset.get_celeba()
    elif dataset_name == 'stl_star':
        trainloader, valloader, testloader = dataset.get_stl_star()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Get input dimension
    for batch in trainloader:
        inputs, _ = batch
        _, dim = inputs.shape
        break
    
    # Model configuration
    configs = {
        'num_epochs': num_epochs,
        'learning_rate': lr,
        'weight_decay': weight_decay,
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
                          num_classes=dataset_info['num_classes'],
                          act_name=configs['act'])
    
    # Get device and setup
    device = trainer.get_best_device()
    net = net.to(device)
    
    # Setup optimizer
    optimizer = trainer.select_optimizer(optimizer_name, lr, net, configs['weight_decay'])
    
    # Enable torch.compile for muon
    if optimizer_name == 'muon':
        net = torch.compile(net, mode='reduce-overhead')
    
    criterion = torch.nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # Save initial model before training starts (for remove_init operation)
    net.cpu()
    d = {'state_dict': trainer.get_clean_state_dict(net)}
    init_model_path = f'{exp_dir}/models/init_model.pth'
    torch.save(d, init_model_path)
    net.to(device)
    print(f"Initial model saved: {init_model_path}")
    
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
        # Training step - calculate actual loss for epoch 0 (initial state)
        if epoch > 0:  
            train_loss = trainer.train_step(net, optimizer, trainloader, criterion, device, scaler)
        else:
            # Calculate initial training loss without training step
            net.eval()
            with torch.no_grad():
                total_loss = 0.0
                num_batches = 0
                for inputs, targets in trainloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)
                    total_loss += loss.item()
                    num_batches += 1
                train_loss = total_loss / num_batches if num_batches > 0 else 0.0
            net.train()  # Set back to training mode
        
        # Analysis every val_interval epochs
        if epoch % val_interval == 0:
            print(f"\n--- Epoch {epoch} Analysis ---")
            
            # Validation loss
            val_loss = trainer.val_step(net, valloader, criterion, device)
            
            # Save model
            net.cpu()
            d = {'state_dict': trainer.get_clean_state_dict(net)}
            model_path = f'{exp_dir}/models/model_epoch_{epoch}.pth'
            torch.save(d, model_path)
            net.to(device)
            
            # Compute AGOP/NFM correlations
            print("Computing AGOP/NFM correlations...")
            # Create a path string that read_configs can parse - use actual dataset name
            config_path = f"{dataset_name}:width:{configs['width']}:depth:{configs['depth']}:act:{configs['act']}:nn"
            temp_model_path = model_path.replace('.pth', f':{config_path}.pth')
            
            # Temporarily rename file to include config info
            os.rename(model_path, temp_model_path)
            
            try:
                correlations = compute_agop_nfm_correlation(temp_model_path, layer_indices, 
                                                           max_samples, init_model_path=init_model_path,
                                                           dataset_name=dataset_name)
            finally:
                # Rename back
                os.rename(temp_model_path, model_path)
            
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
    generate_plots(results, exp_dir, optimizer_name, dataset_name)
    
    print(f"\nTraining completed! Results saved in: {exp_dir}")
    return exp_dir, results

def generate_plots(results, exp_dir, optimizer_name, dataset_name='svhn'):
    """Generate comprehensive visualization plots"""
    
    print("Generating plots...")
    
    # Get dataset configuration for consistent plotting
    dataset_info = get_dataset_info(dataset_name)
    loss_ylim = dataset_info['loss_ylim']
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create subplots: 5 layers in a 2x3 grid (with one empty subplot)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{optimizer_name.upper()} Training Analysis: Loss and AGOP/NFM Correlation', 
                 fontsize=16, fontweight='bold')
    
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
        line1 = ax.plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss', alpha=0.8)
        line2 = ax.plot(epochs, val_losses, 'r-', linewidth=2, label='Val Loss', alpha=0.8)
        
        # Plot correlation
        line3 = ax2.plot(epochs, correlations, 'g-', linewidth=2, label='AGOP/NFM Corr', alpha=0.8)
        
        # Formatting
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss', color='black')
        ax2.set_ylabel('AGOP/NFM Correlation', color='green')
        ax.set_title(f'Layer {layer_idx}', fontweight='bold')
        
        # Color the y-axis labels
        ax.tick_params(axis='y', labelcolor='black')
        ax2.tick_params(axis='y', labelcolor='green')
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        # Legend
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right')
        
        # Set loss y-limits to be consistent across optimizers for the same dataset
        ax.set_ylim(loss_ylim[0], loss_ylim[1])
        
        # Set correlation y-axis to consistent 0-1 range for all layers
        ax2.set_ylim(0, 1)
        ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])  # Clear tick marks
    
    # Hide the last (6th) subplot
    axes_flat[5].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plot_path = f'{exp_dir}/plots/training_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(f'{exp_dir}/plots/training_analysis.pdf', bbox_inches='tight')
    plt.close()
    
    # Also create individual plots for each layer
    for layer_idx in range(5):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax2 = ax.twinx()
        
        correlations = results['layer_correlations'][layer_idx]
        
        # Plot losses
        ax.plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss', alpha=0.8)
        ax.plot(epochs, val_losses, 'r-', linewidth=2, label='Val Loss', alpha=0.8)
        
        # Plot correlation
        ax2.plot(epochs, correlations, 'g-', linewidth=2, label='AGOP/NFM Correlation', alpha=0.8)
        
        # Formatting
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss', color='black')
        ax2.set_ylabel('AGOP/NFM Correlation', color='green')
        ax.set_title(f'{optimizer_name.upper()} - Layer {layer_idx} Analysis', fontweight='bold')
        
        ax.tick_params(axis='y', labelcolor='black')
        ax2.tick_params(axis='y', labelcolor='green')
        ax.grid(True, alpha=0.3)
        
        # Legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f'{exp_dir}/plots/layer_{layer_idx}_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Plots saved in: {exp_dir}/plots/")

def main():
    parser = argparse.ArgumentParser(description='Advanced Training and AGOP Analysis')
    parser.add_argument('--optimizer', type=str, required=True, 
                        choices=['sgd', 'adam', 'muon'],
                        help='Optimizer to use')
    parser.add_argument('--dataset', type=str, default='svhn',
                        choices=['svhn', 'cifar', 'cifar_mnist', 'celeba', 'stl_star'],
                        help='Dataset to use (default: svhn)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: auto-select based on optimizer)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs (default: 500)')
    parser.add_argument('--val_interval', type=int, default=20,
                        help='Validation/analysis interval (default: 20)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max samples for AGOP computation (memory management)')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay (L2 regularization) factor (default: 0)')
    
    args = parser.parse_args()
    
    # Auto-select learning rate if not provided
    if args.lr is None:
        lr_defaults = {'sgd': 0.1, 'adam': 0.001, 'muon': 0.01}
        args.lr = lr_defaults[args.optimizer]
        print(f"Using default learning rate for {args.optimizer}: {args.lr}")
    
    # Run training and analysis
    exp_dir, results = train_with_analysis(
        optimizer_name=args.optimizer,
        lr=args.lr,
        num_epochs=args.epochs,
        val_interval=args.val_interval,
        max_samples=args.max_samples,
        weight_decay=args.weight_decay,
        dataset_name=args.dataset
    )
    
    print(f"\n=== Experiment Complete ===")
    print(f"Results directory: {exp_dir}")
    print(f"Generated plots: {exp_dir}/plots/")
    print(f"Model checkpoints: {exp_dir}/models/")
    print(f"Raw data: {exp_dir}/results.json")

if __name__ == "__main__":
    main()
