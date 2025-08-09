#!/usr/bin/env python3
"""
Custom Training and AGOP Analysis Script - 100 Epochs Version

Modified version with:
- 100 epochs total
- Save models for epochs 1-20 (every epoch)
- Save models for epochs 40, 60, 80, 100 (milestone epochs)
- Generate same visualization plots
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

def get_dataset_info(dataset_name):
    """Get dataset-specific information"""
    dataset_configs = {
        'svhn': {
            'num_classes': 10,
            'loader_func': lambda: dataset.get_svhn(),
            'input_size': 32,
            'channels': 3
        },
        'cifar': {
            'num_classes': 10,
            'loader_func': lambda: dataset.get_cifar(),
            'input_size': 32,
            'channels': 3
        },
        'cifar_mnist': {
            'num_classes': 10,
            'loader_func': lambda: dataset.get_cifar_mnist(),
            'input_size': 32,
            'channels': 3
        },
        'celeba': {
            'num_classes': 2,
            'loader_func': lambda: dataset.get_celeba(feature_idx=20),  # Use feature 20 (Male)
            'input_size': 96,
            'channels': 3
        },
        'stl_star': {
            'num_classes': 2,
            'loader_func': lambda: dataset.get_stl_star(),
            'input_size': 96,
            'channels': 3
        }
    }
    
    if dataset_name not in dataset_configs:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: {list(dataset_configs.keys())}")
    
    return dataset_configs[dataset_name]

def setup_experiment_dir(optimizer_name, dataset_name):
    """Create experiment directory structure"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiments/{dataset_name}_{optimizer_name}_100epochs_{timestamp}"
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f"{exp_dir}/models", exist_ok=True)
    os.makedirs(f"{exp_dir}/plots", exist_ok=True)
    
    return exp_dir

def compute_agop_nfm_correlation_optimized(model_path, layer_indices, max_samples=None, init_model_path=None, dataset_info=None):
    """
    AGOP vs NFM correlation computation using ALL training samples
    
    Args:
        model_path: Path to the saved model
        layer_indices: List of layer indices to analyze
        max_samples: If None, uses ALL training samples (recommended for accurate AGOP)
                    If specified, limits samples for memory management
        init_model_path: Path to initial model (for remove_init operation)
    
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
        # Expected format: model_epoch_X:{dataset}:width:1024:depth:5:act:relu:nn.pth
        filename = os.path.basename(model_path)
        if ':' in filename:
            config_part = filename.split(':')[1:]  # Skip the epoch part
            config_str = ':'.join(config_part).replace('.pth', '')
        else:
            # Fallback: use default config
            config_str = f"{dataset_info['name'] if dataset_info else 'svhn'}:width:1024:depth:5:act:relu:nn"
        
        width, depth, act_name = read_configs(config_str)
        
        # Dataset parameters (use provided dataset_info or defaults)
        if dataset_info:
            NUM_CLASSES = dataset_info['num_classes']
            SIZE = dataset_info['input_size']
            c = dataset_info['channels']
            trainloader, valloader, testloader = dataset_info['loader_func']()
        else:
            # Fallback to SVHN
            NUM_CLASSES = 10
            SIZE = 32
            c = 3
            trainloader, valloader, testloader = dataset.get_svhn()
        
        dim = c * SIZE * SIZE
        
        # Load initial model if provided for remove_init operation
        init_params = None
        if init_model_path is not None:
            init_net = neural_model.Net(dim, width=width, depth=depth,
                                      num_classes=NUM_CLASSES, act_name=act_name)
            init_d = torch.load(init_model_path, map_location='cpu')
            init_net.load_state_dict(clean_compiled_state_dict(init_d['state_dict']))
            init_params = [p.data.cpu().numpy() for p in init_net.parameters()]
        
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
            
            # Get layer output using ALL training samples (same as during training)
            out = get_layer_output(net, trainloader, layer_idx=layer_idx, max_samples=max_samples)
            
            # Compute AGOP (uncentered) 
            total_samples = out.shape[0]
            print(f"    Computing EGOP for {total_samples} samples (ALL training data)...")
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

def get_analysis_epochs(num_epochs=100):
    """
    Define which epochs to analyze:
    - Epochs 0-20: every epoch
    - Epochs 40, 60, 80, 100: milestone epochs
    """
    analysis_epochs = list(range(0, 21))  # 0, 1, 2, ..., 20
    milestone_epochs = [40, 60, 80, 100]
    
    # Only add milestone epochs if they're within the total epochs
    for epoch in milestone_epochs:
        if epoch <= num_epochs:
            analysis_epochs.append(epoch)
    
    return sorted(set(analysis_epochs))

def train_with_analysis_100epochs(optimizer_name, lr, num_epochs=100, dataset_name='svhn'):
    """
    Training with custom analysis schedule for 100 epochs
    
    Analysis schedule:
    - Epochs 0-20: analyze every epoch
    - Epochs 40, 60, 80, 100: milestone analysis
    """
    
    print(f"\n=== Starting 100-Epoch Training with {optimizer_name.upper()} on {dataset_name.upper()} ===")
    print(f"Learning rate: {lr}")
    print(f"Total epochs: {num_epochs}")
    print(f"Dataset: {dataset_name}")
    print(f"Analysis schedule: epochs 0-20 (every epoch), then 40, 60, 80, 100")
    print(f"AGOP Analysis: Using ALL training samples for accurate correlation")
    
    # Get dataset information
    dataset_info = get_dataset_info(dataset_name)
    dataset_info['name'] = dataset_name
    
    # Setup experiment directory
    exp_dir = setup_experiment_dir(optimizer_name, dataset_name)
    print(f"Experiment directory: {exp_dir}")
    
    # Set random seed
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    
    # Load data
    trainloader, valloader, testloader = dataset_info['loader_func']()
    
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
        'dataset': dataset_name
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
    
    # Enable torch.compile for muon (if supported)
    if optimizer_name == 'muon':
        try:
            net = torch.compile(net, mode='reduce-overhead')
            print("Torch.compile enabled for Muon")
        except:
            print("Torch.compile not available, using regular mode")
    
    criterion = torch.nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # Save initial model before training starts (for remove_init operation)
    net.cpu()
    d = {'state_dict': trainer.get_clean_state_dict(net)}
    init_model_path = f'{exp_dir}/models/init_model.pth'
    torch.save(d, init_model_path)
    net.to(device)
    print(f"Initial model saved: {init_model_path}")
    
    # Get analysis epochs
    analysis_epochs = get_analysis_epochs(num_epochs)
    print(f"Will analyze at epochs: {analysis_epochs}")
    
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
        
        # Analysis for specific epochs
        if epoch in analysis_epochs:
            print(f"\n--- Epoch {epoch} Analysis ---")
            
            # Validation loss
            val_loss = trainer.val_step(net, valloader, criterion, device)
            
            # Save model with config info in filename for easier parsing
            net.cpu()
            d = {'state_dict': trainer.get_clean_state_dict(net)}
            model_filename = f'model_epoch_{epoch}:{dataset_name}:width:{configs["width"]}:depth:{configs["depth"]}:act:{configs["act"]}:nn.pth'
            model_path = f'{exp_dir}/models/{model_filename}'
            torch.save(d, model_path)
            net.to(device)
            
            # Compute AGOP/NFM correlations using ALL training samples (with remove_init)
            print("Computing AGOP/NFM correlations using ALL training samples...")
            correlations = compute_agop_nfm_correlation_optimized(model_path, layer_indices, 
                                                                max_samples=None, 
                                                                init_model_path=init_model_path,
                                                                dataset_info=dataset_info)
            
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
    generate_plots_100epochs(results, exp_dir, optimizer_name, dataset_name)
    
    print(f"\nTraining completed! Results saved in: {exp_dir}")
    return exp_dir, results

def generate_plots_100epochs(results, exp_dir, optimizer_name, dataset_name):
    """Generate comprehensive visualization plots for 100-epoch training"""
    
    print("Generating plots...")
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create subplots: 5 layers in a 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'{optimizer_name.upper()} Training Analysis on {dataset_name.upper()} (100 Epochs): Loss and AGOP/NFM Correlation', 
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
        
        # Set correlation y-axis to consistent 0-1 range for all layers
        ax2.set_ylim(0, 1)
        ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])  # Clear tick marks
        
        # Add epoch markers for analysis points
        if len(epochs) > 0:
            # Mark the transition from detailed to milestone analysis
            ax.axvline(x=20, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            ax.text(20, ax.get_ylim()[1]*0.9, 'Detailed→Milestone', rotation=90, 
                   fontsize=8, alpha=0.7, ha='right')
    
    # Hide the last (6th) subplot
    axes_flat[5].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plot_path = f'{exp_dir}/plots/training_analysis_100epochs.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(f'{exp_dir}/plots/training_analysis_100epochs.pdf', bbox_inches='tight')
    
    # Display in Colab (if running in Colab)
    try:
        from IPython.display import display
        plt.show()
    except:
        pass
    
    plt.close()
    
    print(f"Plots saved in: {exp_dir}/plots/")

def run_all_optimizers_100epochs(dataset_name='svhn'):
    """Run 100-epoch training for all three optimizers"""
    
    optimizers = [
        ('sgd', 0.1),
        ('adam', 0.001),
        ('muon', 0.01)
    ]
    
    results_summary = {}
    
    for optimizer_name, lr in optimizers:
        print(f"\n{'='*70}")
        print(f"Running {optimizer_name.upper()} with lr={lr} on {dataset_name.upper()} (100 epochs)")
        print(f"{'='*70}")
        
        try:
            exp_dir, results = train_with_analysis_100epochs(
                optimizer_name=optimizer_name,
                lr=lr,
                num_epochs=100,
                dataset_name=dataset_name
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
    print(f"\n{'='*70}")
    print("FINAL SUMMARY - 100 EPOCHS TRAINING")
    print(f"{'='*70}")
    
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

def run_single_optimizer_100epochs(optimizer_name, lr, dataset_name='svhn'):
    """Run 100-epoch training for a single optimizer"""
    
    print(f"\n{'='*70}")
    print(f"Running {optimizer_name.upper()} with lr={lr} on {dataset_name.upper()} (100 epochs)")
    print(f"Analysis schedule: epochs 0-20 (every epoch), then 40, 60, 80, 100")
    print(f"{'='*70}")
    
    try:
        exp_dir, results = train_with_analysis_100epochs(
            optimizer_name=optimizer_name,
            lr=lr,
            num_epochs=100,
            dataset_name=dataset_name
        )
        
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE - {optimizer_name.upper()}")
        print(f"{'='*70}")
        print(f"Final Train Loss: {results['train_losses'][-1]:.6f}")
        print(f"Final Val Loss: {results['val_losses'][-1]:.6f}")
        print(f"Final Correlations:")
        for layer_idx in range(5):
            corr = results['layer_correlations'][layer_idx][-1]
            print(f"  Layer {layer_idx}: {corr:.6f}")
        print(f"Results saved in: {exp_dir}")
        
        return exp_dir, results
        
    except Exception as e:
        print(f"Error training {optimizer_name}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='100-Epoch Training and AGOP Analysis')
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam', 'muon', 'all'], 
                       default='all', help='Optimizer to use')
    parser.add_argument('--lr', type=float, help='Learning rate (auto-set if not specified)')
    parser.add_argument('--dataset', type=str, default='svhn', 
                        help='Dataset to use for training (default: svhn). Options: svhn, cifar, cifar_mnist, celeba, stl_star')
    
    args = parser.parse_args()
    
    print("=== 100-Epoch Training and AGOP Analysis ===")
    print(f"Dataset: {args.dataset}")
    print("Analysis schedule: epochs 0-20 (every epoch), then 40, 60, 80, 100")
    print("IMPORTANT: AGOP analysis uses ALL training samples for accurate correlation")
    print("(Memory usage will be higher but ensures theoretical correctness)\n")
    
    # Validate dataset
    try:
        dataset_info = get_dataset_info(args.dataset)
        print(f"Dataset configuration:")
        print(f"  Name: {args.dataset}")
        print(f"  Classes: {dataset_info['num_classes']}")
        print(f"  Input size: {dataset_info['input_size']}x{dataset_info['input_size']}")
        print(f"  Channels: {dataset_info['channels']}\n")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    if args.optimizer == 'all':
        # Run all optimizers
        results_summary = run_all_optimizers_100epochs(args.dataset)
    else:
        # Run single optimizer
        if args.lr is None:
            # Set default learning rates
            default_lrs = {'sgd': 0.1, 'adam': 0.001, 'muon': 0.01}
            lr = default_lrs[args.optimizer]
        else:
            lr = args.lr
        
        exp_dir, results = run_single_optimizer_100epochs(args.optimizer, lr, args.dataset)
    
    print("\n=== TRAINING COMPLETE ===")
    print("Check the experiments/ directory for detailed results and plots.")
