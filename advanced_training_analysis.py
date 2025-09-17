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

def pearson_correlate(M, G):
    """
    Compute Pearson correlation coefficient between two matrices.
    
    Args:
        M: Neural Feature Matrix (torch.Tensor)
        G: EGOP matrix (torch.Tensor)
    
    Returns:
        Pearson correlation coefficient (torch.Tensor)
    """
    M = M.double()
    G = G.double()
    
    # Flatten matrices
    M_flat = M.flatten()
    G_flat = G.flatten()
    
    # Center the data (subtract mean)
    M_centered = M_flat - M_flat.mean()
    G_centered = G_flat - G_flat.mean()
    
    # Compute Pearson correlation
    numerator = torch.dot(M_centered, G_centered)
    denominator = torch.sqrt(torch.sum(M_centered**2) * torch.sum(G_centered**2))
    
    # Avoid division by zero
    if denominator == 0:
        return torch.tensor(0.0, dtype=torch.double)
    
    correlation = numerator / denominator
    return correlation

def initialize_all_layer_weights(net, depth, init_value=1e-4, use_original_strategy=True):
    """
    Initialize layer weights to match original paper's strategy exactly
    
    Args:
        net: The neural network
        depth: Network depth 
        init_value: Value for first layer initialization (default: 1e-4)
        use_original_strategy: If True, match original trainer.py exactly
    """
    if use_original_strategy:
        # Original trainer.py strategy: only initialize first layer if init != 'default'
        if init_value != 'default':
            for idx, param in enumerate(net.parameters()):
                if idx == 0:  # Only first layer
                    init = torch.Tensor(param.size()).normal_().float() * init_value
                    param.data = init
                    print(f"Original strategy - layer {idx}: mean={param.mean().item():.8f}, "
                          f"std={param.std().item():.8f}, shape={param.shape}")
        
        # Other layers keep PyTorch default initialization
        print("Other layers: PyTorch default initialization (Uniform)")
        for idx, param in enumerate(net.parameters()):
            if idx > 0 and param.dim() == 2:
                print(f"Layer {idx} (default): mean={param.mean().item():.8f}, "
                      f"std={param.std().item():.8f}, shape={param.shape}")
        print(f"Total layers: 1 custom + {sum(1 for p in net.parameters() if p.dim() == 2) - 1} default")
    else:
        # Your mixed strategy
        with torch.no_grad():
            layer_count = 0
            for name, param in net.named_parameters():
                if 'weight' in name and param.dim() == 2:  # Only Linear layer weights
                    if layer_count == 0:
                        # First layer: Gaussian initialization (equivalent to original)
                        torch.nn.init.normal_(param, mean=0.0, std=init_value)
                        print(f"Mixed strategy - layer {layer_count} ({name}): mean={param.mean().item():.8f}, "
                              f"std={param.std().item():.8f}, shape={param.shape}")
                    else:
                        # Other layers: Kaiming initialization
                        torch.nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
                        print(f"Mixed strategy - layer {layer_count} ({name}): mean={param.mean().item():.8f}, "
                              f"std={param.std().item():.8f}, shape={param.shape}")
                    layer_count += 1
            print(f"Total layers: 1 Gaussian + {layer_count-1} Kaiming")

def get_dataset_info(dataset_name):
    """Get dataset-specific information including loss axis ranges for consistent plotting"""
    dataset_configs = {
        # Image datasets
        'svhn': {
            'num_classes': 10,
            'loader_func': dataset.get_svhn,
            'input_size': 32,
            'channels': 3,
            'loss_ylim': (0, 0.1),  # Consistent loss range for SVHN across all optimizers
            'type': 'image'
        },
        'cifar': {
            'num_classes': 10,
            'loader_func': dataset.get_cifar,
            'input_size': 32,
            'channels': 3,
            'loss_ylim': (0, 0.15),  # Consistent loss range for CIFAR across all optimizers
            'type': 'image'
        },
        'cifar_mnist': {
            'num_classes': 10,
            'loader_func': dataset.get_cifar_mnist,
            'input_size': 32,
            'channels': 3,
            'loss_ylim': (0, 0.12),  # Consistent loss range for CIFAR-MNIST across all optimizers
            'type': 'image'
        },
        'celeba': {
            'num_classes': 2,
            'loader_func': lambda: dataset.get_celeba(feature_idx=20),
            'input_size': 96,
            'channels': 3,
            'loss_ylim': (0, 0.25),  # Consistent loss range for CelebA across all optimizers
            'type': 'image'
        },
        'stl_star': {
            'num_classes': 2,
            'loader_func': dataset.get_stl_star,
            'input_size': 96,
            'channels': 3,
            'loss_ylim': (0, 0.25),  # Consistent loss range for STL-Star across all optimizers
            'type': 'image'
        },
        'mnist': {
            'num_classes': 10,
            'loader_func': dataset.get_mnist,
            'input_size': 32,
            'channels': 3,
            'loss_ylim': (0, 0.08),  # Consistent loss range for MNIST across all optimizers
            'type': 'image'
        },
        'fashion_mnist': {
            'num_classes': 10,
            'loader_func': dataset.get_fashion_mnist,
            'input_size': 32,
            'channels': 3,
            'loss_ylim': (0, 0.12),  # Consistent loss range for Fashion-MNIST across all optimizers
            'type': 'image'
        },
        # Tabular datasets (OpenML Tabular Benchmark) - Note: these return scaler as 4th element
        'credit': {
            'num_classes': 2,
            'loader_func': dataset.get_credit,
            'input_size': 14,  # Australian Credit features
            'channels': 1,     # Not applicable for tabular
            'loss_ylim': (0, 0.2),  # Binary classification
            'type': 'tabular'
        },
        'electricity': {
            'num_classes': 2,
            'loader_func': dataset.get_electricity,
            'input_size': 8,   # Electricity pricing features
            'channels': 1,     # Not applicable for tabular
            'loss_ylim': (0, 0.25),  # Binary classification
            'type': 'tabular'
        },
        'covertype': {
            'num_classes': 7,
            'loader_func': dataset.get_covertype,
            'input_size': 54,  # Forest cover type features
            'channels': 1,     # Not applicable for tabular
            'loss_ylim': (0, 0.3),  # 7-class classification
            'type': 'tabular'
        },
        'pol': {
            'num_classes': 2,
            'loader_func': dataset.get_pol,
            'input_size': 48,  # Political voting features (corrected from 27 to 48)
            'channels': 1,     # Not applicable for tabular
            'loss_ylim': (0, 0.2),  # Binary classification
            'type': 'tabular'
        },
        'house_16H': {
            'num_classes': 2,
            'loader_func': dataset.get_house_16H,
            'input_size': 16,  # House pricing features (corrected from 17 to 16)
            'channels': 1,     # Not applicable for tabular
            'loss_ylim': (0, 0.2),  # Binary classification
            'type': 'tabular'
        },
        'MagicTelescope': {
            'num_classes': 2,
            'loader_func': dataset.get_MagicTelescope,
            'input_size': 10,  # Magic Telescope features
            'channels': 1,     # Not applicable for tabular
            'loss_ylim': (0, 0.15),  # Binary classification
            'type': 'tabular'
        },
        'credit_g': {
            'num_classes': 2,
            'loader_func': dataset.get_credit_g,
            'input_size': 20,  # German Credit features
            'channels': 1,     # Not applicable for tabular
            'loss_ylim': (0, 0.25),  # Binary classification
            'type': 'tabular'
        },
        'kr_vs_kp': {
            'num_classes': 2,
            'loader_func': dataset.get_kr_vs_kp,
            'input_size': 36,  # Chess board features
            'channels': 1,     # Not applicable for tabular
            'loss_ylim': (0, 0.1),  # Binary classification
            'type': 'tabular'
        }
    }

    if dataset_name not in dataset_configs:
        supported_datasets = list(dataset_configs.keys())
        image_datasets = [k for k, v in dataset_configs.items() if v['type'] == 'image']
        tabular_datasets = [k for k, v in dataset_configs.items() if v['type'] == 'tabular']
        raise ValueError(f"Unsupported dataset: {dataset_name}. "
                        f"Supported image datasets: {image_datasets}. "
                        f"Supported tabular datasets: {tabular_datasets}.")
    
    return dataset_configs[dataset_name]

def setup_experiment_dir(optimizer_name, dataset_name='svhn', configs=None):
    """Create experiment directory structure with detailed parameter information"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create detailed directory name with all parameters
    if configs:
        lr = configs.get('learning_rate', 'unknown')
        epochs = configs.get('num_epochs', 'unknown')
        weight_decay = configs.get('weight_decay', 0)
        width = configs.get('width', 1024)
        depth = configs.get('depth', 5)
        act = configs.get('act', 'relu')
        val_interval = configs.get('val_interval', 20)
        init_strategy = configs.get('init_strategy', 'original')
        correlation_type = configs.get('correlation_type', 'cosine')
        
        # Format weight decay for filename (remove dots and scientific notation)
        if weight_decay == 0:
            wd_str = "wd0"
        elif weight_decay >= 1e-3:
            wd_str = f"wd{weight_decay:.3f}".replace('.', '_')
        else:
            # For very small weight decay, use scientific notation format
            wd_str = f"wd{weight_decay:.0e}".replace('-', 'm').replace('+', 'p')
        
        # Format learning rate for filename
        if isinstance(lr, float):
            if lr >= 0.001:
                lr_str = f"lr{lr:.3f}".replace('.', '_')
            else:
                lr_str = f"lr{lr:.0e}".replace('-', 'm').replace('+', 'p')
        else:
            lr_str = f"lr{lr}"
        
        # Create comprehensive directory name
        exp_dir = (f"experiments/{dataset_name}_{optimizer_name}_{lr_str}_{wd_str}_"
                  f"ep{epochs}_int{val_interval}_w{width}_d{depth}_{act}_{init_strategy}_{correlation_type}_{timestamp}")
    else:
        # Fallback to simple naming
        exp_dir = f"experiments/{dataset_name}_{optimizer_name}_default_{timestamp}"
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f"{exp_dir}/models", exist_ok=True)
    os.makedirs(f"{exp_dir}/plots", exist_ok=True)
    
    return exp_dir

def compute_agop_nfm_correlation(model_path, layer_indices, max_samples=None, init_model_path=None, dataset_name='svhn', correlation_type='cosine'):
    """
    AGOP vs NFM correlation computation
    
    Args:
        model_path: Path to the saved model
        layer_indices: List of layer indices to analyze
        max_samples: limit samples for AGOP computation (memory management)
        init_model_path: Path to initial model (for remove_init operation)
        dataset_name: dataset to use for AGOP computation
        correlation_type: 'cosine' or 'pearson' - method for computing correlation
    
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
    
    print(f"Using {correlation_type.upper()} correlation method")
    
    try:
        # Read model config from path
        width, depth, act_name = read_configs(model_path)
        
        # Get dataset-specific parameters
        NUM_CLASSES = dataset_info['num_classes']
        dataset_type = dataset_info['type']
        
        if dataset_type == 'image':
            SIZE = dataset_info['input_size']
            c = dataset_info['channels']
            dim = c * SIZE * SIZE
        else:  # tabular
            dim = dataset_info['input_size']  # Number of features
        
        # Load correct dataset based on dataset_name
        scaler = None  # For tabular datasets
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
        elif dataset_name == 'mnist':
            trainloader, valloader, testloader = dataset.get_mnist()
        elif dataset_name == 'fashion_mnist':
            trainloader, valloader, testloader = dataset.get_fashion_mnist()
        # Tabular datasets - these return scaler as 4th element
        elif dataset_name == 'credit':
            trainloader, valloader, testloader, scaler = dataset.get_credit()
        elif dataset_name == 'electricity':
            trainloader, valloader, testloader, scaler = dataset.get_electricity()
        elif dataset_name == 'covertype':
            trainloader, valloader, testloader, scaler = dataset.get_covertype()
        elif dataset_name == 'pol':
            trainloader, valloader, testloader, scaler = dataset.get_pol()
        elif dataset_name == 'house_16H':
            trainloader, valloader, testloader, scaler = dataset.get_house_16H()
        elif dataset_name == 'MagicTelescope':
            trainloader, valloader, testloader, scaler = dataset.get_MagicTelescope()
        elif dataset_name == 'credit_g':
            trainloader, valloader, testloader, scaler = dataset.get_credit_g()
        elif dataset_name == 'kr_vs_kp':
            trainloader, valloader, testloader, scaler = dataset.get_kr_vs_kp()
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        # Log whether StandardScaler is used
        if scaler is not None:
            print(f"Using StandardScaler for tabular dataset: {dataset_name}")
        else:
            print(f"No scaling applied for image dataset: {dataset_name}")
        
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
            
            # Compute AGOP - use uncentered for both methods
            # Pearson correlation will handle centering in its own calculation
            # Cosine similarity works with uncentered data (original approach)
            G = egop(subnet, out, centering=False)
            print(f"    Using uncentered EGOP (centering handled by correlation method if needed)")
            
            # Compute correlation based on type
            if correlation_type.lower() == 'pearson':
                correlation = pearson_correlate(torch.from_numpy(M), G)
                print(f"    Computing Pearson correlation")
            else:  # cosine (default)
                correlation = correlate(torch.from_numpy(M), G)  # Original cosine similarity
                print(f"    Computing Cosine similarity")
                
            correlations[layer_idx] = correlation.item()
            
            print(f"Layer {layer_idx} AGOP/NFM correlation: {correlation.item():.6f}")
    
    except Exception as e:
        print(f"Error computing AGOP/NFM correlation: {e}")
        for layer_idx in layer_indices:
            correlations[layer_idx] = 0.0
    
    return correlations

def train_with_analysis(optimizer_name, lr, num_epochs=500, val_interval=20, max_samples=None, weight_decay=0, dataset_name='svhn', correlation_type='cosine'):
    """
    Train model with comprehensive analysis
    
    Args:
        optimizer_name: 'sgd', 'adam', or 'muon'
        lr: learning rate
        num_epochs: total epochs
        val_interval: interval for saving and analysis
        max_samples: limit samples for AGOP computation (memory management)
        dataset_name: dataset to use for training
        correlation_type: 'cosine' or 'pearson' - method for computing AGOP correlation
    """
    
    print(f"\n=== Starting Training with {optimizer_name.upper()} on {dataset_name.upper()} ===")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {num_epochs}")
    print(f"Analysis interval: {val_interval}")
    print(f"Correlation type: {correlation_type.upper()}")
    
    # Set random seed
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    
    # Get dataset info and load data
    dataset_info = get_dataset_info(dataset_name)
    scaler = None  # For tabular datasets
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
    elif dataset_name == 'mnist':
        trainloader, valloader, testloader = dataset.get_mnist()
    elif dataset_name == 'fashion_mnist':
        trainloader, valloader, testloader = dataset.get_fashion_mnist()
    # Tabular datasets - these return scaler as 4th element
    elif dataset_name == 'credit':
        trainloader, valloader, testloader, scaler = dataset.get_credit()
    elif dataset_name == 'electricity':
        trainloader, valloader, testloader, scaler = dataset.get_electricity()
    elif dataset_name == 'covertype':
        trainloader, valloader, testloader, scaler = dataset.get_covertype()
    elif dataset_name == 'pol':
        trainloader, valloader, testloader, scaler = dataset.get_pol()
    elif dataset_name == 'house_16H':
        trainloader, valloader, testloader, scaler = dataset.get_house_16H()
    elif dataset_name == 'MagicTelescope':
        trainloader, valloader, testloader, scaler = dataset.get_MagicTelescope()
    elif dataset_name == 'credit_g':
        trainloader, valloader, testloader, scaler = dataset.get_credit_g()
    elif dataset_name == 'kr_vs_kp':
        trainloader, valloader, testloader, scaler = dataset.get_kr_vs_kp()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Log whether StandardScaler is used
    if scaler is not None:
        print(f"Using StandardScaler for tabular dataset: {dataset_name}")
    else:
        print(f"No scaling applied for image dataset: {dataset_name}")
    
    # Get input dimension (works for both image and tabular data)
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
        'val_interval': val_interval,
        'init_strategy': 'original',  # Track initialization strategy
        'correlation_type': correlation_type  # Track correlation method
    }
    
    # Setup experiment directory with detailed parameters
    exp_dir = setup_experiment_dir(optimizer_name, dataset_name, configs)
    print(f"Experiment directory: {exp_dir}")
    
    # Create model
    net = neural_model.Net(dim, width=configs['width'],
                          depth=configs['depth'],
                          num_classes=dataset_info['num_classes'],
                          act_name=configs['act'])
    
    # Initialize weights using original paper's exact strategy
    print(f"\n=== Initializing Network Weights ===")
    initialize_all_layer_weights(net, configs['depth'], init_value=1e-4, use_original_strategy=True)
    
    # Get device and setup
    device = trainer.get_best_device()
    net = net.to(device)
    
    # Setup optimizer
    optimizer = trainer.select_optimizer(optimizer_name, lr, net, configs['weight_decay'])
    
    # Enable torch.compile for muon
    if optimizer_name == 'muon':
        net = torch.compile(net, mode='reduce-overhead')
    
    criterion = torch.nn.MSELoss()
    # Use updated amp.GradScaler syntax
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
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
                                                           dataset_name=dataset_name,
                                                           correlation_type=correlation_type)
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
    
    # Generate plots with configuration parameters
    generate_plots(results, exp_dir, optimizer_name, dataset_name, configs)
    
    print(f"\nTraining completed! Results saved in: {exp_dir}")
    return exp_dir, results

def generate_plots(results, exp_dir, optimizer_name, dataset_name='svhn', configs=None):
    """Generate comprehensive visualization plots"""
    
    print("Generating plots...")
    
    # Get dataset configuration for consistent plotting
    dataset_info = get_dataset_info(dataset_name)
    loss_ylim = dataset_info['loss_ylim']
    
    # Create detailed filename with all parameters
    if configs:
        lr = configs.get('learning_rate', 'unknown')
        epochs = configs.get('num_epochs', 'unknown')
        weight_decay = configs.get('weight_decay', 0)
        width = configs.get('width', 1024)
        depth = configs.get('depth', 5)
        act = configs.get('act', 'relu')
        val_interval = configs.get('val_interval', 20)
        
        # Format weight decay for filename (remove dots and scientific notation)
        if weight_decay == 0:
            wd_str = "wd0"
        elif weight_decay >= 1e-3:
            wd_str = f"wd{weight_decay:.3f}".replace('.', '_')
        else:
            # For very small weight decay, use scientific notation format
            wd_str = f"wd{weight_decay:.0e}".replace('-', 'm').replace('+', 'p')
        
        # Format learning rate for filename
        if isinstance(lr, float):
            if lr >= 0.001:
                lr_str = f"lr{lr:.3f}".replace('.', '_')
            else:
                lr_str = f"lr{lr:.0e}".replace('-', 'm').replace('+', 'p')
        else:
            lr_str = f"lr{lr}"
        
        base_filename = f"{dataset_name}_{optimizer_name}_{lr_str}_{wd_str}_ep{epochs}_int{val_interval}_w{width}_d{depth}_{act}_original_init"
    else:
        base_filename = f"{dataset_name}_{optimizer_name}_default_params"
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create subplots: 5 layers in a 2x3 grid (with one empty subplot)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Create detailed title with parameters
    if configs:
        title = (f'{optimizer_name.upper()} Training Analysis: {dataset_name.upper()}\n'
                f'LR={lr}, WD={weight_decay}, Epochs={epochs}, Interval={val_interval}, '
                f'Arch=[{width}×{depth}], Act={act}, Init=Original(L0:1e-4*Normal, L1+:Default)')
    else:
        title = f'{optimizer_name.upper()} Training Analysis: Loss and AGOP/NFM Correlation'
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
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
    
    # Save plot with detailed filename
    plot_path = f'{exp_dir}/plots/{base_filename}_training_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(f'{exp_dir}/plots/{base_filename}_training_analysis.pdf', bbox_inches='tight')
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
        
        # Detailed title for individual layer plots
        if configs:
            layer_title = (f'{optimizer_name.upper()} Layer {layer_idx} - {dataset_name.upper()}\n'
                          f'LR={lr}, WD={weight_decay}, Epochs={epochs}, Init=Original(L0:1e-4*Normal, L1+:Default)')
        else:
            layer_title = f'{optimizer_name.upper()} - Layer {layer_idx} Analysis'
        
        ax.set_title(layer_title, fontweight='bold')
        
        ax.tick_params(axis='y', labelcolor='black')
        ax2.tick_params(axis='y', labelcolor='green')
        ax.grid(True, alpha=0.3)
        
        # Legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        layer_plot_path = f'{exp_dir}/plots/{base_filename}_layer_{layer_idx}_analysis.png'
        plt.savefig(layer_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Plots saved in: {exp_dir}/plots/")
    print(f"Main plot: {base_filename}_training_analysis.png")
    print(f"Layer plots: {base_filename}_layer_{{0-4}}_analysis.png")

def main():
    parser = argparse.ArgumentParser(description='Advanced Training and AGOP Analysis')
    parser.add_argument('--optimizer', type=str, required=True, 
                        choices=['sgd', 'adam', 'adamw', 'muon'],
                        help='Optimizer to use')
    parser.add_argument('--dataset', type=str, default='svhn',
                        choices=['svhn', 'cifar', 'cifar_mnist', 'celeba', 'stl_star', 'mnist', 'fashion_mnist',
                                'credit', 'electricity', 'covertype', 'pol', 'house_16H', 
                                'MagicTelescope', 'credit_g', 'kr_vs_kp'],
                        help='Dataset to use (default: svhn). '
                             'Image datasets: svhn, cifar, cifar_mnist, celeba, stl_star, mnist, fashion_mnist. '
                             'Tabular datasets (OpenML benchmark): credit, electricity, covertype, pol, house_16H, MagicTelescope, credit_g, kr_vs_kp.')
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
    parser.add_argument('--correlation_type', type=str, default='cosine',
                        choices=['cosine', 'pearson'],
                        help='Correlation type for AGOP-NFM analysis (default: cosine)')
    
    args = parser.parse_args()
    
    # Auto-select learning rate if not provided
    if args.lr is None:
        lr_defaults = {'sgd': 0.1, 'adam': 0.001, 'adamw': 0.001, 'muon': 0.01}
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
        dataset_name=args.dataset,
        correlation_type=args.correlation_type
    )
    
    print(f"\n=== Experiment Complete ===")
    print(f"Results directory: {exp_dir}")
    print(f"Generated plots: {exp_dir}/plots/")
    print(f"Model checkpoints: {exp_dir}/models/")
    print(f"Raw data: {exp_dir}/results.json")

if __name__ == "__main__":
    main()
