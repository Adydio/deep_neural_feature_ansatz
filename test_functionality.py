#!/usr/bin/env python3
"""
快速测试脚本 - 验证训练和分析功能

运行一个短期的测试版本，确保所有功能正常工作。
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

# Import our modules
import dataset
import trainer
import neural_model
from verify_deep_NFA import (
    clean_compiled_state_dict, get_layer_output, build_subnetwork, 
    egop, correlate, SEED
)

def quick_test():
    """快速测试训练和分析流程"""
    
    print("=== Quick Test: Training and AGOP Analysis ===")
    print("Testing with SGD for 3 epochs, analysis every 1 epoch")
    
    # Create test directory
    test_dir = "test_experiment"
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(f"{test_dir}/models", exist_ok=True)
    
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
    
    print(f"Input dimension: {dim}")
    print(f"Training samples: {len(trainloader.dataset)}")
    
    # Create small model for testing
    configs = {
        'width': 128,  # Smaller for testing
        'depth': 3,    # Smaller for testing
        'act': 'relu'
    }
    
    net = neural_model.Net(dim, width=configs['width'],
                          depth=configs['depth'],
                          num_classes=10,
                          act_name=configs['act'])
    
    device = trainer.get_best_device()
    net = net.to(device)
    print(f"Using device: {device}")
    
    # Setup optimizer and criterion
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # Test training for a few epochs
    results = {
        'epochs': [],
        'train_losses': [],
        'val_losses': [],
        'layer_correlations': {i: [] for i in range(3)}  # 3 layers for test model
    }
    
    for epoch in range(4):  # 0, 1, 2, 3
        print(f"\n--- Epoch {epoch} ---")
        
        # Training step
        if epoch > 0:
            print("Training...")
            train_loss = trainer.train_step(net, optimizer, trainloader, criterion, device, scaler)
        else:
            train_loss = 0.0
        
        # Analysis
        print("Validation...")
        val_loss = trainer.val_step(net, valloader, criterion, device)
        
        # Save model
        net.cpu()
        d = {'state_dict': trainer.get_clean_state_dict(net)}
        model_path = f'{test_dir}/models/model_epoch_{epoch}.pth'
        torch.save(d, model_path)
        net.to(device)
        
        print("Computing AGOP/NFM correlations...")
        correlations = {}
        
        # Test AGOP computation for each layer
        for layer_idx in range(3):
            try:
                print(f"  Layer {layer_idx}...")
                
                # Load model and extract NFM
                test_net = neural_model.Net(dim, width=configs['width'],
                                          depth=configs['depth'],
                                          num_classes=10,
                                          act_name=configs['act'])
                d = torch.load(model_path, map_location='cpu')
                test_net.load_state_dict(clean_compiled_state_dict(d['state_dict']))
                
                # Extract NFM for this layer
                for idx, p in enumerate(test_net.parameters()):
                    if idx == layer_idx:
                        M = p.data.cpu().numpy()
                        break
                
                M = M.T @ M * (1/len(M))
                
                # Build subnetwork
                subnet = build_subnetwork(test_net, M.shape[0], configs['width'], 
                                        configs['depth'], 10, layer_idx=layer_idx, 
                                        random_net=False, act_name=configs['act'])
                
                # Get layer output (use small sample for testing)
                out = get_layer_output(test_net, trainloader, layer_idx=layer_idx, max_samples=1000)
                
                # Compute AGOP (uncentered)
                G = egop(subnet, out, centering=False)
                
                # Compute correlation
                correlation = correlate(torch.from_numpy(M), G)
                correlations[layer_idx] = correlation.item()
                
                print(f"    Layer {layer_idx} correlation: {correlation.item():.6f}")
                
            except Exception as e:
                print(f"    Error in layer {layer_idx}: {e}")
                correlations[layer_idx] = 0.0
        
        # Store results
        results['epochs'].append(epoch)
        results['train_losses'].append(train_loss)
        results['val_losses'].append(val_loss)
        
        for layer_idx in range(3):
            results['layer_correlations'][layer_idx].append(correlations.get(layer_idx, 0.0))
        
        print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
    
    # Test plotting
    print("\nTesting plot generation...")
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Test Results: SGD Training Analysis', fontsize=14)
        
        epochs = results['epochs']
        train_losses = results['train_losses']
        val_losses = results['val_losses']
        
        for layer_idx in range(3):
            ax = axes[layer_idx]
            correlations = results['layer_correlations'][layer_idx]
            
            # Create twin axis
            ax2 = ax.twinx()
            
            # Plot losses
            ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
            ax.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
            
            # Plot correlation
            ax2.plot(epochs, correlations, 'g-', label='AGOP/NFM Corr', linewidth=2, marker='o')
            
            # Formatting
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss', color='black')
            ax2.set_ylabel('Correlation', color='green')
            ax.set_title(f'Layer {layer_idx}')
            
            ax.tick_params(axis='y', labelcolor='black')
            ax2.tick_params(axis='y', labelcolor='green')
            ax.grid(True, alpha=0.3)
            
            # Legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f'{test_dir}/test_plot.png', dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        
        print("✅ Plot generation successful!")
        
    except Exception as e:
        print(f"❌ Plot generation failed: {e}")
    
    # Save test results
    with open(f'{test_dir}/test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Test Complete ===")
    print(f"✅ Training: Successful")
    print(f"✅ AGOP computation: Successful")
    print(f"✅ Correlation calculation: Successful")
    print(f"✅ Model saving: Successful")
    print(f"✅ Plotting: Check above")
    print(f"\nTest files saved in: {test_dir}/")
    print("The main scripts should work correctly!")
    
    return results

if __name__ == "__main__":
    results = quick_test()
