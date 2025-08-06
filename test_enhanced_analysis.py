#!/usr/bin/env python3
"""
Enhanced Analysis Test Script

Creates a simulated test to demonstrate the enhanced analysis functionality
before running on real trained models.

Usage:
    python test_enhanced_analysis.py
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

def create_mock_experiment_data():
    """Create mock experiment data for testing"""
    
    # Create experiments directory
    os.makedirs('experiments', exist_ok=True)
    
    # Mock model names
    model_names = ['adam_20250803_173338', 'muon_20250804_041908', 'sgd_20250803_164335']
    
    for model_name in model_names:
        model_dir = os.path.join('experiments', model_name)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(os.path.join(model_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(model_dir, 'plots'), exist_ok=True)
        
        # Create mock results.json
        epochs = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
        
        # Mock data that shows realistic training behavior
        results = {
            'epochs': epochs,
            'train_losses': {},
            'val_losses': {},
            'layer_correlations': {}
        }
        
        # Generate realistic training curves for each layer
        for layer_idx in range(5):
            # Training loss: starts high, decreases with some noise
            base_loss = 2.0 - layer_idx * 0.2
            train_losses = []
            val_losses = []
            correlations = []
            
            for i, epoch in enumerate(epochs):
                # Training loss decreases with noise
                train_loss = base_loss * np.exp(-epoch / 100) + np.random.normal(0, 0.05)
                val_loss = train_loss * 1.1 + np.random.normal(0, 0.03)
                
                # Correlation increases with training
                correlation = min(0.9, 0.1 + (epoch / 200) * 0.8 + np.random.normal(0, 0.05))
                correlation = max(0.0, correlation)
                
                train_losses.append(max(0.001, train_loss))
                val_losses.append(max(0.001, val_loss))
                correlations.append(correlation)
            
            results['train_losses'][layer_idx] = train_losses
            results['val_losses'][layer_idx] = val_losses
            results['layer_correlations'][layer_idx] = correlations
        
        # Save results.json
        with open(os.path.join(model_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Created mock data for {model_name}")

def create_simplified_enhanced_plot():
    """Create a simplified enhanced plot without actual model computation"""
    
    # Create mock enhanced data
    epochs = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Enhanced Training Analysis - DEMO (Mock Data)', fontsize=16, fontweight='bold')
    
    plot_positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    
    for i, layer_idx in enumerate(range(5)):
        row, col = plot_positions[i]
        ax1 = axes[row, col]
        
        # Mock training curves
        train_loss = 1.5 * np.exp(-np.array(epochs) / 80) + np.random.normal(0, 0.03, len(epochs))
        val_loss = train_loss * 1.1 + np.random.normal(0, 0.02, len(epochs))
        
        # Primary y-axis: Loss
        ax1.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2, marker='o', markersize=4)
        ax1.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2, marker='s', markersize=4)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        
        # Secondary y-axis: Similarities
        ax2 = ax1.twinx()
        
        # Mock similarity curves
        agop_nfm = 0.1 + np.array(epochs) / 250 + np.random.normal(0, 0.05, len(epochs))
        g_h_sim = 0.3 + np.sin(np.array(epochs) / 50) * 0.2 + np.random.normal(0, 0.03, len(epochs))
        g_m_sim = 0.5 + np.array(epochs) / 400 + np.random.normal(0, 0.04, len(epochs))
        h_m_sim = 0.2 + np.array(epochs) / 300 + np.random.normal(0, 0.03, len(epochs))
        
        # Clip to [0, 1]
        agop_nfm = np.clip(agop_nfm, 0, 1)
        g_h_sim = np.clip(g_h_sim, 0, 1)
        g_m_sim = np.clip(g_m_sim, 0, 1)
        h_m_sim = np.clip(h_m_sim, 0, 1)
        
        ax2.plot(epochs, agop_nfm, 'g-', label='AGOP/NFM Corr', linewidth=2, marker='^', markersize=4)
        ax2.plot(epochs, g_h_sim, 'm-', label='G vs H', linewidth=2, marker='d', markersize=4)
        ax2.plot(epochs, g_m_sim, 'c-', label='G vs M', linewidth=2, marker='v', markersize=4)
        ax2.plot(epochs, h_m_sim, 'orange', label='H vs M', linewidth=2, marker='*', markersize=4)
        
        ax2.set_ylabel('Cosine Similarity', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.set_ylim(0, 1)
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
        
        ax1.set_title(f'Layer {layer_idx}', fontweight='bold')
        ax1.grid(True, alpha=0.3)
    
    # Add explanation in the last subplot
    axes[1, 2].axis('off')
    axes[1, 2].text(0.1, 0.7, 
                   'Matrix Definitions:\n'
                   '• H_i := E[h_i h_i^T] (Representation)\n'
                   '• G_i := E[g_i g_i^T] (Gradient)\n'
                   '• M_i := W_i^T W_i (Weight/NFM)\n\n'
                   'Similarities:\n'
                   '• G vs H: Gradient-Representation\n'
                   '• G vs M: Gradient-Weight\n'
                   '• H vs M: Representation-Weight\n\n'
                   'NOTE: This is DEMO data\n'
                   'Run on real trained models\n'
                   'for actual results',
                   fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    plt.tight_layout()
    
    # Save demo plot
    os.makedirs('enhanced_analysis', exist_ok=True)
    output_path = 'enhanced_analysis/demo_enhanced_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Demo enhanced analysis plot saved: {output_path}")
    
    plt.show()
    plt.close()

def main():
    print("=== Enhanced Analysis Test Script ===")
    print("Creating mock experiment data...")
    
    # Create mock data
    create_mock_experiment_data()
    
    print("\nCreating demo enhanced analysis plot...")
    create_simplified_enhanced_plot()
    
    print("\n✅ Test completed!")
    print("\nNext steps:")
    print("1. Run actual training to get real models")
    print("2. Use enhanced_analysis_visualization.py on real trained models")
    print("3. Example command:")
    print("   python enhanced_analysis_visualization.py --models adam_20250803_173338 muon_20250804_041908 sgd_20250803_164335")

if __name__ == "__main__":
    main()
