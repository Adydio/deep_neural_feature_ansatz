
import numpy as np
import torch
import random
import dataset
import trainer
import neural_model
import argparse
import os
from datetime import datetime

# Enable cuDNN benchmark for performance
torch.backends.cudnn.benchmark = True

SEED = 1717

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)

def get_name(dataset_name, configs):
    name_str = dataset_name + '_'
    for key in configs:
        if key not in ['exp_dir']:  # Skip exp_dir as it's a path
            name_str += key + '_' + str(configs[key]) + '_'
    name_str += 'nn'
    return name_str

def get_dataset_info(dataset_name):
    """Get dataset-specific information"""
    dataset_configs = {
        'svhn': {
            'num_classes': 10,
            'loader_func': lambda: dataset.get_svhn()
        },
        'cifar': {
            'num_classes': 10,
            'loader_func': lambda: dataset.get_cifar()
        },
        'cifar_mnist': {
            'num_classes': 10,
            'loader_func': lambda: dataset.get_cifar_mnist()
        },
        'celeba': {
            'num_classes': 2,
            'loader_func': lambda: dataset.get_celeba(feature_idx=20)  # Use feature 20 (Male)
        },
        'stl_star': {
            'num_classes': 2,
            'loader_func': lambda: dataset.get_stl_star()
        }
    }
    
    if dataset_name not in dataset_configs:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: {list(dataset_configs.keys())}")
    
    return dataset_configs[dataset_name]

def setup_experiment_dir(dataset_name, optimizer, timestamp=None):
    """Setup experiment directory in experiments/ folder"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    exp_name = f"{dataset_name}_{optimizer}_{timestamp}"
    exp_dir = os.path.join("experiments", exp_name)
    
    # Create directories
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "models"), exist_ok=True)
    
    print(f"Experiment directory created: {exp_dir}")
    return exp_dir

def main():
    parser = argparse.ArgumentParser(description='Train Deep NFA model on various datasets')
    parser.add_argument('--dataset', type=str, default='svhn', 
                        help='Dataset to use for training (default: svhn). Options: svhn, cifar, cifar_mnist, celeba, stl_star')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam', 'muon'], 
                        help='Optimizer (default: sgd)')
    parser.add_argument('--lr', type=float, default=0.1, 
                        help='Learning rate (default: 0.1)')
    parser.add_argument('--num_epochs', type=int, default=500, 
                        help='Number of epochs (default: 500)')
    parser.add_argument('--width', type=int, default=1024, 
                        help='Width of hidden layers (default: 1024)')
    parser.add_argument('--depth', type=int, default=5, 
                        help='Depth of the network (default: 5)')
    parser.add_argument('--act', type=str, default='relu', 
                        help='Activation function (default: relu)')
    parser.add_argument('--val_interval', type=int, default=20, 
                        help='Validation interval (default: 20)')
    
    args = parser.parse_args()
    
    # Get dataset information
    dataset_info = get_dataset_info(args.dataset)
    
    # Setup experiment directory
    exp_dir = setup_experiment_dir(args.dataset, args.optimizer)
    
    # Pick configs to save model
    configs = {}
    configs['num_epochs'] = args.num_epochs
    configs['learning_rate'] = args.lr
    configs['weight_decay'] = 0
    configs['init'] = 'default'
    configs['optimizer'] = args.optimizer
    configs['freeze'] = False
    configs['width'] = args.width
    configs['depth'] = args.depth
    configs['act'] = args.act
    configs['val_interval'] = args.val_interval
    configs['exp_dir'] = exp_dir
    
    # Print configuration
    print(f"Training configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Number of classes: {dataset_info['num_classes']}")
    print(f"  Experiment directory: {exp_dir}")
    for key, value in configs.items():
        if key not in ['exp_dir']:
            print(f"  {key}: {value}")
    
    # Load dataset using the appropriate loader function
    trainloader, valloader, testloader = dataset_info['loader_func']()
    
    # Train network
    trainer.train_network(trainloader, valloader, testloader, dataset_info['num_classes'],
                         name=get_name(args.dataset, configs), configs=configs)

if __name__ == "__main__":
    main()
