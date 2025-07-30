
import numpy as np
import torch
import random
import dataset
import trainer
import argparse

# Enable cuDNN benchmark for performance
torch.backends.cudnn.benchmark = True

SEED = 1717

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)

def get_name(dataset_name, configs):
    name_str = dataset_name + ':'
    for key in configs:
        name_str += key + ':' + str(configs[key]) + ':'
    name_str += 'nn'
    return name_str

def main():
    parser = argparse.ArgumentParser(description='Train neural net with flexible validation interval and optimizer.')
    parser.add_argument('--val_interval', type=int, default=20, help='Validation interval (default: 20)')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam', 'muon'], help='Optimizer (default: sgd)')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate (default: 0.1)')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs (default: 500)')
    args = parser.parse_args()

    # Pick configs to save model
    configs = {}
    configs['num_epochs'] = args.num_epochs
    configs['learning_rate'] = args.lr
    configs['weight_decay'] = 0
    configs['init'] = 'default'
    configs['optimizer'] = args.optimizer
    configs['freeze'] = False
    configs['width'] = 1024
    configs['depth'] = 5
    configs['act'] = 'relu'
    configs['val_interval'] = args.val_interval

    # Code to load and train net on selected dataset.
    NUM_CLASSES = 10
    trainloader, valloader, testloader = dataset.get_svhn()
    trainer.train_network(trainloader, valloader, testloader, NUM_CLASSES,
                         name=get_name('svhn', configs), configs=configs)

if __name__ == "__main__":
    main()
