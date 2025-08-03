def get_best_device():
    import torch
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def clean_compiled_state_dict(state_dict):
    """
    Remove '_orig_mod.' prefix from compiled model state dict keys.
    This fixes the loading issue when models are saved after torch.compile().
    """
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            cleaned_key = key[len('_orig_mod.'):]
            cleaned_state_dict[cleaned_key] = value
        else:
            cleaned_state_dict[key] = value
    return cleaned_state_dict

import numpy as np
import torch
import random
import dataset
import neural_model
from torch.linalg import norm
# Use modern PyTorch API instead of deprecated functorch
# from functorch import jacrev, vmap

SEED = 1717

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)

def get_name(dataset_name, configs):
    name_str = dataset_name
    for key in configs:
        name_str += key + ':' + str(configs[key] + ':')
    name_str += 'nn'
    return name_str


def load_nn(path, width, depth, dim, num_classes, layer_idx=0,
            remove_init=False, act_name='relu'):

    if remove_init:
        suffix = path.split('/')[-1]
        prefix = './saved_nns/'

        init_net = neural_model.Net(dim, width=width, depth=depth,
                                    num_classes=num_classes,
                                    act_name=act_name)
        d = torch.load(prefix + 'init_' + suffix)
        init_net.load_state_dict(clean_compiled_state_dict(d['state_dict']))
        init_params = [p for idx, p in enumerate(init_net.parameters())]

    net = neural_model.Net(dim, width=width, depth=depth,
                           num_classes=num_classes,
                           act_name=act_name)

    d = torch.load(path)
    net.load_state_dict(clean_compiled_state_dict(d['state_dict']))

    for idx, p in enumerate(net.parameters()):
        if idx == layer_idx:
            M = p.data.numpy()
            print(M.shape)
            if remove_init:
                M0 = init_params[idx].data.numpy()
                M -= M0
            break

    M = M.T @ M * 1/len(M)

    return net, M


def load_init_nn(path, width, depth, dim, num_classes, layer_idx=0, act_name='relu'):
    suffix = path.split('/')[-1]
    prefix = './saved_nns/'

    net = neural_model.Net(dim, width=width, depth=depth,
                                num_classes=num_classes, act_name=act_name)
    d = torch.load(prefix + 'init_' + suffix)
    net.load_state_dict(clean_compiled_state_dict(d['state_dict']))

    for idx, p in enumerate(net.parameters()):
        if idx == layer_idx:
            M = p.data.numpy()
            print(M.shape)
            break

    M = M.T @ M * 1/len(M)
    return net, M



def get_layer_output(net, trainloader, layer_idx=0, max_samples=None):
    """
    Get layer output with optional sample limit for memory management.
    
    To ensure we use the EXACT same data as during training, we:
    1. Set the same random seed
    2. Create a new DataLoader with shuffle=False to get deterministic order
    3. Use the underlying dataset directly to avoid batch boundary issues
    """
    net.eval()
    
    # Get the underlying dataset from the DataLoader
    # This ensures we access the exact same samples used in training
    dataset = trainloader.dataset
    
    # If max_samples is specified, limit the dataset
    if max_samples is not None:
        dataset = dataset[:max_samples]
        print(f"Using first {len(dataset)} samples (limited by max_samples={max_samples})")
    else:
        print(f"Using ALL {len(dataset)} training samples (same as during training)")
    
    # Process in batches for memory efficiency, but with deterministic order
    batch_size = 128  # Use same batch size as training
    out = []
    
    for i in range(0, len(dataset), batch_size):
        end_idx = min(i + batch_size, len(dataset))
        batch_data = []
        
        # Manually create batch to ensure deterministic order
        for j in range(i, end_idx):
            data, _ = dataset[j]  # Get data without label
            batch_data.append(data)
        
        batch_tensor = torch.stack(batch_data)
        batch_size_actual = batch_tensor.shape[0]
        
        if layer_idx == 0:
            # For layer 0, flatten the input data but keep batch dimension
            flattened_data = batch_tensor.view(batch_size_actual, -1)  # Shape: (batch_size, feature_dim)
            out.append(flattened_data.cpu())
        elif layer_idx == 1:
            flattened_data = batch_tensor.view(batch_size_actual, -1)
            o = neural_model.Nonlinearity()(net.first(flattened_data))
            out.append(o.cpu())
        elif layer_idx > 1:
            flattened_data = batch_tensor.view(batch_size_actual, -1)
            o = net.first(flattened_data)
            for l_idx, m in enumerate(net.middle):
                o = m(o)
                if l_idx + 1 == layer_idx:
                    o = neural_model.Nonlinearity()(o)
                    out.append(o.cpu())
                    break
    
    out = torch.cat(out, dim=0)
    net.cpu()
    print(f"Layer {layer_idx} output shape: {out.shape}")
    print(f"Data consistency: Using exact training dataset order")
    return out


def build_subnetwork(net, dim, width, depth, num_classes,
                     layer_idx=0, random_net=False, act_name='relu'):

    net_ = neural_model.Net(dim, depth=depth - layer_idx,
                            width=width, num_classes=num_classes,
                            act_name=act_name)

    params = [p for idx, p in enumerate(net.parameters())]
    if not random_net:
        for idx, p_ in enumerate(net_.parameters()):
            p_.data = params[idx + layer_idx].data

    return net_


def get_jacobian(net, data):
    with torch.no_grad():
        return torch.vmap(torch.func.jacrev(net))(data).transpose(0, 2).transpose(0, 1)


def egop(net, dataset, centering=False):
    """
    Memory-optimized EGOP computation with reduced batch size and streaming processing
    """
    device = get_best_device()
    # Reduce batch size to save memory
    bs = 200  # Reduced from 1000 to 200
    batches = torch.split(dataset, bs)
    net = net.to(device)
    
    print(f"Processing {len(dataset)} samples in {len(batches)} batches of size {bs}")
    
    # First pass: compute mean for centering (if needed) without storing all Jacobians
    if centering:
        print("Computing mean for centering...")
        J_sum = None
        total_samples = 0
        
        for batch_idx, data in enumerate(batches):
            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f"Mean computation batch: {batch_idx}/{len(batches)}")
            
            data = data.to(device)
            J = get_jacobian(net, data)  # Shape: (output_dim, param_dim, batch_size)
            
            if J_sum is None:
                J_sum = J.sum(dim=-1).cpu()
            else:
                J_sum += J.sum(dim=-1).cpu()
            
            total_samples += J.shape[-1]
            del J, data
            torch.cuda.empty_cache() if device.type == 'cuda' else None
        
        J_mean = (J_sum / total_samples).unsqueeze(-1)
        del J_sum
    else:
        J_mean = None
    
    # Second pass: compute G matrix incrementally
    print("Computing G matrix...")
    G = None
    
    for batch_idx, data in enumerate(batches):
        if batch_idx % 10 == 0:  # Print every 10 batches
            print(f"G computation batch: {batch_idx}/{len(batches)}")
        
        data = data.to(device)
        J = get_jacobian(net, data)  # Shape: (output_dim, param_dim, batch_size)
        
        # Apply centering if needed
        if centering and J_mean is not None:
            J = J - J_mean.to(device)
        
        # Transpose for computation: (batch_size, output_dim, param_dim)
        J = J.permute(2, 0, 1)
        
        # Compute contribution to G: J^T @ J for this batch
        batch_G = torch.einsum('bop,boP->pP', J, J)
        
        if G is None:
            G = batch_G.cpu()
        else:
            G += batch_G.cpu()
        
        del J, batch_G, data
        torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    # Normalize by total number of samples
    G = G / len(dataset)
    
    return G


def correlate(M, G):
    M = M.double()
    G = G.double()
    normM = norm(M.flatten())
    normG = norm(G.flatten())

    corr = torch.dot(M.flatten(), G.flatten()) / (normM * normG)
    return corr


def read_configs(path):
    tokens = path.strip().split(':')
    print(tokens)
    act_name = 'relu'
    width = None
    depth = None
    for idx, t in enumerate(tokens):
        if t == 'width':
            width = eval(tokens[idx+1])
        if t == 'depth':
            depth = eval(tokens[idx+1])
        if t == 'act':
            act_name = tokens[idx+1]
    if width is None or depth is None:
        raise ValueError(f"width or depth not found in path: {path}")
    return width, depth, act_name


def verify_NFA(path, dataset_name, feature_idx=None, layer_idx=0, max_samples=None):
    """
    Verify Neural Feature Ansatz
    
    Args:
        max_samples: Maximum number of training samples to use (for memory management)
    """
    # Ensure same random seed as training for data consistency
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    
    remove_init = True
    random_net = False

    if dataset_name == 'celeba':
        NUM_CLASSES = 2
        FEATURE_IDX = feature_idx
        SIZE = 96
        c = 3
        dim = c * SIZE * SIZE
    elif dataset_name == 'svhn' or dataset_name == 'cifar':
        NUM_CLASSES = 10
        SIZE = 32
        c = 3
        dim = c * SIZE * SIZE
    elif dataset_name == 'cifar_mnist':
        NUM_CLASSES = 10
        c = 3
        SIZE = 32
        dim = c * SIZE * SIZE * 2
    elif dataset_name == 'stl_star':
        NUM_CLASSES = 2
        c = 3
        SIZE = 96
        dim = c * SIZE * SIZE

    width, depth, act_name = read_configs(path)

    net, M = load_nn(path, width, depth, dim, NUM_CLASSES, layer_idx=layer_idx,
                     remove_init=remove_init, act_name=act_name)
    net0, M0 = load_init_nn(path, width, depth, dim, NUM_CLASSES, layer_idx=layer_idx,
                            act_name=act_name)
    subnet = build_subnetwork(net, M.shape[0], width, depth, NUM_CLASSES, layer_idx=layer_idx,
                              random_net=random_net, act_name=act_name)

    init_correlation = correlate(torch.from_numpy(M),
                                 torch.from_numpy(M0))
 
    print("Init Net Feature Matrix Correlation: " , init_correlation)

    # Load the SAME dataset that was used for training
    # CRITICAL: Use identical random seed and dataset loading parameters
    print("Loading dataset with SAME random seed as training...")
    if dataset_name == 'celeba':
        trainloader, valloader, testloader = dataset.get_celeba(FEATURE_IDX,
                                                                num_train=20000,
                                                                num_test=1)
    elif dataset_name == 'svhn':
        # Use the FULL training set as in training (from main.py/trainer.py)
        trainloader, valloader, testloader = dataset.get_svhn()  # No num_train limit
    elif dataset_name == 'cifar':
        trainloader, valloader, testloader = dataset.get_cifar()  # No num_train limit

    elif dataset_name == 'cifar_mnist':
        trainloader, valloader, testloader = dataset.get_cifar_mnist(num_train_per_class=1000,
                                                                     num_test_per_class=1)
    elif dataset_name == 'stl_star':
        trainloader, valloader, testloader = dataset.get_stl_star(num_train=1000,
                                                                  num_test=1)
    
    # Get layer output with proper sample management
    # This now ensures we use the EXACT same samples as during training
    out = get_layer_output(net, trainloader, layer_idx=layer_idx, max_samples=max_samples)
    print(f"Final dataset shape for EGOP: {out.shape}")
    
    # Ensure we have 2D data for Jacobian computation
    if out.dim() == 1:
        raise ValueError(f"Output data is 1D but should be 2D. Shape: {out.shape}")
    
    G = egop(subnet, out, centering=True)
    G2 = egop(subnet, out, centering=False)

    centered_correlation = correlate(torch.from_numpy(M), G)
    uncentered_correlation = correlate(torch.from_numpy(M), G2)
    print("Full Matrix Correlation Centered: " , centered_correlation)
    print("Full Matrix Correlation Uncentered: " , uncentered_correlation)

    return init_correlation, centered_correlation, uncentered_correlation


import argparse

def main():
    parser = argparse.ArgumentParser(description='Verify Deep NFA')
    parser.add_argument('--path', type=str, required=True,
                        help='Path to saved neural net model')
    parser.add_argument('--dataset', type=str, default='svhn',
                        help='Dataset name (default: svhn)')
    parser.add_argument('--layers', type=int, nargs='+', default=[0,1,2,3,4],
                        help='Layer indices to compute EGOP (default: 0 1 2 3 4)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to use (default: None, use all training samples)')
    args = parser.parse_args()

    path = args.path
    dataset_name = args.dataset
    idxs = args.layers
    max_samples = args.max_samples
    
    if max_samples is not None:
        print(f"Using maximum {max_samples} samples for memory efficiency")
    else:
        print("Using ALL training samples (same as during training)")
    
    init, centered, uncentered = [], [], []
    for idx in idxs:
        print(f"\n=== Processing Layer {idx} ===")
        results = verify_NFA(path, dataset_name, layer_idx=idx, max_samples=max_samples)
        i, c, u = results
        init.append(i.numpy().item())
        centered.append(c.numpy().item())
        uncentered.append(u.numpy().item())
    
    print("\n=== Final Results ===")
    for idx in idxs:
        print("Layer " + str(idx), init[idx], centered[idx], uncentered[idx])


if __name__ == "__main__":
    main()
