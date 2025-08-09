def get_best_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import time
import neural_model
import numpy as np
from sklearn.metrics import r2_score
import os


def clean_compiled_state_dict(state_dict):
    """
    Remove '_orig_mod.' prefix from compiled model state dict keys.
    This ensures compatibility when loading models saved after torch.compile().
    """
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            cleaned_key = key[len('_orig_mod.'):]
            cleaned_state_dict[cleaned_key] = value
        else:
            cleaned_state_dict[key] = value
    return cleaned_state_dict


def get_clean_state_dict(net):
    """
    Get a clean state dict from the model, removing torch.compile prefixes if present.
    """
    state_dict = net.state_dict()
    return clean_compiled_state_dict(state_dict)

@torch.compile
def zeropower_via_newtonschulz5(G, steps=3, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                if 'momentum_buffer' not in state.keys():
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group['nesterov'] else buf

                p.data.mul_(len(p.data)**0.5 / p.data.norm()) # normalize the weight
                update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape) # whiten the update
                p.data.add_(update, alpha=-lr) # take a step


def select_optimizer(name, lr, net, weight_decay):
    if name == 'sgd':
        return torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == 'adam':
        return torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == 'muon':
        return Muon(list(net.parameters()), lr=lr, momentum=0.6, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def train_network(train_loader, val_loader, test_loader, num_classes,
                  name='default_nn', configs=None, regression=False):

    for idx, batch in enumerate(train_loader):
        inputs, labels = batch
        _, dim = inputs.shape
        break

    if configs is not None:
        num_epochs = configs['num_epochs'] + 1
        net = neural_model.Net(dim, width=configs['width'],
                               depth=configs['depth'],
                               num_classes=num_classes,
                               act_name=configs['act'])

        if configs['init'] != 'default':
            for idx, param in enumerate(net.parameters()):
                if idx == 0:
                    init = torch.Tensor(param.size()).normal_().float() * configs['init']
                    param.data = init

        if configs['freeze']:
            for idx, param in enumerate(net.parameters()):
                if idx > 0:
                    param.requires_grad = False

        optimizer = select_optimizer(configs['optimizer'],
                                     configs['learning_rate'],
                                     net,
                                     configs['weight_decay'])
    else:
        num_epochs = 501
        net = neural_model.Net(dim, num_classes=num_classes)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    # Get device and move model
    device = get_best_device()
    net = net.to(device)
    
    # For muon optimization, enable torch.compile
    if configs and configs['optimizer'] == 'muon':
        net = torch.compile(net, mode='reduce-overhead')
    
    # Use MSE Loss (L2 loss)
    criterion = nn.MSELoss()

    # Get save directory from configs or use default
    save_dir = configs.get('exp_dir', 'saved_nns') if configs else 'saved_nns'
    save_dir = os.path.join(save_dir, 'models') if configs and 'exp_dir' in configs else save_dir
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    d = {}
    d['state_dict'] = get_clean_state_dict(net)
    if name is not None:
        torch.save(d, os.path.join(save_dir, 'init_' + name + '.pth'))

    best_val_acc = 0
    best_test_acc = 0
    best_val_loss = float("inf")
    best_test_loss = 0

    val_interval = configs.get('val_interval', 20) if configs else 20
    
    # Use AMP for performance
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    for i in range(num_epochs):
        train_loss = train_step(net, optimizer, train_loader, criterion, device, scaler)
        
        if (i % val_interval == 0) or (i == num_epochs - 1):
            val_loss = val_step(net, val_loader, criterion, device)
            test_loss = val_step(net, test_loader, criterion, device)
            
            if regression:
                train_acc = get_r2(net, train_loader, device)
                val_acc = get_r2(net, val_loader, device)
                test_acc = get_r2(net, test_loader, device)
            else:
                train_acc = get_acc(net, train_loader, device)
                val_acc = get_acc(net, val_loader, device)
                test_acc = get_acc(net, test_loader, device)

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                net.cpu()
                d = {}
                d['state_dict'] = get_clean_state_dict(net)
                if name is not None:
                    torch.save(d, os.path.join(save_dir, name + '.pth'))
                net.to(device)

            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss

            print("Epoch: ", i,
                  "Train Loss: ", train_loss, "Test Loss: ", test_loss,
                  "Train Acc: ", train_acc, "Test Acc: ", test_acc,
                  "Best Val Acc: ", best_val_acc, "Best Val Loss: ", best_val_loss,
                  "Best Test Acc: ", best_test_acc, "Best Test Loss: ", best_test_loss)
        else:
            print(f"Epoch: {i} Train Loss: {train_loss}")

    net.cpu()
    d = {}
    d['state_dict'] = get_clean_state_dict(net)
    torch.save(d, os.path.join(save_dir, name + '_final.pth'))
    return best_val_acc, best_test_acc


def train_step(net, optimizer, train_loader, criterion, device, scaler=None):
    net.train()
    start = time.time()
    train_loss = 0.
    
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        if scaler and device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                output = net(inputs)
                loss = criterion(output, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = net(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        net.zero_grad(set_to_none=True)
        train_loss += loss.item() * len(inputs)
        
    end = time.time()
    print("Time: ", end - start)
    train_loss = train_loss / len(train_loader.dataset)
    return train_loss


def val_step(net, val_loader, criterion, device):
    net.eval()
    val_loss = 0.
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            output = net(inputs)
            loss = criterion(output, labels)
            val_loss += loss.item() * len(inputs)
            
    val_loss = val_loss / len(val_loader.dataset)
    return val_loss


def get_acc(net, loader, device):
    net.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = net(inputs)
            
            # Handle both one-hot and index labels
            if labels.dim() > 1 and labels.size(1) > 1:  # one-hot encoded
                predicted = torch.argmax(outputs, dim=1)
                labels = torch.argmax(labels, dim=1)
            else:  # index labels
                predicted = torch.argmax(outputs, dim=1)
                
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return 100 * correct / total


def get_r2(net, loader, device):
    net.eval()
    preds = []
    labels_list = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            output = net(inputs).flatten().cpu().numpy()
            target = targets.flatten().cpu().numpy()
            preds.append(output)
            labels_list.append(target)
            
    preds = np.concatenate(preds)
    labels_list = np.concatenate(labels_list)
    return r2_score(labels_list, preds)
