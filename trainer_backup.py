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

@torch.compile
def zeropower_via_newtonschulz5(G, steps=3, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
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
        params = [p for p in net.parameters() if p.requires_grad]
        return Muon(params, lr=lr, momentum=0.6, nesterov=True)
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

    d = {}
    d['state_dict'] = net.state_dict()
    if name is not None:
        torch.save(d, 'saved_nns/init_' + name + '.pth')


    device = get_best_device()
    net = net.to(device)
    best_val_acc = 0
    best_test_acc = 0
    best_val_loss = float("inf")
    best_test_loss = 0

    val_interval = configs.get('val_interval', 20)
    for i in range(num_epochs):
        train_loss = train_step(net, optimizer, train_loader)
        if (i % val_interval == 0) or (i == num_epochs - 1):
            val_loss = val_step(net, val_loader)
            test_loss = val_step(net, test_loader)
            if regression:
                train_acc = get_r2(net, train_loader)
                val_acc = get_r2(net, val_loader)
                test_acc = get_r2(net, test_loader)
            else:
                train_acc = get_acc(net, train_loader)
                val_acc = get_acc(net, val_loader)
                test_acc = get_acc(net, test_loader)

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                net.cpu()
                d = {}
                d['state_dict'] = net.state_dict()
                if name is not None:
                    torch.save(d, 'saved_nns/' + name + '.pth')
                device = torch.device('cpu')
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
    d['state_dict'] = net.state_dict()
    torch.save(d, 'saved_nns/' + name + '_final.pth')
    return best_val_acc, best_test_acc

def train_step(net, optimizer, train_loader):
    net.train()
    start = time.time()
    train_loss = 0.
    num_batches = len(train_loader)

    device = get_best_device()
    net = net.to(device)
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, labels = batch
        targets = labels
        inputs = inputs.to(device)
        targets = targets.to(device)
        output = net(Variable(inputs))
        target = Variable(targets)
        loss = torch.mean(torch.pow(output - target, 2))
        loss.backward()
        optimizer.step()
        train_loss += loss.cpu().data.numpy() * len(inputs)
    end = time.time()
    print("Time: ", end - start)
    train_loss = train_loss / len(train_loader.dataset)
    return train_loss


def val_step(net, val_loader):
    net.eval()
    val_loss = 0.
    device = get_best_device()
    net = net.to(device)
    for batch_idx, batch in enumerate(val_loader):
        inputs, labels = batch
        targets = labels
        inputs = inputs.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            output = net(Variable(inputs))
            target = Variable(targets)
        loss = torch.mean(torch.pow(output - target, 2))
        val_loss += loss.cpu().data.numpy() * len(inputs)
    val_loss = val_loss / len(val_loader.dataset)
    return val_loss


def get_acc(net, loader):
    net.eval()
    count = 0
    device = get_best_device()
    net = net.to(device)
    for batch_idx, batch in enumerate(loader):
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            output = net(Variable(inputs))
            target = Variable(targets)

        preds = torch.argmax(output, dim=-1)
        labels = torch.argmax(target, dim=-1)

        count += torch.sum(labels == preds).cpu().data.numpy()
    return count / len(loader.dataset) * 100


def get_r2(net, loader):
    net.eval()
    count = 0
    preds = []
    labels = []
    device = get_best_device()
    net = net.to(device)
    for batch_idx, batch in enumerate(loader):
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            output = net(Variable(inputs)).flatten().cpu().numpy()
            target = Variable(targets).flatten().cpu().numpy()
            preds.append(output)
            labels.append(target)
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    return r2_score(labels, preds)
