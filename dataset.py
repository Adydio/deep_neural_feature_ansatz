import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
from tqdm import tqdm
from numpy.linalg import norm
import multiprocessing as mp


def get_optimal_num_workers():
    """
    Get optimal number of workers for DataLoader based on system capabilities.
    Returns min(cpu_count, 12) to avoid the warning about excessive worker creation.
    """
    cpu_count = mp.cpu_count()
    # PyTorch suggests max 12 workers to avoid slowness/freeze
    optimal_workers = min(cpu_count, 12)
    print(f"Using {optimal_workers} workers (CPU cores: {cpu_count})")
    return optimal_workers


def one_hot_data(dataset, num_classes, num_samples=-1, shift_label=False):
    labelset = {}
    for i in range(num_classes):
        one_hot = torch.zeros(num_classes)
        one_hot[i] = 1
        labelset[i] = one_hot

    offset = 0
    if shift_label:
        offset = -1

    subset = [(ex.flatten(), labelset[label + offset]) \
              for idx, (ex, label) in enumerate(dataset) if idx < num_samples]

    return subset


def split(trainset, p=.8):
    train, val = train_test_split(trainset, train_size=p)
    return train, val



def get_svhn(split_percentage=.8, num_train=float('inf'), num_test=float('inf')):
    NUM_CLASSES = 10
    transform = transforms.Compose([transforms.ToTensor()])
    svhn_path = '~/datasets/'

    trainset = torchvision.datasets.SVHN(root=svhn_path,
                                         split='train',
                                         transform=transform,
                                         download=True)
    trainset = one_hot_data(trainset, NUM_CLASSES, num_samples=num_train)
    trainset, valset = split(trainset, p=split_percentage)

    # Get optimal number of workers to avoid warnings
    num_workers = get_optimal_num_workers()

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=num_workers)

    valloader = torch.utils.data.DataLoader(valset, batch_size=128,
                                            shuffle=False, num_workers=num_workers)

    testset = torchvision.datasets.SVHN(root=svhn_path,
                                        split='test',
                                        transform=transform,
                                        download=True)

    testset = one_hot_data(testset, NUM_CLASSES, num_samples=num_test)

    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=num_workers)

    print("Num Train: ", len(trainset), "Num Val: ", len(valset),
          "Num Test: ", len(testset))
    return trainloader, valloader, testloader


def get_cifar(split_percentage=.8, num_train=float('inf'), num_test=float('inf')):

    NUM_CLASSES = 10
    transform = transforms.Compose([transforms.ToTensor()])
    path = '~/datasets/'

    trainset = torchvision.datasets.CIFAR10(root=path,
                                            train=True,
                                            transform=transform,
                                            download=True)

    trainset = one_hot_data(trainset, NUM_CLASSES, num_samples=num_train)
    trainset, valset = split(trainset, p=split_percentage)

    # Get optimal number of workers
    num_workers = get_optimal_num_workers()

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=num_workers)

    valloader = torch.utils.data.DataLoader(valset, batch_size=128,
                                            shuffle=False, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root=path,
                                           train=False,
                                           transform=transform,
                                           download=True)

    testset = one_hot_data(testset, NUM_CLASSES, num_samples=num_test)

    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=num_workers)

    print("Num Train: ", len(trainset), "Num Val: ", len(valset),
          "Num Test: ", len(testset))
    return trainloader, valloader, testloader


def sample_data(num, d):
    X = np.random.normal(size=(num, d))
    y = X[:, 0] * X[:, 1]
    y = y.reshape(-1, 1)
    return torch.from_numpy(X).float(), torch.from_numpy(y).float()


def get_two_coordinates(split_percentage=.8, num_train=2000, num_test=1000, d=100):
    X, y = sample_data(num_train, d)
    trainset = list(zip(X, y))
    trainset, valset = split(trainset, p=split_percentage)
    X_test, y_test = sample_data(num_test, d)
    testset = list(zip(X_test, y_test))

    # Get optimal number of workers
    num_workers = get_optimal_num_workers()

    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(valset, batch_size=128, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=num_workers)
    print("Num Train: ", len(trainset), "Num Val: ", len(valset),
          "Num Test: ", len(testset))

    return train_loader, val_loader, test_loader


def celeba_subset(dataset, feature_idx, num_samples=-1):

    NUM_CLASSES = 2
    labelset = {}
    for i in range(NUM_CLASSES):
        one_hot = torch.zeros(NUM_CLASSES)
        one_hot[i] = 1
        labelset[i] = one_hot

    by_class = {}
    features = []
    for idx in tqdm(range(len(dataset))):
        ex, label = dataset[idx]
        features.append(label[feature_idx])
        g = label[feature_idx].numpy().item()
        #ex = torch.mean(ex, dim=0)
        ex = ex.flatten()
        ex = ex / norm(ex)
        if g in by_class:
            by_class[g].append((ex, labelset[g]))
        else:
            by_class[g] = [(ex, labelset[g])]
        if idx > num_samples:
            break
    data = []
    if 1 in by_class:
        max_len = min(25000, len(by_class[1]))
        data.extend(by_class[1][:max_len])
        data.extend(by_class[0][:max_len])
    else:
        max_len = 1
        data.extend(by_class[0][:max_len])
    return data



def get_celeba(feature_idx, split_percentage=.8,
               num_train=float('inf'), num_test=float('inf')):
    celeba_path = '~/datasets/'
    SIZE = 96
    transform = transforms.Compose(
        [transforms.Resize([SIZE,SIZE]),
         transforms.ToTensor()
        ])

    trainset = torchvision.datasets.CelebA(root=celeba_path,
                                           split='train',
                                           transform=transform,
                                           download=True)
    trainset = celeba_subset(trainset, feature_idx, num_samples=num_train)
    trainset, valset = split(trainset, p=split_percentage)
    
    # Get optimal number of workers
    num_workers = get_optimal_num_workers()
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=num_workers)

    valloader = torch.utils.data.DataLoader(valset, batch_size=128,
                                            shuffle=False, num_workers=num_workers)

    testset = torchvision.datasets.CelebA(root=celeba_path,
                                              split='test',
                                              transform=transform,
                                              download=True)
    testset = celeba_subset(testset, feature_idx, num_samples=num_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=num_workers)

    print("Train Size: ", len(trainset), "Val Size: ", len(valset), "Test Size: ", len(testset))
    return trainloader, valloader, testloader


def group_by_class(dataset):
    labelset = {}
    for i in range(10):
        labelset[i] = []
    for i, batch in enumerate(dataset):
        img, label = batch
        labelset[label].append(img.view(1, 3, 32, 32))
    return labelset


def merge_data(cifar, mnist, n):
    cifar_by_label = group_by_class(cifar)
    mnist_by_label = group_by_class(mnist)

    merged_data = []
    merged_labels = []

    labelset = {}

    for i in range(10):
        one_hot = torch.zeros(1, 10)
        one_hot[0, i] = 1
        labelset[i] = one_hot

    for l in cifar_by_label:

        cifar_data = torch.cat(cifar_by_label[l])
        mnist_data = torch.cat(mnist_by_label[l])
        min_len = min(len(mnist_data), len(cifar_data))
        m = min(n, min_len)
        cifar_data = cifar_data[:m]
        mnist_data = mnist_data[:m]

        merged = torch.cat([cifar_data, mnist_data], axis=-1)
        #for i in range(3):
        #    vis.image(merged[i])
        merged_data.append(merged.reshape(m, -1))
        #print(merged.shape)
        merged_labels.append(np.repeat(labelset[l], m, axis=0))
    merged_data = torch.cat(merged_data, axis=0)

    merged_labels = np.concatenate(merged_labels, axis=0)
    merged_labels = torch.from_numpy(merged_labels)

    return list(zip(merged_data, merged_labels))


def get_cifar_mnist(split_percentage=.8, num_train_per_class=float('inf'),
                    num_test_per_class=float('inf')):

        path = '~/datasets/'
        transform = transforms.Compose(
            [#transforms.Resize([32,32]),
                transforms.ToTensor()
            ])

        mnist_transform = transforms.Compose(
            [transforms.Resize([32,32]),
             transforms.ToTensor(),
             transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            ])


        cifar_trainset = torchvision.datasets.CIFAR10(root=path,
                                                      train=True,
                                                      transform=transform,
                                                      download=True)

        mnist_trainset = torchvision.datasets.MNIST(root=path,
                                                    train=True,
                                                    transform=mnist_transform,
                                                    download=True)
        trainset = merge_data(cifar_trainset, mnist_trainset, num_train_per_class)
        trainset, valset = split(trainset, p=split_percentage)
        
        # Get optimal number of workers
        num_workers = get_optimal_num_workers()
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                                  shuffle=True, num_workers=num_workers)
        valloader = torch.utils.data.DataLoader(valset, batch_size=128,
                                                shuffle=False, num_workers=num_workers)

        cifar_testset = torchvision.datasets.CIFAR10(root=path,
                                                     train=False,
                                                     transform=transform,
                                                     download=True)

        mnist_testset = torchvision.datasets.MNIST(root=path,
                                                   train=False,
                                                   transform=mnist_transform,
                                                   download=True)

        testset = merge_data(cifar_testset, mnist_testset, num_test_per_class)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                                 shuffle=False, num_workers=num_workers)

        print("Num Train: ", len(trainset), "Num Val: ", len(valset),
              "Num Test: ", len(testset))
        return trainloader, valloader, testloader


def draw_star(ex, v, c=3):
    ex[:c, 5:6, 7:14] = v
    ex[:c, 4, 9:12] = v
    ex[:c, 3, 10] = v
    ex[:c, 6, 8:13] = v
    ex[:c, 7, 9:12] = v
    ex[:c, 8, 8:13] = v
    ex[:c, 9, 8:10] = v
    ex[:c, 9, 11:13] = v
    return ex


def one_hot_stl_toy(dataset, num_samples=-1):
    labelset = {}
    for i in range(2):
        one_hot = torch.zeros(2)
        one_hot[i] = 1
        labelset[i] = one_hot

    subset = [(ex, label) for idx, (ex, label) in enumerate(dataset) \
              if idx < num_samples and (label == 0 or label == 9)]

    adjusted = []
    for idx, (ex, label) in enumerate(subset):
        if label == 9:
            ex = draw_star(ex,1, c=2)
            y = 1
        else:
            ex = draw_star(ex, 0)
            y = 0
        ex = ex.flatten()
        adjusted.append((ex, labelset[y]))
    return adjusted



def get_stl_star(split_percentage=.8, num_train=float('inf'),
                 num_test=float('inf')):
    SIZE = 96
    transform = transforms.Compose(
        [transforms.Resize([SIZE, SIZE]),
         transforms.ToTensor()
        ])

    path = '~/datasets/'
    trainset = torchvision.datasets.STL10(root=path,
                                          split='train',
                                          #train=True,
                                          transform=transform,
                                          download=True)
    trainset = one_hot_stl_toy(trainset, num_samples=num_train)
    trainset, valset = split(trainset, p=split_percentage)
    
    # Get optimal number of workers
    num_workers = get_optimal_num_workers()
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=num_workers)

    valloader = torch.utils.data.DataLoader(valset, batch_size=128,
                                            shuffle=False, num_workers=num_workers)
    testset = torchvision.datasets.STL10(root=path,
                                         split='test',
                                         transform=transform,
                                         download=True)
    testset = one_hot_stl_toy(testset, num_samples=num_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=num_workers)
    print("Num Train: ", len(trainset), "Num Val: ", len(valset),
          "Num Test: ", len(testset))
    return trainloader, valloader, testloader

# =============================================================================
# OpenML Tabular Benchmark Dataset Support
# Following the standard tabular benchmark from https://github.com/LeoGrin/tabular-benchmark
# =============================================================================

import openml

def tabular_to_one_hot_data(X, y, num_classes):
    """Convert tabular data to one-hot format compatible with existing pipeline"""
    # Create one-hot labels
    labelset = {}
    for i in range(num_classes):
        one_hot = torch.zeros(num_classes)
        one_hot[i] = 1
        labelset[i] = one_hot
    
    # Convert to list of (features, one_hot_label) tuples
    dataset = []
    for i in range(len(X)):
        features = torch.from_numpy(X[i]).float()
        label = labelset[y[i]]
        dataset.append((features, label))
    
    return dataset


def load_openml_dataset(dataset_id, split_percentage=0.8, normalize=True, max_samples=None):
    """
    Load OpenML dataset by ID
    """
    print(f"Loading OpenML dataset {dataset_id}...")
    
    # Load dataset from OpenML
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    
    # Handle categorical features (encode as numerical)
    if categorical_indicator is not None and any(categorical_indicator):
        # Encode categorical features
        for i, is_cat in enumerate(categorical_indicator):
            if is_cat and i < X.shape[1]:  # Skip target column
                le = LabelEncoder()
                X.iloc[:, i] = le.fit_transform(X.iloc[:, i].astype(str))
    
    # Convert to numpy
    X = X.values.astype(np.float32)
    
    # Handle categorical target (encode as numerical)
    if y.dtype == 'object' or y.dtype.name == 'category':
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        y = y.values.astype(int)
    
    # Limit dataset size if specified
    if max_samples is not None and len(X) > max_samples:
        indices = np.random.choice(len(X), max_samples, replace=False)
        X = X[indices]
        y = y[indices]
    
    # Get number of classes
    num_classes = len(np.unique(y))
    
    # Normalize features
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Further split train into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=(1-split_percentage)/0.8, 
        random_state=42, stratify=y_train
    )
    
    # Convert to one-hot format
    trainset = tabular_to_one_hot_data(X_train, y_train, num_classes)
    valset = tabular_to_one_hot_data(X_val, y_val, num_classes)
    testset = tabular_to_one_hot_data(X_test, y_test, num_classes)
    
    # Get optimal number of workers
    num_workers = get_optimal_num_workers()
    
    # Create data loaders
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=num_workers)
    valloader = DataLoader(valset, batch_size=128, shuffle=False, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=num_workers)
    
    print(f"OpenML Dataset {dataset_id} ({dataset.name}) - Train: {len(trainset)}, Val: {len(valset)}, Test: {len(testset)}")
    print(f"Features: {X.shape[1]}, Classes: {num_classes}")
    
    return trainloader, valloader, testloader


# Standard OpenML Tabular Benchmark Datasets
# From Suite 337 (Classification with numerical features only)

def get_credit(split_percentage=0.8, normalize=True):
    """OpenML Dataset 1590: credit (Australian Credit Approval)"""
    return load_openml_dataset(1590, split_percentage, normalize, max_samples=20000)


def get_electricity(split_percentage=0.8, normalize=True):
    """OpenML Dataset 151: electricity (Electricity pricing)"""
    return load_openml_dataset(151, split_percentage, normalize, max_samples=50000)


def get_covertype(split_percentage=0.8, normalize=True):
    """OpenML Dataset 1596: covertype (Forest Cover Type - sampled for memory)"""
    return load_openml_dataset(1596, split_percentage, normalize, max_samples=50000)


def get_pol(split_percentage=0.8, normalize=True):
    """OpenML Dataset 722: pol (Political voting)"""
    return load_openml_dataset(722, split_percentage, normalize)


def get_house_16H(split_percentage=0.8, normalize=True):
    """OpenML Dataset 821: house_16H (House prices)"""
    return load_openml_dataset(821, split_percentage, normalize)


def get_MagicTelescope(split_percentage=0.8, normalize=True):
    """OpenML Dataset 1120: MagicTelescope (MAGIC Gamma Telescope)"""
    return load_openml_dataset(1120, split_percentage, normalize)


# Additional classic OpenML datasets for variety

def get_credit_g(split_percentage=0.8, normalize=True):
    """OpenML Dataset 31: credit-g (German Credit)"""
    return load_openml_dataset(31, split_percentage, normalize)


def get_kr_vs_kp(split_percentage=0.8, normalize=True):
    """OpenML Dataset 3: kr-vs-kp (Chess King-Rook vs King-Pawn)"""
    return load_openml_dataset(3, split_percentage, normalize)
