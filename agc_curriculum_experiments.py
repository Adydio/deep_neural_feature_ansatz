# -*- coding: utf-8 -*-
"""
AGOP-aware Curriculum: multi-task, multi-seed experiments with plots.
Add AGC-InvCFP (counterfactual-pair curriculum with consistency).
Saves CSV/PNG under: experiments/curriculum/<task>/<timestamp>/
"""
import os, math, random, argparse, time, json, copy
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# -----------------------
# Utils & Repro
# -----------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# -----------------------
# Spurious-color datasets
# -----------------------
class ColoredBinaryBase(Dataset):
    """
    Base wrapper: takes a base dataset (PIL image, raw_label), maps to binary y and spurious color c,
    where P[c==y] = p_corr. Colors are cached at __init__ for reproducibility.
    Provides counterfactual flip for each index.
    """
    def __init__(self, base_ds, make_binary_label, split: str, p_train=0.99, p_test=0.1, seed=0):
        assert split in ["train","test"]
        self.base = base_ds
        self.split = split
        self.p_corr = p_train if split=="train" else p_test
        rng = np.random.RandomState(seed if seed is not None else 0)

        self.labels = []
        self.colors = []
        N = len(self.base)
        for i in range(N):
            _, raw_y = self.base[i]
            y = int(make_binary_label(raw_y))
            corr = rng.rand() < self.p_corr
            c = y if corr else 1 - y
            self.labels.append(y)
            self.colors.append(c)
        self.labels = np.array(self.labels, dtype=np.int64)
        self.colors = np.array(self.colors, dtype=np.int64)

    def __len__(self): return len(self.base)

    def _pil_to_gray_np(self, pil_img) -> np.ndarray:
        arr = np.array(pil_img, dtype=np.float32)/255.0  # HxW or HxWx3
        if arr.ndim == 3 and arr.shape[2] == 3:
            # luminance
            arr = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]
        return arr  # HxW float

    def _colorize_gray(self, gray: np.ndarray, c: int) -> torch.Tensor:
        R = gray if c==0 else np.zeros_like(gray)
        G = gray if c==1 else np.zeros_like(gray)
        B = np.zeros_like(gray)
        arr = np.stack([R, G, B], axis=0)  # 3xHxW
        return torch.from_numpy(arr.astype(np.float32))

    def get_flip_tensor(self, idx: int) -> torch.Tensor:
        """Return counterfactual color-flip tensor x_flip for sample idx (same content, color flipped)."""
        pil_img, _ = self.base[idx]
        gray = self._pil_to_gray_np(pil_img)
        c_flip = 1 - int(self.colors[idx])
        return self._colorize_gray(gray, c_flip)

    def __getitem__(self, idx):
        pil_img, _ = self.base[idx]
        gray = self._pil_to_gray_np(pil_img)
        y = int(self.labels[idx]); c = int(self.colors[idx])
        x = self._colorize_gray(gray, c)
        return x, y, c, idx

class ColoredMNIST(ColoredBinaryBase):
    def __init__(self, root, split="train", p_train=0.99, p_test=0.1, seed=0, download=True):
        base = datasets.MNIST(root=root, train=(split=="train"), download=download)
        super().__init__(base, make_binary_label=lambda d: (d<5), split=split,
                         p_train=p_train, p_test=p_test, seed=seed)

class ColoredFashionMNIST(ColoredBinaryBase):
    def __init__(self, root, split="train", p_train=0.99, p_test=0.1, seed=0, download=True):
        base = datasets.FashionMNIST(root=root, train=(split=="train"), download=download)
        super().__init__(base, make_binary_label=lambda d: (d<5), split=split,
                         p_train=p_train, p_test=p_test, seed=seed)

class ColoredCIFAR10(ColoredBinaryBase):
    ANIMALS = {2,3,4,5,6,7}
    VEHICLES = {0,1,8,9}
    def __init__(self, root, split="train", p_train=0.995, p_test=0.1, seed=0, download=True):
        base = datasets.CIFAR10(root=root, train=(split=="train"), download=download)
        def make_binary_label(raw_y: int) -> int:
            return 1 if raw_y in self.ANIMALS else 0
        super().__init__(base, make_binary_label=make_binary_label, split=split,
                         p_train=p_train, p_test=p_test, seed=seed)

# -----------------------
# Models
# -----------------------
class CNNFeatSmall(nn.Module):
    """For 32x32 or 28x28, light CNN with feature head."""
    def __init__(self, d_feat=256, num_classes=2):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14 or 16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 7 or 8
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4,4)),
            nn.Flatten(),
            nn.Linear(128*4*4, 256), nn.ReLU(inplace=True),
            nn.Linear(256, d_feat), nn.ReLU(inplace=True)  # 多一层稳定特征
        )
        self.classifier = nn.Linear(d_feat, num_classes)

    def forward(self, x, return_feat=False):
        z = self.feat(x)
        logits = self.classifier(z)
        if return_feat:
            return logits, z
        return logits

# -----------------------
# AGOP utilities
# -----------------------
@torch.no_grad()
def topk_basis_from_classifier(model: nn.Module, k_desired=2, eps=1e-8) -> torch.Tensor:
    """
    Use W^T W to approximate AGOP subspace (NFA). Auto-select effective rank (<= num_classes).
    Returns U : [d, k]
    """
    W = model.classifier.weight.detach()
    M = W.T @ W
    evals, evecs = torch.linalg.eigh(M)
    if float(evals.max()) <= 0:
        return evecs[:, -1:].contiguous()
    mask = evals > (eps * float(evals.max()))
    r = int(mask.sum().item())
    k = max(1, min(k_desired, r))
    U = evecs[:, -k:].contiguous()
    return U

@torch.no_grad()
def spectral_entropy(model: nn.Module) -> float:
    W = model.classifier.weight.detach()
    M = W.T @ W
    evals = torch.linalg.eigvalsh(M).clamp(min=1e-12)
    p = (evals / evals.sum()).cpu().numpy()
    return float(-(p * np.log(p)).sum())

@torch.no_grad()
def compute_align_scores(model: nn.Module, loader: DataLoader, U: torch.Tensor) -> Dict[int, float]:
    """ s(i) = 1 - ||U^T z||^2 / ||z||^2 """
    model.eval()
    scores = {}
    U = U.to(DEVICE)
    for x, y, c, idx in loader:
        x = x.to(DEVICE)
        _, z = model(x, return_feat=True)
        z = F.normalize(z, dim=1)
        s = (1 - ((z @ U)**2).sum(dim=1)).clamp(min=0.0, max=1.0).detach().cpu().numpy()
        for i, sid in enumerate(idx.tolist()):
            scores[sid] = float(s[i])
    return scores

# NEW: counterfactual-pair sensitivity Δ_U
@torch.no_grad()
def compute_pair_sensitivity(model: nn.Module, loader: DataLoader, dataset: Dataset,
                             U: torch.Tensor, probe_frac: float = 1.0) -> Dict[int, float]:
    """
    For each idx in (a probed subset of) loader, compute Δ_U(i) = ||U^T z - U^T z_flip||^2 / (||z||^2 + ||z_flip||^2)
    Return dict idx -> Δ_U (higher means more spurious-sensitive).
    """
    model.eval()
    U = U.to(DEVICE)
    scores = {}
    rng = np.random.RandomState(0)
    for x, y, c, idx in loader:
        idx_list = idx.tolist()
        if probe_frac < 1.0:
            take = rng.rand(len(idx_list)) < probe_frac
            idx_list = [idx_list[i] for i,b in enumerate(take) if b]
            if len(idx_list) == 0:
                continue
            # reselect the tensors accordingly
            mask = torch.tensor([i in set(idx_list) for i in idx.tolist()], dtype=torch.bool)
            x = x[mask]; y = y[mask]; c = c[mask]; idx = idx[mask]

        x = x.to(DEVICE)
        # build flip batch
        x_flip = torch.stack([dataset.get_flip_tensor(int(i)) for i in idx_list], dim=0).to(DEVICE)

        _, z = model(x, return_feat=True)
        _, zf = model(x_flip, return_feat=True)
        z  = F.normalize(z, dim=1)
        zf = F.normalize(zf, dim=1)
        p  = z @ U
        pf = zf @ U
        num = ((p - pf)**2).sum(dim=1)           # energy change on U
        den = (z.pow(2).sum(dim=1) + zf.pow(2).sum(dim=1) + 1e-8)
        delta = (num / den).detach().cpu().numpy()
        for i, sid in enumerate(idx_list):
            scores[sid] = float(delta[i])
    return scores

# FIX: device-safe direction alignment
@torch.no_grad()
def direction_alignment(model: nn.Module, loader: DataLoader) -> Tuple[float,float]:
    """
    alpha: alignment to label direction; beta: alignment to color direction.
    Compute on CPU to avoid device mismatch.
    """
    model.eval()
    Z, Ys, Cs = [], [], []
    for x, y, c, _ in loader:
        x = x.to(DEVICE)
        _, z = model(x, return_feat=True)
        Z.append(z.detach().cpu())
        Ys.append(y.detach().cpu())
        Cs.append(c.detach().cpu())
    Z = torch.cat(Z).to(torch.float32)
    Ys = torch.cat(Ys); Cs = torch.cat(Cs)
    if (Ys==0).sum()==0 or (Ys==1).sum()==0 or (Cs==0).sum()==0 or (Cs==1).sum()==0:
        return 0.0, 0.0
    z0 = Z[Ys==0].mean(0); z1 = Z[Ys==1].mean(0)
    u_y = F.normalize(z1 - z0, dim=0)
    zc0 = Z[Cs==0].mean(0); zc1 = Z[Cs==1].mean(0)
    u_c = F.normalize(zc1 - zc0, dim=0)
    W = model.classifier.weight.detach().cpu().to(torch.float32)
    M = W.T @ W
    evals, evecs = torch.linalg.eigh(M)
    v1 = F.normalize(evecs[:, -1], dim=0)
    alpha = float(torch.abs(torch.dot(v1, u_y)))
    beta  = float(torch.abs(torch.dot(v1, u_c)))
    if not np.isfinite(alpha): alpha = 0.0
    if not np.isfinite(beta):  beta  = 0.0
    return alpha, beta

# -----------------------
# Train / Eval
# -----------------------
def run_epoch(model, loader, optimizer, criterion, train=True):
    if train: model.train()
    else: model.eval()
    total, correct, total_loss = 0, 0, 0.0
    for x, y, c, idx in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        pred = logits.argmax(1)
        total += y.size(0)
        correct += (pred==y).sum().item()
        total_loss += float(loss.item()) * y.size(0)
    return total_loss/total, correct/total

@torch.no_grad()
def eval_worst_group(model, loader):
    model.eval()
    stats = {(y,c): [0,0] for y in [0,1] for c in [0,1]}
    for x, y, c, idx in loader:
        x = x.to(DEVICE)
        pred = model(x).argmax(1).cpu()
        for i in range(len(y)):
            key = (int(y[i]), int(c[i]))
            stats[key][0] += int(pred[i]==y[i])
            stats[key][1] += 1
    accs = {k: (v[0]/v[1] if v[1]>0 else 0.0) for k,v in stats.items()}
    worst = min(accs.values())
    avg = sum(accs.values())/len(accs)
    return worst, avg, accs

def train_erm(model, train_loader, test_loader, epochs=30, lr=3e-4, wd=1e-4):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()
    logs=[]
    for ep in range(1, epochs+1):
        tr_loss, tr_acc = run_epoch(model, train_loader, opt, crit, train=True)
        te_loss, te_acc = run_epoch(model, test_loader, None, crit, train=False)
        wg, _, _ = eval_worst_group(model, test_loader)
        H = spectral_entropy(model)
        a, b = direction_alignment(model, train_loader)
        logs.append({"epoch": ep, "tr_acc": tr_acc, "te_acc": te_acc, "worst_group_acc": wg, "H": H, "alpha": a, "beta": b})
        print(f"[ERM] ep{ep:02d} te_acc={te_acc:.3f} worstG={wg:.3f} H={H:.3f} alpha={a:.3f} beta={b:.3f}")
    return model, logs

def train_jtt(train_set, test_loader, total_epochs=30, stage1_epochs=5, upsample=10, lr=3e-4):
    base = CNNFeatSmall().to(DEVICE)
    opt = torch.optim.AdamW(base.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    base_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
    logs=[]
    for ep in range(1, stage1_epochs+1):
        run_epoch(base, base_loader, opt, crit, train=True)
        te_loss, te_acc = run_epoch(base, test_loader, None, crit, train=False)
        wg, _, _ = eval_worst_group(base, test_loader)
        H = spectral_entropy(base); a,b = direction_alignment(base, base_loader)
        logs.append({"epoch": ep, "te_acc": te_acc, "worst_group_acc": wg, "H": H, "alpha": a, "beta": b})
        print(f"[JTT-Stage1] ep{ep:02d} te_acc={te_acc:.3f} worstG={wg:.3f}")

    base.eval()
    mis_idx=[]
    with torch.no_grad():
        for x, y, c, idx in DataLoader(train_set, batch_size=256, shuffle=False, num_workers=2, pin_memory=True):
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = base(x).argmax(1)
            mis = (pred!=y).cpu().numpy()
            mis_idx += list(np.array(idx)[mis])

    weights = torch.ones(len(train_set), dtype=torch.double)
    if len(mis_idx) > 0:
        weights[torch.tensor(mis_idx, dtype=torch.long)] *= float(upsample)
    sampler = WeightedRandomSampler(weights, num_samples=len(train_set), replacement=True)
    train_loader = DataLoader(train_set, batch_size=256, sampler=sampler, num_workers=2, pin_memory=True)

    model = CNNFeatSmall().to(DEVICE)
    opt2 = torch.optim.AdamW(model.parameters(), lr=lr)
    for ep in range(stage1_epochs+1, total_epochs+1):
        tr_loss, tr_acc = run_epoch(model, train_loader, opt2, crit, train=True)
        te_loss, te_acc = run_epoch(model, test_loader, None, crit, train=False)
        wg, _, _ = eval_worst_group(model, test_loader)
        H = spectral_entropy(model); a,b = direction_alignment(model, DataLoader(train_set, batch_size=256, shuffle=False))
        logs.append({"epoch": ep, "tr_acc": tr_acc, "te_acc": te_acc, "worst_group_acc": wg, "H": H, "alpha": a, "beta": b})
        print(f"[JTT] ep{ep:02d} te_acc={te_acc:.3f} worstG={wg:.3f} H={H:.3f}")
    return model, logs

def _unwrap_colored_dataset(ds: Dataset) -> ColoredBinaryBase:
    base = ds
    while isinstance(base, Subset):
        base = base.dataset
    assert isinstance(base, ColoredBinaryBase), "This method needs Colored* dataset."
    return base

# NEW: AGC-InvCFP training
def train_agc_invcfp(train_set, test_loader, total_epochs=30, lr=3e-4,
                     keep_start=0.3, keep_end=0.9, k_desired=2,
                     probe_frac=0.6, lambda_cons=0.2):
    """
    Counterfactual-pair curriculum:
      - score by Δ_U = ||U^T z - U^T z_flip||^2 / (||z||^2 + ||z_flip||^2)
      - select top keep_ratio to form a subset each epoch
      - train with CE on (x) and (x_flip) + consistency loss on features
    """
    dataset_base = _unwrap_colored_dataset(train_set)
    model = CNNFeatSmall().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    base_loader = DataLoader(train_set, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    # labels dict for mild class-balance
    labels = {}
    for _, y, _, idx in base_loader:
        for i, sid in enumerate(idx.tolist()):
            labels[sid] = int(y[i])

    logs=[]
    for ep in range(1, total_epochs+1):
        keep_ratio = min(keep_end, keep_start + (keep_end - keep_start) * (ep-1)/(total_epochs-1))

        # AGOP subspace
        U = topk_basis_from_classifier(model, k_desired=k_desired)

        # Δ_U sensitivity
        delta = compute_pair_sensitivity(model, base_loader, dataset_base, U, probe_frac=probe_frac)
        n = len(delta); m = max(1, int(n * keep_ratio))
        # select top-m by Δ_U
        top_idx = [i for i,_ in sorted(delta.items(), key=lambda kv: kv[1], reverse=True)[:m]]

        # mix in small random for stability
        remain = list(set(delta.keys()) - set(top_idx))
        if len(remain) > 0:
            top_idx += random.sample(remain, min(int(0.1*n), len(remain)))

        subset = Subset(train_set, top_idx)
        train_loader = DataLoader(subset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)

        # one epoch with pair-consistency
        model.train()
        total, correct, total_loss = 0, 0, 0.0
        for x, y, c, idx in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x_flip = torch.stack([dataset_base.get_flip_tensor(int(i)) for i in idx.tolist()], dim=0).to(DEVICE)

            logits, z   = model(x, return_feat=True)
            logits_f, zf = model(x_flip, return_feat=True)

            ce = crit(logits, y) + crit(logits_f, y)
            cons = F.mse_loss(z, zf)
            loss = ce + lambda_cons * cons

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            pred = logits.argmax(1)
            total += y.size(0)
            correct += (pred==y).sum().item()
            total_loss += float(loss.item()) * y.size(0)

        # eval
        te_loss, te_acc = run_epoch(model, test_loader, None, crit, train=False)
        wg, _, _ = eval_worst_group(model, test_loader)
        H = spectral_entropy(model)
        a, b = direction_alignment(model, base_loader)
        logs.append({"epoch": ep, "keep_ratio": keep_ratio,
                     "tr_acc": correct/total if total>0 else 0.0,
                     "te_acc": te_acc, "worst_group_acc": wg, "H": H, "alpha": a, "beta": b})
        print(f"[AGC_INV-CFP] ep{ep:02d} keep={keep_ratio:.2f} te_acc={te_acc:.3f} worstG={wg:.3f} H={H:.3f} alpha={a:.3f} beta={b:.3f}")

    return model, logs

def train_agc(train_set, test_loader, variant="agc_despur",
              total_epochs=30, lr=3e-4, keep_start=0.3, keep_end=0.9, k_desired=2):
    assert variant in {"agc_despur","agc_easy"}
    model = CNNFeatSmall().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    base_loader = DataLoader(train_set, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    labels = {}
    for _, y, _, idx in base_loader:
        for i, sid in enumerate(idx.tolist()):
            labels[sid] = int(y[i])

    logs=[]
    for ep in range(1, total_epochs+1):
        keep_ratio = min(keep_end, keep_start + (keep_end - keep_start) * (ep-1)/(total_epochs-1))
        U = topk_basis_from_classifier(model, k_desired=k_desired)
        scores = compute_align_scores(model, base_loader, U)
        selected = select_indices_by_curriculum(scores, labels, keep_ratio, variant=variant, min_random_frac=0.1)

        subset = Subset(train_set, selected)
        train_loader = DataLoader(subset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
        tr_loss, tr_acc = run_epoch(model, train_loader, opt, crit, train=True)

        te_loss, te_acc = run_epoch(model, test_loader, None, crit, train=False)
        wg, _, _ = eval_worst_group(model, test_loader)
        H = spectral_entropy(model)
        a,b = direction_alignment(model, base_loader)
        logs.append({"epoch": ep, "keep_ratio": keep_ratio, "tr_acc": tr_acc,
                     "te_acc": te_acc, "worst_group_acc": wg, "H": H, "alpha": a, "beta": b})
        print(f"[{variant.upper()}] ep{ep:02d} keep={keep_ratio:.2f} te_acc={te_acc:.3f} worstG={wg:.3f} H={H:.3f}")
    return model, logs

# -----------------------
# Task builder
# -----------------------
def build_task(task: str, root="./data", seed=0, p_train=0.99, p_test=0.1,
               train_fraction=1.0, download=True):
    task = task.lower()
    if task=="colored_mnist":
        tr = ColoredMNIST(root, "train", p_train=p_train, p_test=p_test, seed=seed, download=download)
        te = ColoredMNIST(root, "test",  p_train=p_train, p_test=p_test, seed=seed+1, download=download)
        batch = 256
    elif task=="colored_fmnist":
        tr = ColoredFashionMNIST(root, "train", p_train=p_train, p_test=p_test, seed=seed, download=download)
        te = ColoredFashionMNIST(root, "test",  p_train=p_train, p_test=p_test, seed=seed+1, download=download)
        batch = 256
    elif task=="colored_cifar10":
        tr = ColoredCIFAR10(root, "train", p_train=max(p_train, 0.995), p_test=p_test, seed=seed, download=download)
        te = ColoredCIFAR10(root, "test",  p_train=max(p_train, 0.995), p_test=p_test, seed=seed+1, download=download)
        batch = 128
    else:
        raise ValueError(f"Unknown task: {task}")

    if train_fraction < 1.0:
        n = len(tr)
        idx = np.random.RandomState(seed).choice(n, size=int(n*train_fraction), replace=False)
        tr = Subset(tr, idx.tolist())

    train_loader = DataLoader(tr, batch_size=batch, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(te, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)
    return tr, te, train_loader, test_loader

# -----------------------
# Logging & Plotting
# -----------------------
def write_csv(path: str, logs: List[Dict]):
    import csv
    if not logs: return
    keys = sorted(list({k for row in logs for k in row.keys()}))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in logs:
            writer.writerow(row)

def aggregate_across_seeds(results_per_seed: List[List[Dict]], metric: str, max_epoch=None):
    min_len = min(len(logs) for logs in results_per_seed)
    if max_epoch is not None:
        min_len = min(min_len, max_epoch)
    epochs = np.array([results_per_seed[0][i]["epoch"] for i in range(min_len)])
    vals = np.stack([[logs[i].get(metric, np.nan) for i in range(min_len)] for logs in results_per_seed], axis=0)
    mean = np.nanmean(vals, axis=0)
    std = np.nanstd(vals, axis=0, ddof=1) if vals.shape[0] >= 2 else np.zeros_like(mean)
    se = std / np.sqrt(max(vals.shape[0],1))
    ci = 1.96 * se
    return epochs, mean, mean - ci, mean + ci

def plot_metric_curves(agg: Dict[str, Dict[str, Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]]],
                       metric_name: str, out_png: str, title: str):
    plt.figure(figsize=(7,5))
    for method, d in agg.items():
        ep, mu, lo, hi = d[metric_name]
        plt.plot(ep, mu, label=method)
        plt.fill_between(ep, lo, hi, alpha=0.2)
    plt.xlabel("Epoch"); plt.ylabel(metric_name); plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

def plot_bars_best(results_by_method_seeds: Dict[str, List[List[Dict]]],
                   metric: str, out_png: str, title: str):
    labels = []
    means = []
    ci95  = []
    for method, seed_logs in results_by_method_seeds.items():
        bests = []
        for logs in seed_logs:
            arr = [row.get(metric, float("nan")) for row in logs]
            bests.append(np.nanmax(arr))
        mu = float(np.nanmean(bests)); std = float(np.nanstd(bests, ddof=1)) if len(bests)>=2 else 0.0
        se = std / math.sqrt(max(1,len(bests))); ci = 1.96*se
        labels.append(method); means.append(mu); ci95.append(ci)
    x = np.arange(len(labels))
    plt.figure(figsize=(6,5))
    plt.bar(x, means, yerr=ci95, capsize=4)
    plt.xticks(x, labels, rotation=15)
    plt.ylabel(metric); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

# -----------------------
# Runner
# -----------------------
def run_all(task: str, seeds: List[int], outdir: str,
            epochs=30, p_train=0.99, p_test=0.1, train_fraction=1.0,
            lr=3e-4, upsample=10):
    ensure_dir(outdir)
    cfg = dict(task=task, seeds=seeds, epochs=epochs, p_train=p_train, p_test=p_test,
               train_fraction=train_fraction, lr=lr, upsample=upsample)
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    results_by_method = {
        "ERM": [],
        "JTT": [],
        "AGC-Despur": [],
        "AGC-Easy": [],
        "AGC-InvCFP": []  # NEW
    }

    for s in seeds:
        set_seed(s)
        tr_set, te_set, tr_loader, te_loader = build_task(task, seed=s, p_train=p_train, p_test=p_test,
                                                          train_fraction=train_fraction)

        # ERM
        model = CNNFeatSmall()
        erm_model, erm_logs = train_erm(model, tr_loader, te_loader, epochs=epochs, lr=lr)
        write_csv(os.path.join(outdir, f"ERM_seed{s}.csv"), erm_logs)
        results_by_method["ERM"].append(erm_logs)

        # JTT
        jtt_model, jtt_logs = train_jtt(tr_set, te_loader, total_epochs=epochs, stage1_epochs=max(1,epochs//6),
                                        upsample=upsample, lr=lr)
        write_csv(os.path.join(outdir, f"JTT_seed{s}.csv"), jtt_logs)
        results_by_method["JTT"].append(jtt_logs)

        # AGC-Despur
        agc_d_model, agc_d_logs = train_agc(tr_set, te_loader, variant="agc_despur", total_epochs=epochs, lr=lr,
                                            keep_start=0.3, keep_end=0.9, k_desired=2)
        write_csv(os.path.join(outdir, f"AGC_Despur_seed{s}.csv"), agc_d_logs)
        results_by_method["AGC-Despur"].append(agc_d_logs)

        # AGC-Easy
        agc_e_model, agc_e_logs = train_agc(tr_set, te_loader, variant="agc_easy", total_epochs=epochs, lr=lr,
                                            keep_start=0.3, keep_end=0.9, k_desired=2)
        write_csv(os.path.join(outdir, f"AGC_Easy_seed{s}.csv"), agc_e_logs)
        results_by_method["AGC-Easy"].append(agc_e_logs)

        # NEW: AGC-InvCFP
        agc_inv_model, agc_inv_logs = train_agc_invcfp(tr_set, te_loader, total_epochs=epochs, lr=lr,
                                                       keep_start=0.3, keep_end=0.9, k_desired=2,
                                                       probe_frac=0.6, lambda_cons=0.2)
        write_csv(os.path.join(outdir, f"AGC_InvCFP_seed{s}.csv"), agc_inv_logs)
        results_by_method["AGC-InvCFP"].append(agc_inv_logs)

    # Aggregation & plots
    agg = {}
    for method, seed_logs in results_by_method.items():
        agg[method] = {
            "te_acc": aggregate_across_seeds(seed_logs, metric="te_acc"),
            "worst_group_acc": aggregate_across_seeds(seed_logs, metric="worst_group_acc"),
            "H": aggregate_across_seeds(seed_logs, metric="H"),
            "alpha": aggregate_across_seeds(seed_logs, metric="alpha"),
            "beta": aggregate_across_seeds(seed_logs, metric="beta"),
        }

    plot_metric_curves(agg, "te_acc", os.path.join(outdir, "curve_te_acc.png"),
                       f"{task}: Test Accuracy (mean±95% CI)")
    plot_metric_curves(agg, "worst_group_acc", os.path.join(outdir, "curve_worst_group.png"),
                       f"{task}: Worst-group Acc (mean±95% CI)")
    plot_metric_curves(agg, "H", os.path.join(outdir, "curve_spectral_entropy.png"),
                       f"{task}: Spectral Entropy H")
    plot_metric_curves(agg, "alpha", os.path.join(outdir, "curve_alpha.png"),
                       f"{task}: Alignment to Label dir (alpha)")
    plot_metric_curves(agg, "beta", os.path.join(outdir, "curve_beta.png"),
                       f"{task}: Alignment to Color dir (beta)")

    plot_bars_best(results_by_method, "te_acc", os.path.join(outdir, "best_te_acc.png"),
                   f"{task}: Best Test Acc across seeds")
    plot_bars_best(results_by_method, "worst_group_acc", os.path.join(outdir, "best_worst_group.png"),
                   f"{task}: Best Worst-group Acc across seeds")

    print(f"[DONE] Results saved under: {outdir}")

# -----------------------
# Main (CLI)
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["colored_mnist", "colored_fmnist", "colored_cifar10"],
                        help="Which tasks to run")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs per method")
    parser.add_argument("--p-train", type=float, default=0.99, help="Train correlation P[c==y]")
    parser.add_argument("--p-test", type=float, default=0.1, help="Test correlation P[c==y]")
    parser.add_argument("--train-fraction", type=float, default=1.0, help="Subsample fraction of training set")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--upsample", type=int, default=10, help="JTT upweight factor")
    parser.add_argument("--out-root", type=str, default="experiments/curriculum")
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for task in args.tasks:
        outdir = os.path.join(args.out_root, task, timestamp)
        ensure_dir(outdir)
        print(f"\n=== Task: {task} | seeds={seeds} | epochs={args.epochs} ===")
        run_all(task=task, seeds=seeds, outdir=outdir,
                epochs=args.epochs, p_train=args.p_train, p_test=args.p_test,
                train_fraction=args.train_fraction, lr=args.lr, upsample=args.upsample)

if __name__ == "__main__":
    main()
