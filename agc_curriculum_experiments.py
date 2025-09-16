# -*- coding: utf-8 -*-
"""
AGOP-aware Curriculum: multi-task, multi-seed experiments with plots.

Compare: ERM, JTT, AGC-InvCFP-old (z + W^T W + full MSE), AGC-InvCFP-new (tap h_l + AGOP + projection MSE)
Tasks: colored_mnist, colored_fmnist, colored_kmnist, colored_cifar10

Saves CSV/PNG under: experiments/curriculum/<task>/<timestamp>/
"""

import os, math, random, argparse, json
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets
import matplotlib.pyplot as plt

# -----------------------
# Small utilities
# -----------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# === Subset 构造助手：把“底层索引”映射为当前 train_set 的位置索引 ===
def subset_from_base_indices(train_set, base_indices: List[int]) -> Subset:
    if isinstance(train_set, Subset):
        base_to_pos = {int(b): i for i, b in enumerate(train_set.indices)}
        pos = [base_to_pos[i] for i in base_indices if int(i) in base_to_pos]
        if len(pos) == 0:
            pos = list(range(len(train_set)))
        return Subset(train_set, pos)
    else:
        return Subset(train_set, base_indices)

# -----------------------
# Spurious-color datasets
# -----------------------
class ColoredBinaryBase(Dataset):
    """
    Wrap a base dataset (PIL image, raw_label) into binary label y and spurious color c,
    with P[c==y] = p_corr. Colors are cached at __init__ for reproducibility.
    Provides counterfactual color-flip for each index.
    """
    def __init__(self, base_ds, make_binary_label, split: str, p_train=0.99, p_test=0.1, seed=0):
        assert split in ["train","test"]
        self.base = base_ds
        self.split = split
        self.p_corr = p_train if split=="train" else p_test
        rng = np.random.RandomState(seed if seed is not None else 0)

        self.labels = []
        self.colors = []
        for i in range(len(self.base)):
            _, raw_y = self.base[i]
            y = int(make_binary_label(raw_y))
            corr = rng.rand() < self.p_corr
            c = y if corr else 1 - y
            self.labels.append(y); self.colors.append(c)
        self.labels = np.array(self.labels, dtype=np.int64)
        self.colors = np.array(self.colors, dtype=np.int64)

    def __len__(self): return len(self.base)

    def _pil_to_gray_np(self, pil_img) -> np.ndarray:
        arr = np.array(pil_img, dtype=np.float32)/255.0  # HxW or HxWx3
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]
        return arr

    def _colorize_gray(self, gray: np.ndarray, c: int) -> torch.Tensor:
        R = gray if c==0 else np.zeros_like(gray)
        G = gray if c==1 else np.zeros_like(gray)
        B = np.zeros_like(gray)
        return torch.from_numpy(np.stack([R, G, B], axis=0).astype(np.float32))

    def get_flip_tensor(self, idx: int) -> torch.Tensor:
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

class ColoredKMNIST(ColoredBinaryBase):
    def __init__(self, root, split="train", p_train=0.99, p_test=0.1, seed=0, download=True):
        base = datasets.KMNIST(root=root, train=(split=="train"), download=download)
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
# Model (with a "tap" to expose an intermediate representation h_l)
# -----------------------
class CNNFeatSmall(nn.Module):
    """Light CNN with a feature head (for 28x28 or 32x32), exposing a tap 'pre_proj2' (dim=256)."""
    def __init__(self, d_feat=256, num_classes=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4,4)),
            nn.Flatten(),
        )
        self.proj1 = nn.Sequential(nn.Linear(128*4*4, 256), nn.ReLU(inplace=True))  # tap here
        self.proj2 = nn.Sequential(nn.Linear(256, d_feat), nn.ReLU(inplace=True))
        self.classifier = nn.Linear(d_feat, num_classes)

    def forward_with_tap(self, x, tap="pre_proj2", return_all=False):
        h0 = self.block(x)              # [B, 128*4*4]
        h1 = self.proj1(h0)             # tap: [B, 256]
        z  = self.proj2(h1)             # [B, d_feat]
        logits = self.classifier(z)
        if return_all:
            return logits, h0, h1, z
        if tap == "pre_proj2":
            return logits, h1
        elif tap == "flat":
            return logits, h0
        elif tap == "feat":
            return logits, z
        else:
            raise ValueError(f"Unknown tap: {tap}")

    def head_from_tap(self, h):  # logits from a given tap h
        z  = self.proj2(h)
        logits = self.classifier(z)
        return logits

    def forward(self, x, return_feat=False):
        logits, z = self.forward_with_tap(x, tap="feat")
        if return_feat:
            return logits, z
        return logits

# -----------------------
# AGOP / NFA utilities
# -----------------------
@torch.no_grad()
def topk_basis_from_classifier(model: nn.Module, k_desired=2, eps=1e-8) -> torch.Tensor:
    """For the last linear layer only (W^T W)."""
    W = model.classifier.weight.detach()
    M = W.T @ W
    evals, evecs = torch.linalg.eigh(M)
    if float(evals.max()) <= 0:
        return evecs[:, -1:].contiguous()
    mask = evals > (eps * float(evals.max()))
    r = int(mask.sum().item())
    k = max(1, min(k_desired, r))
    return evecs[:, -k:].contiguous()

def estimate_agop_topk(model: nn.Module, loader: DataLoader, tap="pre_proj2",
                       k=2, max_batches=5, device=DEVICE) -> torch.Tensor:
    """
    Estimate AGOP G_l ≈ sum_{b,c} (∂ sum f_c / ∂ h_l)ᵀ(∂ sum f_c / ∂ h_l) on a few batches.
    Return top-k eigenvectors U (D x k) at the tap representation.
    """
    model.eval()
    G = None
    seen = 0
    C = None
    batches = 0
    for b, (x, y, c, idx) in enumerate(loader):
        if b >= max_batches: break
        batches += 1
        x = x.to(device)
        with torch.no_grad():
            logits, h = model.forward_with_tap(x, tap=tap)  # h: [B, D]
        D = h.shape[1]
        if C is None:
            C = logits.shape[1]
        h = h.detach()
        h.requires_grad_(True)
        logits_h = model.head_from_tap(h)                  # only head depends on h
        G_batch = torch.zeros(D, D, device=device)
        for cls in range(C):
            s = logits_h[:, cls].sum()
            g = torch.autograd.grad(s, h, retain_graph=True, create_graph=False)[0]  # [B, D]
            G_batch += g.T @ g
        if G is None:
            G = G_batch.detach()
        else:
            G += G_batch.detach()
        seen += x.size(0)

    if G is None:
        # fallback
        W = model.classifier.weight.detach().to(device)
        G = W.T @ W

    G = G.float().detach().cpu()
    evals, evecs = torch.linalg.eigh(G)
    U = evecs[:, -k:].contiguous()  # [D, k]
    return U.to(device)

@torch.no_grad()
def spectral_entropy(model: nn.Module) -> float:
    W = model.classifier.weight.detach()
    M = W.T @ W
    evals = torch.linalg.eigvalsh(M).clamp(min=1e-12)
    p = (evals / evals.sum()).cpu().numpy()
    return float(-(p * np.log(p)).sum())

# -----------------------
# Selection scores (Δ_U) & directions
# -----------------------
@torch.no_grad()
def compute_pair_sensitivity_old(model: nn.Module, loader: DataLoader, dataset: Dataset,
                                 U: torch.Tensor, probe_frac: float = 1.0) -> Dict[int, float]:
    """AGC-InvCFP-old: use final feature z and U from W^T W."""
    model.eval(); U = U.to(DEVICE)
    scores = {}; rng = np.random.RandomState(0)
    for x, y, c, idx in loader:
        idx_list = idx.tolist()
        if probe_frac < 1.0:
            take = rng.rand(len(idx_list)) < probe_frac
            idx_list = [idx_list[i] for i,b in enumerate(take) if b]
            if len(idx_list) == 0: continue
            mask = torch.tensor([i in set(idx_list) for i in idx.tolist()], dtype=torch.bool)
            x = x[mask]; idx = idx[mask]
        x = x.to(DEVICE)
        x_flip = torch.stack([dataset.get_flip_tensor(int(i)) for i in idx_list], dim=0).to(DEVICE)

        _, z  = model(x, return_feat=True)
        _, zf = model(x_flip, return_feat=True)
        z  = F.normalize(z, dim=1); zf = F.normalize(zf, dim=1)
        p  = z @ U; pf = zf @ U
        num = ((p - pf)**2).sum(dim=1)
        den = (z.pow(2).sum(dim=1) + zf.pow(2).sum(dim=1) + 1e-8)
        delta = (num / den).detach().cpu().numpy()
        for i, sid in enumerate(idx_list):
            scores[sid] = float(delta[i])
    return scores

@torch.no_grad()
def compute_pair_sensitivity_tap(model: nn.Module, loader: DataLoader, dataset: Dataset, U: torch.Tensor,
                                 tap="pre_proj2", probe_frac: float = 1.0) -> Dict[int, float]:
    """AGC-InvCFP-new: use tap representation h_l and AGOP-U."""
    model.eval(); U = U.to(DEVICE)
    scores = {}; rng = np.random.RandomState(0)
    for x, y, c, idx in loader:
        idx_list = idx.tolist()
        if probe_frac < 1.0:
            take = rng.rand(len(idx_list)) < probe_frac
            idx_list = [idx_list[i] for i,b in enumerate(take) if b]
            if len(idx_list) == 0: continue
            mask = torch.tensor([i in set(idx_list) for i in idx.tolist()], dtype=torch.bool)
            x = x[mask]; idx = idx[mask]
        x = x.to(DEVICE)
        x_flip = torch.stack([dataset.get_flip_tensor(int(i)) for i in idx_list], dim=0).to(DEVICE)

        _, h  = model.forward_with_tap(x, tap=tap)
        _, hf = model.forward_with_tap(x_flip, tap=tap)
        h  = F.normalize(h, dim=1); hf = F.normalize(hf, dim=1)
        p  = h @ U; pf = hf @ U
        num = ((p - pf)**2).sum(dim=1)
        den = (h.pow(2).sum(dim=1) + hf.pow(2).sum(dim=1) + 1e-8)
        delta = (num / den).detach().cpu().numpy()
        for i, sid in enumerate(idx_list):
            scores[sid] = float(delta[i])
    return scores

@torch.no_grad()
def direction_alignment(model: nn.Module, loader: DataLoader) -> Tuple[float,float]:
    """alpha: alignment to label dir; beta: alignment to color dir (on W^T W top eigenvector)."""
    model.eval()
    Z, Ys, Cs = [], [], []
    for x, y, c, _ in loader:
        x = x.to(DEVICE)
        _, z = model(x, return_feat=True)
        Z.append(z.detach().cpu()); Ys.append(y.detach().cpu()); Cs.append(c.detach().cpu())
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
# Core train / eval
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

@torch.no_grad()
def eval_group_counts(loader):
    counts = {(y,c):0 for y in [0,1] for c in [0,1]}
    for _, y, c, _ in loader:
        for i in range(len(y)):
            counts[(int(y[i]), int(c[i]))] += 1
    return counts

@torch.no_grad()
def eval_flip_acc(model, test_set, batch_size=256):
    base = test_set
    while isinstance(base, Subset):
        base = base.dataset
    assert hasattr(base, "get_flip_tensor"), "flip eval only for Colored* datasets"
    model.eval()
    Xs, Ys = [], []
    N = len(test_set)
    for i in range(N):
        idx = int(test_set.indices[i]) if isinstance(test_set, Subset) else i
        x_flip = base.get_flip_tensor(idx)
        _, y, _, _ = test_set[i]
        Xs.append(x_flip); Ys.append(y)
    X = torch.stack(Xs, dim=0); Y = torch.tensor(Ys, dtype=torch.long)
    acc = 0.0; tot = 0
    for s in range(0, len(X), batch_size):
        xb = X[s:s+batch_size].to(DEVICE)
        yb = Y[s:s+batch_size].to(DEVICE)
        pred = model(xb).argmax(1)
        acc += (pred==yb).sum().item()
        tot += yb.numel()
    return acc / max(1, tot)

@torch.no_grad()
def eval_perm_acc(model, test_loader, seed=0):
    """Shuffle test labels; accuracy should be near chance (0.5) if无泄漏。"""
    rng = np.random.RandomState(seed)
    model.eval(); tot=0; corr=0
    all_y = []
    all_pred = []
    for x, y, c, idx in test_loader:
        x = x.to(DEVICE)
        pred = model(x).argmax(1).cpu().numpy()
        all_pred.append(pred); all_y.append(y.numpy())
    all_pred = np.concatenate(all_pred)
    all_y = np.concatenate(all_y)
    perm = rng.permutation(len(all_y))
    y_shuf = all_y[perm]
    corr = (all_pred == y_shuf).sum()
    tot  = len(all_y)
    return corr / max(1, tot)

@torch.no_grad()
def eval_deltaU_mean_tap(model, test_loader, test_set, k=2, tap="pre_proj2",
                         probe_frac=0.5, agop_eval_batches=2):
    base = test_set
    while isinstance(base, Subset):
        base = base.dataset
    U = estimate_agop_topk(model, test_loader, tap=tap, k=k, max_batches=agop_eval_batches)
    d = compute_pair_sensitivity_tap(model, test_loader, base, U, tap=tap, probe_frac=probe_frac)
    return float(np.mean(list(d.values()))) if len(d)>0 else float("nan")

def check_train_test_overlap(train_set, test_set) -> int:
    def unwrap_indices(ds):
        if isinstance(ds, Subset):
            base_idx = set(int(i) for i in ds.indices)
            base = ds.dataset
        else:
            base = ds
            base_idx = set(range(len(base)))
        # try to unwrap nested subset
        while isinstance(base, Subset):
            base = base.dataset
        return base_idx
    tr_idx = unwrap_indices(train_set)
    te_idx = unwrap_indices(test_set)
    return len(tr_idx & te_idx)

def color_only_baseline_acc(loader) -> float:
    """Use c or 1-c to predict y on test set, take the better."""
    ys, cs = [], []
    for _, y, c, _ in loader:
        ys.append(y.numpy()); cs.append(c.numpy())
    y = np.concatenate(ys); c = np.concatenate(cs)
    acc1 = (y == c).mean()
    acc2 = (y == 1 - c).mean()
    return max(acc1, acc2)

# -----------------------
# Training variants
# -----------------------
def train_erm(model, train_loader, test_loader, te_set,
              epochs=30, lr=3e-4, wd=1e-4,
              eval_flip_every=5, eval_delta_every=5, eval_perm_every=5,
              delta_probe_frac=0.5, agop_eval_batches=2, perm_seed=0):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()
    logs=[]
    printed_groups=False
    for ep in range(1, epochs+1):
        tr_loss, tr_acc = run_epoch(model, train_loader, opt, crit, train=True)
        te_loss, te_acc = run_epoch(model, test_loader, None, crit, train=False)
        wg, _, _ = eval_worst_group(model, test_loader)
        H = spectral_entropy(model)
        a, b = direction_alignment(model, train_loader)

        flip_acc = np.nan
        deltaU_mean = np.nan
        perm_acc = np.nan
        if eval_flip_every>0 and ((ep % eval_flip_every)==1 or ep==epochs):
            flip_acc = eval_flip_acc(model, te_set)
        if eval_delta_every>0 and ((ep % eval_delta_every)==1 or ep==epochs):
            deltaU_mean = eval_deltaU_mean_tap(model, test_loader, te_set,
                                               k=2, tap="pre_proj2",
                                               probe_frac=delta_probe_frac,
                                               agop_eval_batches=agop_eval_batches)
        if eval_perm_every>0 and ((ep % eval_perm_every)==1 or ep==epochs):
            perm_acc = eval_perm_acc(model, test_loader, seed=perm_seed)

        logs.append({"epoch": ep, "tr_acc": tr_acc, "te_acc": te_acc, "worst_group_acc": wg,
                     "H": H, "alpha": a, "beta": b,
                     "flip_acc": flip_acc, "deltaU_mean": deltaU_mean, "perm_acc": perm_acc})
        if not printed_groups:
            print("[test group counts]", eval_group_counts(test_loader)); printed_groups=True
        msg = f"[ERM] ep{ep:02d} te_acc={te_acc:.3f} worstG={wg:.3f} H={H:.3f} alpha={a:.3f} beta={b:.3f}"
        if not np.isnan(flip_acc):   msg += f" flip-acc={flip_acc:.3f}"
        if not np.isnan(deltaU_mean):msg += f" ΔU={deltaU_mean:.4f}"
        if not np.isnan(perm_acc):   msg += f" perm-acc={perm_acc:.3f}"
        print(msg)
    return model, logs

def train_jtt(train_set, test_loader, te_set,
              total_epochs=30, stage1_epochs=5, upsample=10, lr=3e-4,
              eval_flip_every=5, eval_delta_every=5, eval_perm_every=5,
              delta_probe_frac=0.5, agop_eval_batches=2, perm_seed=0):
    base = CNNFeatSmall().to(DEVICE)
    opt = torch.optim.AdamW(base.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    base_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
    logs=[]
    printed_groups=False
    for ep in range(1, stage1_epochs+1):
        run_epoch(base, base_loader, opt, crit, train=True)
        te_loss, te_acc = run_epoch(base, test_loader, None, crit, train=False)
        wg, _, _ = eval_worst_group(base, test_loader)
        H = spectral_entropy(base); a,b = direction_alignment(base, base_loader)
        logs.append({"epoch": ep, "te_acc": te_acc, "worst_group_acc": wg, "H": H, "alpha": a, "beta": b})
        if not printed_groups:
            print("[test group counts]", eval_group_counts(test_loader)); printed_groups=True
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
        flip_acc   = eval_flip_acc(model, te_set) if (eval_flip_every>0 and ((ep % eval_flip_every)==1 or ep==total_epochs)) else np.nan
        deltaU_mean= eval_deltaU_mean_tap(model, test_loader, te_set, k=2, tap="pre_proj2",
                                          probe_frac=delta_probe_frac, agop_eval_batches=agop_eval_batches) \
                     if (eval_delta_every>0 and ((ep % eval_delta_every)==1 or ep==total_epochs)) else np.nan
        perm_acc   = eval_perm_acc(model, test_loader, seed=perm_seed) \
                     if (eval_perm_every>0 and ((ep % eval_perm_every)==1 or ep==total_epochs)) else np.nan

        logs.append({"epoch": ep, "tr_acc": tr_acc, "te_acc": te_acc, "worst_group_acc": wg,
                     "H": H, "alpha": a, "beta": b,
                     "flip_acc": flip_acc, "deltaU_mean": deltaU_mean, "perm_acc": perm_acc})
        msg = f"[JTT] ep{ep:02d} te_acc={te_acc:.3f} worstG={wg:.3f} H={H:.3f}"
        if not np.isnan(flip_acc):   msg += f" flip-acc={flip_acc:.3f}"
        if not np.isnan(deltaU_mean):msg += f" ΔU={deltaU_mean:.4f}"
        if not np.isnan(perm_acc):   msg += f" perm-acc={perm_acc:.3f}"
        print(msg)
    return model, logs

def _unwrap_colored_dataset(ds: Dataset) -> ColoredBinaryBase:
    base = ds
    while isinstance(base, Subset):
        base = base.dataset
    assert isinstance(base, ColoredBinaryBase), "This method needs Colored* dataset."
    return base

# AGC-InvCFP-old
def train_agc_invcfp_old(train_set, test_loader, te_set,
                         total_epochs=30, lr=3e-4,
                         keep_start=0.3, keep_end=0.9, k_desired=2,
                         probe_frac=0.6, lambda_cons=0.2,
                         eval_flip_every=5, eval_delta_every=5, eval_perm_every=5,
                         delta_probe_frac=0.5, agop_eval_batches=2, perm_seed=0):
    dataset_base = _unwrap_colored_dataset(train_set)
    model = CNNFeatSmall().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    base_loader = DataLoader(train_set, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    logs=[]; printed_groups=False
    for ep in range(1, total_epochs+1):
        keep_ratio = min(keep_end, keep_start + (keep_end - keep_start) * (ep-1)/(total_epochs-1))
        U = topk_basis_from_classifier(model, k_desired=k_desired)  # last-layer W^T W

        delta = compute_pair_sensitivity_old(model, base_loader, dataset_base, U, probe_frac=probe_frac)
        n = len(delta); m = max(1, int(n * keep_ratio))
        top_idx = [i for i,_ in sorted(delta.items(), key=lambda kv: kv[1], reverse=True)[:m]]

        remain = list(set(delta.keys()) - set(top_idx))
        if len(remain) > 0:
            top_idx += random.sample(remain, min(int(0.1*n), len(remain)))

        subset = subset_from_base_indices(train_set, top_idx)
        train_loader = DataLoader(subset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)

        # one epoch with full-feature consistency (old)
        model.train()
        total, correct = 0, 0
        for x, y, c, idx in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x_flip = torch.stack([dataset_base.get_flip_tensor(int(i)) for i in idx.tolist()], dim=0).to(DEVICE)
            logits, z   = model(x, return_feat=True)
            logits_f, zf= model(x_flip, return_feat=True)
            ce = crit(logits, y) + crit(logits_f, y)
            cons = F.mse_loss(z, zf)  # full
            loss = ce + lambda_cons * cons
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            pred = logits.argmax(1)
            total += y.size(0); correct += (pred==y).sum().item()

        te_loss, te_acc = run_epoch(model, test_loader, None, crit, train=False)
        wg, _, _ = eval_worst_group(model, test_loader)
        H = spectral_entropy(model); a, b = direction_alignment(model, base_loader)

        flip_acc   = eval_flip_acc(model, te_set) if (eval_flip_every>0 and ((ep % eval_flip_every)==1 or ep==total_epochs)) else np.nan
        deltaU_mean= eval_deltaU_mean_tap(model, test_loader, te_set, k=2, tap="pre_proj2",
                                          probe_frac=delta_probe_frac, agop_eval_batches=agop_eval_batches) \
                     if (eval_delta_every>0 and ((ep % eval_delta_every)==1 or ep==total_epochs)) else np.nan
        perm_acc   = eval_perm_acc(model, test_loader, seed=perm_seed) \
                     if (eval_perm_every>0 and ((ep % eval_perm_every)==1 or ep==total_epochs)) else np.nan

        logs.append({"epoch": ep, "keep_ratio": keep_ratio,
                     "tr_acc": correct/total if total>0 else 0.0,
                     "te_acc": te_acc, "worst_group_acc": wg, "H": H, "alpha": a, "beta": b,
                     "flip_acc": flip_acc, "deltaU_mean": deltaU_mean, "perm_acc": perm_acc})
        if not printed_groups:
            print("[test group counts]", eval_group_counts(test_loader)); printed_groups=True
        msg = f"[AGC_INV-CFP_OLD] ep{ep:02d} keep={keep_ratio:.2f} te_acc={te_acc:.3f} worstG={wg:.3f} H={H:.3f}"
        if not np.isnan(flip_acc):   msg += f" flip-acc={flip_acc:.3f}"
        if not np.isnan(deltaU_mean):msg += f" ΔU={deltaU_mean:.4f}"
        if not np.isnan(perm_acc):   msg += f" perm-acc={perm_acc:.3f}"
        print(msg)
    return model, logs

# AGC-InvCFP-new (AGOP @ tap + projection consistency)
def train_agc_invcfp_new(train_set, test_loader, te_set,
                         total_epochs=30, lr=3e-4,
                         keep_start=0.3, keep_end=0.9, k_desired=2,
                         probe_frac=0.6, lambda_cons=0.2,
                         agop_update_every=1, agop_train_batches=3,
                         eval_flip_every=5, eval_delta_every=5, eval_perm_every=5,
                         delta_probe_frac=0.5, agop_eval_batches=2, perm_seed=0,
                         tap="pre_proj2"):
    dataset_base = _unwrap_colored_dataset(train_set)
    model = CNNFeatSmall().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    base_loader = DataLoader(train_set, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    logs=[]; printed_groups=False
    U = estimate_agop_topk(model, base_loader, tap=tap, k=k_desired, max_batches=agop_train_batches)
    for ep in range(1, total_epochs+1):
        keep_ratio = min(keep_end, keep_start + (keep_end - keep_start) * (ep-1)/(total_epochs-1))
        if (ep==1) or (agop_update_every>0 and ep % agop_update_every == 0):
            U = estimate_agop_topk(model, base_loader, tap=tap, k=k_desired, max_batches=agop_train_batches)

        delta = compute_pair_sensitivity_tap(model, base_loader, dataset_base, U, tap=tap, probe_frac=probe_frac)
        n = len(delta); m = max(1, int(n * keep_ratio))
        top_idx = [i for i,_ in sorted(delta.items(), key=lambda kv: kv[1], reverse=True)[:m]]

        remain = list(set(delta.keys()) - set(top_idx))
        if len(remain) > 0:
            top_idx += random.sample(remain, min(int(0.1*n), len(remain)))

        subset = subset_from_base_indices(train_set, top_idx)
        train_loader = DataLoader(subset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)

        # one epoch with projection consistency on AGOP subspace
        model.train()
        total, correct = 0, 0
        for x, y, c, idx in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x_flip = torch.stack([dataset_base.get_flip_tensor(int(i)) for i in idx.tolist()], dim=0).to(DEVICE)

            logits = model(x)
            logits_f = model(x_flip)

            # projection consistency on tap
            _, h  = model.forward_with_tap(x, tap=tap)
            _, hf = model.forward_with_tap(x_flip, tap=tap)
            proj  = h  @ U
            projf = hf @ U

            ce = crit(logits, y) + crit(logits_f, y)
            cons = F.mse_loss(proj, projf)
            loss = ce + lambda_cons * cons

            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

            pred = logits.argmax(1)
            total += y.size(0); correct += (pred==y).sum().item()

        te_loss, te_acc = run_epoch(model, test_loader, None, crit, train=False)
        wg, _, _ = eval_worst_group(model, test_loader)
        H = spectral_entropy(model); a, b = direction_alignment(model, base_loader)

        flip_acc   = eval_flip_acc(model, te_set) if (eval_flip_every>0 and ((ep % eval_flip_every)==1 or ep==total_epochs)) else np.nan
        deltaU_mean= eval_deltaU_mean_tap(model, test_loader, te_set, k=k_desired, tap=tap,
                                          probe_frac=delta_probe_frac, agop_eval_batches=agop_eval_batches) \
                     if (eval_delta_every>0 and ((ep % eval_delta_every)==1 or ep==total_epochs)) else np.nan
        perm_acc   = eval_perm_acc(model, test_loader, seed=perm_seed) \
                     if (eval_perm_every>0 and ((ep % eval_perm_every)==1 or ep==total_epochs)) else np.nan

        logs.append({"epoch": ep, "keep_ratio": keep_ratio,
                     "tr_acc": correct/total if total>0 else 0.0,
                     "te_acc": te_acc, "worst_group_acc": wg, "H": H, "alpha": a, "beta": b,
                     "flip_acc": flip_acc, "deltaU_mean": deltaU_mean, "perm_acc": perm_acc})
        if not printed_groups:
            print("[test group counts]", eval_group_counts(test_loader)); printed_groups=True
        msg = f"[AGC_INV-CFP_NEW] ep{ep:02d} keep={keep_ratio:.2f} te_acc={te_acc:.3f} worstG={wg:.3f} H={H:.3f}"
        if not np.isnan(flip_acc):   msg += f" flip-acc={flip_acc:.3f}"
        if not np.isnan(deltaU_mean):msg += f" ΔU={deltaU_mean:.4f}"
        if not np.isnan(perm_acc):   msg += f" perm-acc={perm_acc:.3f}"
        print(msg)
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
    elif task=="colored_kmnist":
        tr = ColoredKMNIST(root, "train", p_train=p_train, p_test=p_test, seed=seed, download=download)
        te = ColoredKMNIST(root, "test",  p_train=p_train, p_test=p_test, seed=seed+1, download=download)
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
        if metric_name not in d: continue
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
            lr=3e-4, upsample=10,
            eval_flip_every=5, eval_delta_every=5, eval_perm_every=5,
            delta_probe_frac=0.5, agop_eval_batches=2, perm_seed=0,
            agop_update_every=1, agop_train_batches=3):
    ensure_dir(outdir)
    # Build once (for overlap & color-only stats)
    tr_set0, te_set0, tr_loader0, te_loader0 = build_task(task, seed=seeds[0], p_train=p_train, p_test=p_test,
                                                          train_fraction=train_fraction)
    overlap = check_train_test_overlap(tr_set0, te_set0)
    color_acc = color_only_baseline_acc(te_loader0)
    print(f"[sanity] train/test index overlap = {overlap} (should be 0)")
    print(f"[sanity] color-only baseline acc on test = {color_acc:.3f}")

    cfg = dict(task=task, seeds=seeds, epochs=epochs, p_train=p_train, p_test=p_test,
               train_fraction=train_fraction, lr=lr, upsample=upsample,
               eval_flip_every=eval_flip_every, eval_delta_every=eval_delta_every,
               eval_perm_every=eval_perm_every, delta_probe_frac=delta_probe_frac,
               agop_eval_batches=agop_eval_batches, perm_seed=perm_seed,
               agop_update_every=agop_update_every, agop_train_batches=agop_train_batches,
               color_only_acc=color_acc, train_test_overlap=overlap)
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    results_by_method = {
        "ERM": [],
        "JTT": [],
        "AGC-InvCFP-old": [],
        "AGC-InvCFP-new": []
    }

    for s in seeds:
        set_seed(s)
        tr_set, te_set, tr_loader, te_loader = build_task(task, seed=s, p_train=p_train, p_test=p_test,
                                                          train_fraction=train_fraction)

        # ERM
        model = CNNFeatSmall()
        erm_model, erm_logs = train_erm(model, tr_loader, te_loader, te_set,
                                        epochs=epochs, lr=lr,
                                        eval_flip_every=eval_flip_every,
                                        eval_delta_every=eval_delta_every,
                                        eval_perm_every=eval_perm_every,
                                        delta_probe_frac=delta_probe_frac,
                                        agop_eval_batches=agop_eval_batches,
                                        perm_seed=perm_seed)
        write_csv(os.path.join(outdir, f"ERM_seed{s}.csv"), erm_logs)
        results_by_method["ERM"].append(erm_logs)

        # JTT
        jtt_model, jtt_logs = train_jtt(tr_set, te_loader, te_set,
                                        total_epochs=epochs, stage1_epochs=max(1,epochs//6),
                                        upsample=upsample, lr=lr,
                                        eval_flip_every=eval_flip_every,
                                        eval_delta_every=eval_delta_every,
                                        eval_perm_every=eval_perm_every,
                                        delta_probe_frac=delta_probe_frac,
                                        agop_eval_batches=agop_eval_batches,
                                        perm_seed=perm_seed)
        write_csv(os.path.join(outdir, f"JTT_seed{s}.csv"), jtt_logs)
        results_by_method["JTT"].append(jtt_logs)

        # AGC-InvCFP-old
        agc_old_model, agc_old_logs = train_agc_invcfp_old(tr_set, te_loader, te_set,
                                                           total_epochs=epochs, lr=lr,
                                                           keep_start=0.3, keep_end=0.9, k_desired=2,
                                                           probe_frac=0.6, lambda_cons=0.2,
                                                           eval_flip_every=eval_flip_every,
                                                           eval_delta_every=eval_delta_every,
                                                           eval_perm_every=eval_perm_every,
                                                           delta_probe_frac=delta_probe_frac,
                                                           agop_eval_batches=agop_eval_batches,
                                                           perm_seed=perm_seed)
        write_csv(os.path.join(outdir, f"AGC_InvCFP_old_seed{s}.csv"), agc_old_logs)
        results_by_method["AGC-InvCFP-old"].append(agc_old_logs)

        # AGC-InvCFP-new
        agc_new_model, agc_new_logs = train_agc_invcfp_new(tr_set, te_loader, te_set,
                                                           total_epochs=epochs, lr=lr,
                                                           keep_start=0.3, keep_end=0.9, k_desired=2,
                                                           probe_frac=0.6, lambda_cons=0.2,
                                                           agop_update_every=agop_update_every,
                                                           agop_train_batches=agop_train_batches,
                                                           eval_flip_every=eval_flip_every,
                                                           eval_delta_every=eval_delta_every,
                                                           eval_perm_every=eval_perm_every,
                                                           delta_probe_frac=delta_probe_frac,
                                                           agop_eval_batches=agop_eval_batches,
                                                           perm_seed=perm_seed,
                                                           tap="pre_proj2")
        write_csv(os.path.join(outdir, f"AGC_InvCFP_new_seed{s}.csv"), agc_new_logs)
        results_by_method["AGC-InvCFP-new"].append(agc_new_logs)

    # Aggregation & plots
    agg = {}
    for method, seed_logs in results_by_method.items():
        agg[method] = {
            "te_acc":       aggregate_across_seeds(seed_logs, metric="te_acc"),
            "worst_group_acc": aggregate_across_seeds(seed_logs, metric="worst_group_acc"),
            "H":            aggregate_across_seeds(seed_logs, metric="H"),
            "alpha":        aggregate_across_seeds(seed_logs, metric="alpha"),
            "beta":         aggregate_across_seeds(seed_logs, metric="beta"),
            "flip_acc":     aggregate_across_seeds(seed_logs, metric="flip_acc"),
            "deltaU_mean":  aggregate_across_seeds(seed_logs, metric="deltaU_mean"),
            "perm_acc":     aggregate_across_seeds(seed_logs, metric="perm_acc"),
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
    plot_metric_curves(agg, "flip_acc", os.path.join(outdir, "curve_flip_acc.png"),
                       f"{task}: Flip-Acc (mean±95% CI)")
    plot_metric_curves(agg, "deltaU_mean", os.path.join(outdir, "curve_deltaU_mean.png"),
                       f"{task}: Δ_U mean on test (lower is better)")
    plot_metric_curves(agg, "perm_acc", os.path.join(outdir, "curve_perm_acc.png"),
                       f"{task}: Permuted-label Acc (should be ~0.5)")

    plot_bars_best(results_by_method, "te_acc", os.path.join(outdir, "best_te_acc.png"),
                   f"{task}: Best Test Acc across seeds")
    plot_bars_best(results_by_method, "worst_group_acc", os.path.join(outdir, "best_worst_group.png"),
                   f"{task}: Best Worst-group Acc across seeds")

    print(f"[DONE] Results saved under: {outdir}")

# -----------------------
# CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["colored_mnist", "colored_fmnist", "colored_kmnist", "colored_cifar10"],
                        help="Which tasks to run")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs per method")
    parser.add_argument("--p-train", type=float, default=0.99, help="Train correlation P[c==y]")
    parser.add_argument("--p-test", type=float, default=0.1, help="Test correlation P[c==y]")
    parser.add_argument("--train-fraction", type=float, default=1.0, help="Subsample fraction of training set")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--upsample", type=int, default=10, help="JTT upweight factor")

    parser.add_argument("--eval-flip-every", type=int, default=5, help="Eval flip-acc every K epochs (1 = every epoch, 0 = never)")
    parser.add_argument("--eval-delta-every", type=int, default=5, help="Eval Δ_U mean every K epochs (1 = every epoch, 0 = never)")
    parser.add_argument("--eval-perm-every", type=int, default=5, help="Eval permuted-label acc every K epochs (1 = every epoch, 0 = never)")
    parser.add_argument("--delta-probe-frac", type=float, default=0.5, help="Fraction for Δ_U test probing (0<frac<=1)")
    parser.add_argument("--agop-eval-batches", type=int, default=2, help="Batches used to estimate AGOP at eval")
    parser.add_argument("--perm-seed", type=int, default=0)

    parser.add_argument("--agop-update-every", type=int, default=1, help="Update AGOP U every K epochs in AGC-new")
    parser.add_argument("--agop-train-batches", type=int, default=3, help="Batches to estimate AGOP at train (AGC-new)")

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
                train_fraction=args.train_fraction, lr=args.lr, upsample=args.upsample,
                eval_flip_every=args.eval_flip_every, eval_delta_every=args.eval_delta_every,
                eval_perm_every=args.eval_perm_every,
                delta_probe_frac=args.delta_probe_frac, agop_eval_batches=args.agop_eval_batches,
                perm_seed=args.perm_seed,
                agop_update_every=args.agop_update_every, agop_train_batches=args.agop_train_batches)

if __name__ == "__main__":
    main()
