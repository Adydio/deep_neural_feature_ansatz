# -*- coding: utf-8 -*-
"""
AGOP-aware Curriculum & Stability:
- ERM, JTT, AGC-InvCFP (OLD/NEW), and NEW: AGOP-SC (self-consistent AGOP, no prior)
- Multi-seed, multi-task runner with plots, and "spur vs nospur" (p_train/p_test) comparison.
- Saves CSV/PNG under: experiments/curriculum/<task>/<timestamp>/<spur|nospur>/

Key new metrics per epoch:
  * subspace_gap: ||P1 - P2||_F^2  (two weak views, AGOP top-k projectors)
  * H_AGOP: spectral entropy of AGOP (from the same eval batches)
  * SUS: flip_acc_joint - te_acc   (higher => more spurious reliance)
  * worstG_at_Nmin: worst group acc among groups with count >= Nmin

Notes:
- For colored_* datasets we have a label-preserving "flip" (color invert).
- "nospur" sets p_train=p_test=0.5 (independent), flip remains label-preserving but SUS~0 expected.
- AGOP-SC training uses a first-order surrogate:
    loss = CE(view1)+CE(view2) + lam_sc * || U_bar^T h1 - U_bar^T h2 ||^2
  where U_bar is the top-k AGOP basis of the *current batch* averaged over the two views,
  computed from gradients wrt features; U_bar is detached (stop-grad) to keep first-order.
"""
import os, math, random, argparse, time, json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets
import matplotlib.pyplot as plt

# -----------------------
# Global utils
# -----------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# -----------------------
# Colored binary datasets with label-preserving "flip" (color invert)
# -----------------------
class ColoredBinaryBase(Dataset):
    """
    Base wrapper: takes a base dataset (PIL image, raw_label), maps to binary y and spurious color c,
    where P[c==y] = p_corr. Colors are cached for reproducibility.
    Provides counterfactual flip for each index: invert color channels (R<->G choice).
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
    def __init__(self, root, split="train", p_train=0.99, p_test=0.1, seed=0, download=True):
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
            nn.Linear(256, d_feat), nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(d_feat, 2)

    def forward(self, x, return_feat=False):
        z = self.feat(x)
        logits = self.classifier(z)
        if return_feat:
            return logits, z
        return logits

# -----------------------
# Weak, label-stable augmentations (tensor space; differentiable)
# -----------------------
def weak_augment(x: torch.Tensor, strength: float = 0.05) -> torch.Tensor:
    """
    Simple per-sample brightness/contrast jitter + small Gaussian noise.
    Assumes x in [0,1]; returns clamped [0,1].
    """
    B = x.size(0)
    device = x.device
    scale = (1.0 + (torch.rand(B,1,1,1, device=device)*2-1.0)*strength)
    bias  = ((torch.rand(B,1,1,1, device=device)*2-1.0)*strength)
    x_aug = x * scale + bias
    if strength > 0:
        x_aug = x_aug + torch.randn_like(x_aug) * (strength*0.2)
    return torch.clamp(x_aug, 0.0, 1.0)

# -----------------------
# AGOP utilities (feature-tap gradient covariance)
# -----------------------
def _agop_grad_cov_from_batch(z: torch.Tensor, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute batch AGOP G = (1/B) g^T g, where g = d CE(logits,y) / d z."""
    ce = F.cross_entropy(logits, y, reduction="mean")
    g = torch.autograd.grad(ce, z, retain_graph=True, create_graph=False)[0]  # [B, D]
    g = g.detach()
    G = (g.T @ g) / (g.size(0) + 1e-8)  # [D, D]
    return G

@torch.no_grad()
def _topk_projector_from_G(G: torch.Tensor, k: int) -> torch.Tensor:
    """Return projector P = U U^T with U = top-k eigenvectors of symmetric G."""
    evals, evecs = torch.linalg.eigh(G)
    U = evecs[:, -max(1,k):]  # [D,k]
    P = U @ U.T
    return P

@torch.no_grad()
def _U_from_G(G: torch.Tensor, k: int) -> torch.Tensor:
    evals, evecs = torch.linalg.eigh(G)
    U = evecs[:, -max(1,k):]  # [D,k]
    return U

@torch.no_grad()
def agop_spectral_entropy(G: torch.Tensor) -> float:
    evals = torch.linalg.eigvalsh(G).clamp(min=1e-12)
    p = (evals / evals.sum()).cpu().numpy()
    return float(-(p * np.log(p)).sum())

@torch.no_grad()
def estimate_agop_topk(model: nn.Module, loader: DataLoader, k: int = 2, max_batches: int = 4) -> torch.Tensor:
    """
    Estimate top-k AGOP basis U on a small number of batches from 'loader' at the feature tap.
    Returns U: [D, k] (on CPU).
    """
    model.eval()
    G_sum = None
    n = 0
    it = 0
    for x, y, c, idx in loader:
        it += 1
        if it > max_batches: break
        x, y = x.to(DEVICE), y.to(DEVICE)
        x.requires_grad_(False)
        logits, z = model(x, return_feat=True)  # z will be part of graph
        G = _agop_grad_cov_from_batch(z, logits, y)
        G = G.detach().cpu()
        G_sum = G if G_sum is None else (G_sum + G)
        n += 1
    if G_sum is None:  # fallback
        dummy = torch.eye(model.classifier.in_features, dtype=torch.float32)
        return dummy[:, :max(1,k)].contiguous()
    G_mean = G_sum / max(n,1)
    return _U_from_G(G_mean, k)

# -----------------------
# NFA proxy via classifier (kept for OLD variant)
# -----------------------
@torch.no_grad()
def topk_basis_from_classifier(model: nn.Module, k_desired=2, eps=1e-8) -> torch.Tensor:
    """
    Use W^T W to approximate NFA subspace (weight proxy). Returns U : [d, k].
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
def spectral_entropy_W(model: nn.Module) -> float:
    W = model.classifier.weight.detach()
    M = W.T @ W
    evals = torch.linalg.eigvalsh(M).clamp(min=1e-12)
    p = (evals / evals.sum()).cpu().numpy()
    return float(-(p * np.log(p)).sum())

# -----------------------
# Alignment diagnostics (optional)
# -----------------------
@torch.no_grad()
def direction_alignment(model: nn.Module, loader: DataLoader) -> Tuple[float,float]:
    """
    alpha: alignment to label direction; beta: alignment to color direction.
    Computed on CPU to avoid device mismatch (uses feature means).
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
# Evaluation helpers
# -----------------------
@torch.no_grad()
def eval_avg_loss_acc(model, loader, criterion):
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    for x, y, c, idx in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = criterion(logits, y)
        pred = logits.argmax(1)
        total += y.size(0)
        correct += (pred==y).sum().item()
        total_loss += float(loss.item()) * y.size(0)
    return total_loss/max(total,1), correct/max(total,1)

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
    return worst, avg, accs, stats

@torch.no_grad()
def eval_worst_group_at_Nmin(model, loader, Nmin:int=50):
    model.eval()
    _, _, accs, stats = eval_worst_group(model, loader)
    worst = 1.0
    found = False
    for k, v in stats.items():
        n = v[1]
        if n >= Nmin:
            found = True
            worst = min(worst, accs[k])
    if not found:
        return float('nan')
    return worst

@torch.no_grad()
def eval_flip_acc_and_SUS(model, loader, dataset) -> Tuple[float, float]:
    """
    flip-acc on test set after flipping spurious color (label unchanged), and SUS=flip-acc - te-acc.
    """
    model.eval()
    total = 0; correct_te = 0; correct_flip = 0
    for x, y, c, idx in loader:
        x = x.to(DEVICE); y = y.to(DEVICE)
        # original
        pred = model(x).argmax(1)
        correct_te += (pred==y).sum().item()
        # flip
        idx_list = idx.tolist()
        x_flip = torch.stack([dataset.get_flip_tensor(int(i)) for i in idx_list], dim=0).to(DEVICE)
        pred_f = model(x_flip).argmax(1)
        correct_flip += (pred_f==y).sum().item()
        total += y.size(0)
    te_acc = correct_te/max(total,1)
    flip_acc = correct_flip/max(total,1)
    sus = flip_acc - te_acc
    return flip_acc, sus

@torch.no_grad()
def eval_agop_subspace_gap_and_entropy(model, loader, k:int=2, max_batches:int=3) -> Tuple[float, float]:
    """
    On a few batches of 'loader':
      - build two weak views per batch, compute AGOP G1/G2 at feature tap
      - subspace_gap = mean ||P1-P2||_F^2
      - H_AGOP = spectral entropy of average G
    """
    model.eval()
    gap_sum = 0.0; count = 0
    G_sum = None; nG = 0
    it = 0
    for x, y, c, idx in loader:
        it += 1
        if it > max_batches: break
        x, y = x.to(DEVICE), y.to(DEVICE)
        x1 = weak_augment(x, 0.05)
        x2 = weak_augment(x, 0.05)
        # view1
        logits1, z1 = model(x1, return_feat=True)
        G1 = _agop_grad_cov_from_batch(z1, logits1, y)
        # view2
        logits2, z2 = model(x2, return_feat=True)
        G2 = _agop_grad_cov_from_batch(z2, logits2, y)
        # projectors
        P1 = _topk_projector_from_G(G1, k).cpu()
        P2 = _topk_projector_from_G(G2, k).cpu()
        gap = torch.norm(P1 - P2, p='fro').item()**2
        gap_sum += gap; count += 1
        # entropy on average G
        Gm = ((G1 + G2)/2).detach().cpu()
        G_sum = Gm if G_sum is None else (G_sum + Gm); nG += 1
    gap_avg = gap_sum/max(count,1)
    H_agop = agop_spectral_entropy(G_sum/max(nG,1)) if G_sum is not None else float('nan')
    return gap_avg, H_agop

# -----------------------
# Train / Eval generic epoch
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
    return total_loss/max(total,1), correct/max(total,1)

# -----------------------
# AGC-InvCFP OLD (U from classifier; ΔU from counterfactual pairs)
# -----------------------
@torch.no_grad()
def _unwrap_colored_dataset(ds: Dataset) -> ColoredBinaryBase:
    base = ds
    while isinstance(base, Subset):
        base = base.dataset
    assert isinstance(base, ColoredBinaryBase), "This method needs Colored* dataset."
    return base

@torch.no_grad()
def compute_pair_sensitivity(model: nn.Module, loader: DataLoader, dataset: Dataset,
                             U: torch.Tensor) -> Dict[int, float]:
    """
    Δ_U(i) = ||U^T z - U^T z_flip||^2 / (||z||^2 + ||z_flip||^2).
    Return dict idx -> Δ_U (higher => more spurious-sensitive).
    """
    model.eval()
    U = U.to(DEVICE)
    scores = {}
    for x, y, c, idx in loader:
        idx_list = idx.tolist()
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

def train_agc_invcfp_old(train_set, test_loader, total_epochs=30, lr=3e-4,
                         keep_start=0.3, keep_end=0.9, k_desired=2,
                         lambda_cons=0.2):
    """
    OLD: U from classifier (W^T W), select top Δ_U pairs, do pair CE + full-feature MSE consistency.
    """
    dataset_base = _unwrap_colored_dataset(train_set)
    model = CNNFeatSmall().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    base_loader = DataLoader(train_set, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    logs=[]
    for ep in range(1, total_epochs+1):
        keep_ratio = min(keep_end, keep_start + (keep_end - keep_start) * (ep-1)/(total_epochs-1))
        U = topk_basis_from_classifier(model, k_desired=k_desired)
        delta = compute_pair_sensitivity(model, base_loader, dataset_base, U)
        n = len(delta); m = max(1, int(n*keep_ratio))
        # top-m + 10% random
        top_idx = [i for i,_ in sorted(delta.items(), key=lambda kv: kv[1], reverse=True)[:m]]
        remain = list(set(delta.keys()) - set(top_idx))
        if len(remain) > 0:
            top_idx += random.sample(remain, min(int(0.1*n), len(remain)))
        # subset
        if isinstance(train_set, Subset):
            base_to_pos = {int(b): i for i, b in enumerate(train_set.indices)}
            pos = [base_to_pos[i] for i in top_idx if int(i) in base_to_pos]
            subset = Subset(train_set, pos if pos else list(range(len(train_set))))
        else:
            subset = Subset(train_set, top_idx)

        train_loader = DataLoader(subset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)

        # one epoch train
        model.train()
        total, correct = 0, 0
        for x, y, c, idx in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x_flip = torch.stack([dataset_base.get_flip_tensor(int(i)) for i in idx.tolist()], dim=0).to(DEVICE)
            logits, z    = model(x, return_feat=True)
            logits_f, zf = model(x_flip, return_feat=True)
            ce = crit(logits, y) + crit(logits_f, y)
            cons = F.mse_loss(z, zf)
            loss = ce + lambda_cons * cons
            opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step()
            pred = logits.argmax(1)
            total += y.size(0); correct += (pred==y).sum().item()

        # eval
        te_loss, te_acc = eval_avg_loss_acc(model, test_loader, crit)
        wg, _, _, _ = eval_worst_group(model, test_loader)
        # AGOP diagnostics on test
        gap, H_agop = eval_agop_subspace_gap_and_entropy(model, test_loader, k=k_desired, max_batches=3)
        # SUS
        flip_acc, sus = eval_flip_acc_and_SUS(model, test_loader, _unwrap_colored_dataset(test_loader.dataset))
        logs.append({"epoch": ep, "keep_ratio": keep_ratio,
                     "tr_acc": correct/max(total,1), "te_acc": te_acc,
                     "worst_group_acc": wg, "subspace_gap": gap, "H_AGOP": H_agop,
                     "SUS": sus})
        print(f"[AGC_INV-CFP_OLD] ep{ep:02d} keep={keep_ratio:.2f} te_acc={te_acc:.3f} worstG={wg:.3f} "
              f"gap={gap:.3f} H_AGOP={H_agop:.3f} SUS={sus:.3f}")
    return model, logs

# -----------------------
# AGC-InvCFP NEW (U from AGOP gradients; pair consistency on projection U)
# -----------------------
def train_agc_invcfp_new(train_set, test_loader, total_epochs=30, lr=3e-4,
                         keep_start=0.3, keep_end=0.9, k_agop=2,
                         lambda_cons=0.2, agop_eval_batches=3):
    """
    NEW: U from AGOP (grad wrt features). Select top Δ_U (using this U).
         Pair CE + *projection* consistency on U (not full-feature).
    """
    dataset_base = _unwrap_colored_dataset(train_set)
    model = CNNFeatSmall().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    base_loader_eval = DataLoader(train_set, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    logs=[]
    for ep in range(1, total_epochs+1):
        keep_ratio = min(keep_end, keep_start + (keep_end - keep_start) * (ep-1)/(total_epochs-1))
        # U from AGOP (on a few batches)
        U = estimate_agop_topk(model, base_loader_eval, k=k_agop, max_batches=agop_eval_batches).to(DEVICE)
        # Δ_U scores
        scores = {}
        for x, y, c, idx in base_loader_eval:
            idx_list = idx.tolist()
            x = x.to(DEVICE)
            x_flip = torch.stack([dataset_base.get_flip_tensor(int(i)) for i in idx_list], dim=0).to(DEVICE)
            _, z  = model(x, return_feat=True)
            _, zf = model(x_flip, return_feat=True)
            z  = F.normalize(z, dim=1); zf = F.normalize(zf, dim=1)
            p  = z @ U; pf = zf @ U
            num = ((p - pf)**2).sum(dim=1)
            den = (z.pow(2).sum(dim=1) + zf.pow(2).sum(dim=1) + 1e-8)
            delta = (num / den).detach().cpu().numpy()
            for i, sid in enumerate(idx_list):
                scores[sid] = float(delta[i])
        n = len(scores); m = max(1, int(n*keep_ratio))
        top_idx = [i for i,_ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:m]]
        remain = list(set(scores.keys()) - set(top_idx))
        if len(remain) > 0:
            top_idx += random.sample(remain, min(int(0.1*n), len(remain)))

        # subset mapping
        if isinstance(train_set, Subset):
            base_to_pos = {int(b): i for i, b in enumerate(train_set.indices)}
            pos = [base_to_pos[i] for i in top_idx if int(i) in base_to_pos]
            subset = Subset(train_set, pos if pos else list(range(len(train_set))))
        else:
            subset = Subset(train_set, top_idx)
        train_loader = DataLoader(subset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)

        # one epoch (projection consistency on U)
        model.train()
        total, correct = 0, 0
        for x, y, c, idx in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x_flip = torch.stack([dataset_base.get_flip_tensor(int(i)) for i in idx.tolist()], dim=0).to(DEVICE)
            logits, z    = model(x, return_feat=True)
            logits_f, zf = model(x_flip, return_feat=True)
            ce = crit(logits, y) + crit(logits_f, y)
            # projection consistency
            z_n  = F.normalize(z,  dim=1)
            zf_n = F.normalize(zf, dim=1)
            p, pf = z_n @ U, zf_n @ U
            cons = F.mse_loss(p, pf)
            loss = ce + lambda_cons * cons
            opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step()
            pred = logits.argmax(1)
            total += y.size(0); correct += (pred==y).sum().item()

        # eval
        te_loss, te_acc = eval_avg_loss_acc(model, test_loader, crit)
        wg, _, _, _ = eval_worst_group(model, test_loader)
        gap, H_agop = eval_agop_subspace_gap_and_entropy(model, test_loader, k=k_agop, max_batches=agop_eval_batches)
        flip_acc, sus = eval_flip_acc_and_SUS(model, test_loader, _unwrap_colored_dataset(test_loader.dataset))
        logs.append({"epoch": ep, "keep_ratio": keep_ratio,
                     "tr_acc": correct/max(total,1), "te_acc": te_acc,
                     "worst_group_acc": wg, "subspace_gap": gap, "H_AGOP": H_agop,
                     "SUS": sus})
        print(f"[AGC_INV-CFP_NEW] ep{ep:02d} keep={keep_ratio:.2f} te_acc={te_acc:.3f} worstG={wg:.3f} "
              f"gap={gap:.3f} H_AGOP={H_agop:.3f} SUS={sus:.3f}")
    return model, logs

# -----------------------
# NEW: AGOP-SC (no prior, self-consistent AGOP across weak views)
# -----------------------
def train_agop_sc(train_set, test_loader, total_epochs=30, lr=3e-4,
                  k_agop:int=2, lam_sc:float=0.3, agop_eval_batches:int=3):
    """
    For each minibatch, create two weak views (x1,x2).
    Loss = CE1 + CE2 + lam_sc * || U_bar^T h1 - U_bar^T h2 ||^2,
      where U_bar is top-k eigenvectors of (G1+G2)/2 with
      Gv = E[(dCE/dh_v)(dCE/dh_v)^T], computed per-batch at feature tap, then detached.
    """
    model = CNNFeatSmall().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    base_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)

    logs=[]
    for ep in range(1, total_epochs+1):
        model.train()
        total, correct = 0, 0
        for x, y, c, idx in base_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x1 = weak_augment(x, 0.05)
            x2 = weak_augment(x, 0.05)
            # forward both views
            logits1, z1 = model(x1, return_feat=True)
            logits2, z2 = model(x2, return_feat=True)
            ce = crit(logits1, y) + crit(logits2, y)
            # AGOP per view (detached)
            G1 = _agop_grad_cov_from_batch(z1, logits1, y)
            G2 = _agop_grad_cov_from_batch(z2, logits2, y)
            Gm = ((G1 + G2)/2).detach()
            U_bar = _U_from_G(Gm, k_agop).to(DEVICE)  # stop-gradient basis
            # projection consistency
            z1n = F.normalize(z1, dim=1)
            z2n = F.normalize(z2, dim=1)
            p1 = z1n @ U_bar
            p2 = z2n @ U_bar
            stab = F.mse_loss(p1, p2)
            loss = ce + lam_sc * stab
            opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step()
            pred = logits1.argmax(1)
            total += y.size(0); correct += (pred==y).sum().item()

        # eval (every epoch)
        te_loss, te_acc = eval_avg_loss_acc(model, test_loader, crit)
        wg, _, _, _ = eval_worst_group(model, test_loader)
        gap, H_agop = eval_agop_subspace_gap_and_entropy(model, test_loader, k=k_agop, max_batches=agop_eval_batches)
        flip_acc, sus = eval_flip_acc_and_SUS(model, test_loader, _unwrap_colored_dataset(test_loader.dataset))
        logs.append({"epoch": ep, "tr_acc": correct/max(total,1), "te_acc": te_acc,
                     "worst_group_acc": wg, "subspace_gap": gap, "H_AGOP": H_agop,
                     "SUS": sus})
        print(f"[AGOP-SC] ep{ep:02d} te_acc={te_acc:.3f} worstG={wg:.3f} gap={gap:.3f} H_AGOP={H_agop:.3f} SUS={sus:.3f}")
    return model, logs

# -----------------------
# ERM / JTT (baseline)
# -----------------------
def train_erm(model, train_loader, test_loader, te_set,
              epochs=30, lr=3e-4, wd=1e-4, k_agop:int=2, agop_eval_batches:int=3, nmin:int=50):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()
    logs=[]
    for ep in range(1, epochs+1):
        tr_loss, tr_acc = run_epoch(model, train_loader, opt, crit, train=True)
        te_loss, te_acc = eval_avg_loss_acc(model, test_loader, crit)
        wg, _, _, _ = eval_worst_group(model, test_loader)
        wgN = eval_worst_group_at_Nmin(model, test_loader, nmin)
        # diagnostics
        gap, H_agop = eval_agop_subspace_gap_and_entropy(model, test_loader, k=k_agop, max_batches=agop_eval_batches)
        flip_acc, sus = eval_flip_acc_and_SUS(model, test_loader, _unwrap_colored_dataset(te_set))
        logs.append({"epoch": ep, "tr_acc": tr_acc, "te_acc": te_acc,
                     "worst_group_acc": wg, "worstG_at_Nmin": wgN,
                     "subspace_gap": gap, "H_AGOP": H_agop, "SUS": sus})
        print(f"[ERM] ep{ep:02d} te_acc={te_acc:.3f} worstG={wg:.3f} worstG@{nmin}={wgN:.3f if not np.isnan(wgN) else float('nan'):.3f} "
              f"gap={gap:.3f} H_AGOP={H_agop:.3f} SUS={sus:.3f}")
    return model, logs

def train_jtt(train_set, test_loader, total_epochs=30, stage1_epochs=5, upsample=10,
              lr=3e-4, k_agop:int=2, agop_eval_batches:int=3, nmin:int=50):
    base = CNNFeatSmall().to(DEVICE)
    opt = torch.optim.AdamW(base.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    base_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
    logs=[]
    # stage1
    for ep in range(1, stage1_epochs+1):
        run_epoch(base, base_loader, opt, crit, train=True)
        te_loss, te_acc = eval_avg_loss_acc(base, test_loader, crit)
        wg, _, _, _ = eval_worst_group(base, test_loader)
        gap, H_agop = eval_agop_subspace_gap_and_entropy(base, test_loader, k=k_agop, max_batches=agop_eval_batches)
        flip_acc, sus = eval_flip_acc_and_SUS(base, test_loader, _unwrap_colored_dataset(test_loader.dataset))
        logs.append({"epoch": ep, "te_acc": te_acc, "worst_group_acc": wg,
                     "subspace_gap": gap, "H_AGOP": H_agop, "SUS": sus})
        print(f"[JTT-Stage1] ep{ep:02d} te_acc={te_acc:.3f} worstG={wg:.3f}")

    # find misclassified
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
        te_loss, te_acc = eval_avg_loss_acc(model, test_loader, crit)
        wg, _, _, _ = eval_worst_group(model, test_loader)
        wgN = eval_worst_group_at_Nmin(model, test_loader, nmin)
        gap, H_agop = eval_agop_subspace_gap_and_entropy(model, test_loader, k=k_agop, max_batches=agop_eval_batches)
        flip_acc, sus = eval_flip_acc_and_SUS(model, test_loader, _unwrap_colored_dataset(test_loader.dataset))
        logs.append({"epoch": ep, "tr_acc": tr_acc, "te_acc": te_acc,
                     "worst_group_acc": wg, "worstG_at_Nmin": wgN,
                     "subspace_gap": gap, "H_AGOP": H_agop, "SUS": sus})
        print(f"[JTT] ep{ep:02d} te_acc={te_acc:.3f} worstG={wg:.3f} gap={gap:.3f} H_AGOP={H_agop:.3f} SUS={sus:.3f}")
    return model, logs

# -----------------------
# Task builder (+ spur vs nospur)
# -----------------------
def build_task(task: str, root="./data", seed=0,
               p_train=0.99, p_test=0.1, train_fraction=1.0, download=True):
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
        tr = ColoredCIFAR10(root, "train", p_train=p_train, p_test=p_test, seed=seed, download=download)
        te = ColoredCIFAR10(root, "test",  p_train=p_train, p_test=p_test, seed=seed+1, download=download)
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

@torch.no_grad()
def sanity_prints(train_set, test_set):
    # index overlap check (subset-safe)
    tr_idx = []
    if isinstance(train_set, Subset): tr_idx = list(map(int, train_set.indices))
    else: tr_idx = list(range(len(train_set)))
    te_idx = list(range(len(test_set)))
    overlap = len(set(tr_idx).intersection(set(te_idx)))
    print(f"[sanity] train/test index overlap = {overlap} (should be 0)")
    # color-only baseline on test
    labels = []; colors=[]
    base_te = _unwrap_colored_dataset(test_set)
    for i in range(len(test_set)):
        if isinstance(test_set, Subset):
            bi = int(test_set.indices[i])
        else:
            bi = i
        labels.append(int(base_te.labels[bi])); colors.append(int(base_te.colors[bi]))
    labels = np.array(labels); colors = np.array(colors)
    color_only_acc = float(np.mean(labels==colors))
    print(f"[sanity] color-only baseline acc on test = {color_only_acc:.3f}")
    # group counts
    counts={}
    for i in range(len(test_set)):
        if isinstance(test_set, Subset):
            bi = int(test_set.indices[i])
        else:
            bi = i
        y = int(base_te.labels[bi]); c = int(base_te.colors[bi])
        counts[(y,c)] = counts.get((y,c),0)+1
    print(f"[test group counts] {counts}")

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
    if not results_per_seed: return None
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
        if d.get(metric_name) is None: continue
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
            # for SUS we'd prefer min (lower is better); keep using max for acc and use min for SUS:
            if metric.lower() == "sus":
                arr = [(-v if np.isfinite(v) else float("nan")) for v in arr]  # invert to use 'max'
            bests.append(np.nanmax(arr))
        # invert back if SUS
        if metric.lower() == "sus":
            bests = [(-v if np.isfinite(v) else float("nan")) for v in bests]
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
# Runner (one spur-mode)
# -----------------------
def run_all(task: str, seeds: List[int], outdir: str,
            epochs=30, lr=3e-4, upsample=10,
            p_train=0.99, p_test=0.1, train_fraction=1.0,
            k_agop:int=2, lam_sc:float=0.3, agop_eval_batches:int=3, nmin:int=50):
    ensure_dir(outdir)
    cfg = dict(task=task, seeds=seeds, epochs=epochs, p_train=p_train, p_test=p_test,
               train_fraction=train_fraction, lr=lr, upsample=upsample,
               k_agop=k_agop, lam_sc=lam_sc, agop_eval_batches=agop_eval_batches, nmin=nmin)
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    results_by_method = {
        "ERM": [],
        "JTT": [],
        "AGC-InvCFP_OLD": [],
        "AGC-InvCFP_NEW": [],
        "AGOP-SC": []
    }

    for s in seeds:
        set_seed(s)
        tr_set, te_set, tr_loader, te_loader = build_task(task, seed=s,
                                                          p_train=p_train, p_test=p_test,
                                                          train_fraction=train_fraction)
        sanity_prints(tr_set, te_set)

        # ERM
        model = CNNFeatSmall()
        erm_model, erm_logs = train_erm(model, tr_loader, te_loader, te_set,
                                        epochs=epochs, lr=lr, k_agop=k_agop,
                                        agop_eval_batches=agop_eval_batches, nmin=nmin)
        write_csv(os.path.join(outdir, f"ERM_seed{s}.csv"), erm_logs)
        results_by_method["ERM"].append(erm_logs)

        # JTT
        jtt_model, jtt_logs = train_jtt(tr_set, te_loader, total_epochs=epochs,
                                        stage1_epochs=max(1,epochs//6),
                                        upsample=upsample, lr=lr, k_agop=k_agop,
                                        agop_eval_batches=agop_eval_batches, nmin=nmin)
        write_csv(os.path.join(outdir, f"JTT_seed{s}.csv"), jtt_logs)
        results_by_method["JTT"].append(jtt_logs)

        # AGC-InvCFP OLD
        agc_old_model, agc_old_logs = train_agc_invcfp_old(tr_set, te_loader, total_epochs=epochs,
                                                           lr=lr, keep_start=0.3, keep_end=0.9,
                                                           k_desired=k_agop, lambda_cons=0.2)
        write_csv(os.path.join(outdir, f"AGC_InvCFP_OLD_seed{s}.csv"), agc_old_logs)
        results_by_method["AGC-InvCFP_OLD"].append(agc_old_logs)

        # AGC-InvCFP NEW
        agc_new_model, agc_new_logs = train_agc_invcfp_new(tr_set, te_loader, total_epochs=epochs,
                                                           lr=lr, keep_start=0.3, keep_end=0.9,
                                                           k_agop=k_agop, lambda_cons=0.2,
                                                           agop_eval_batches=agop_eval_batches)
        write_csv(os.path.join(outdir, f"AGC_InvCFP_NEW_seed{s}.csv"), agc_new_logs)
        results_by_method["AGC-InvCFP_NEW"].append(agc_new_logs)

        # AGOP-SC (no prior)
        agop_sc_model, agop_sc_logs = train_agop_sc(tr_set, te_loader, total_epochs=epochs,
                                                    lr=lr, k_agop=k_agop, lam_sc=lam_sc,
                                                    agop_eval_batches=agop_eval_batches)
        write_csv(os.path.join(outdir, f"AGOP_SC_seed{s}.csv"), agop_sc_logs)
        results_by_method["AGOP-SC"].append(agop_sc_logs)

    # Aggregation & plots
    agg = {}
    for method, seed_logs in results_by_method.items():
        agg[method] = {
            "te_acc": aggregate_across_seeds(seed_logs, metric="te_acc"),
            "worst_group_acc": aggregate_across_seeds(seed_logs, metric="worst_group_acc"),
            "worstG_at_Nmin": aggregate_across_seeds(seed_logs, metric="worstG_at_Nmin"),
            "subspace_gap": aggregate_across_seeds(seed_logs, metric="subspace_gap"),
            "H_AGOP": aggregate_across_seeds(seed_logs, metric="H_AGOP"),
            "SUS": aggregate_across_seeds(seed_logs, metric="SUS"),
        }

    plot_metric_curves(agg, "te_acc", os.path.join(outdir, "curve_te_acc.png"),
                       f"{task}: Test Accuracy (mean±95% CI)")
    plot_metric_curves(agg, "worst_group_acc", os.path.join(outdir, "curve_worst_group.png"),
                       f"{task}: Worst-group Acc (mean±95% CI)")
    plot_metric_curves(agg, "worstG_at_Nmin", os.path.join(outdir, "curve_worstG_at_Nmin.png"),
                       f"{task}: Worst-group@Nmin (mean±95% CI)")
    plot_metric_curves(agg, "subspace_gap", os.path.join(outdir, "curve_subspace_gap.png"),
                       f"{task}: Subspace-Gap (mean±95% CI)")
    plot_metric_curves(agg, "H_AGOP", os.path.join(outdir, "curve_H_AGOP.png"),
                       f"{task}: AGOP Spectral Entropy (mean±95% CI)")
    plot_metric_curves(agg, "SUS", os.path.join(outdir, "curve_SUS.png"),
                       f"{task}: SUS=flip_acc_joint - te_acc (mean±95% CI)")

    plot_bars_best(results_by_method, "te_acc", os.path.join(outdir, "best_te_acc.png"),
                   f"{task}: Best Test Acc across seeds")
    plot_bars_best(results_by_method, "worst_group_acc", os.path.join(outdir, "best_worst_group.png"),
                   f"{task}: Best Worst-group Acc across seeds")
    plot_bars_best(results_by_method, "SUS", os.path.join(outdir, "best_SUS.png"),
                   f"{task}: Best (lowest) SUS across seeds")

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
    parser.add_argument("--p-train", type=float, default=0.99, help="Train correlation P[c==y] for spur-mode")
    parser.add_argument("--p-test", type=float, default=0.1, help="Test correlation P[c==y] for spur-mode")
    parser.add_argument("--train-fraction", type=float, default=1.0, help="Subsample fraction of training set")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--upsample", type=int, default=10, help="JTT upweight factor")
    parser.add_argument("--out-root", type=str, default="experiments/curriculum")
    parser.add_argument("--compare-nospur", action="store_true",
                        help="Also run nospur setting (p_train=p_test=0.5) for each task")
    parser.add_argument("--agop-k", type=int, default=2, help="Top-k eigenvectors for AGOP subspace")
    parser.add_argument("--lam-sc", type=float, default=0.3, help="Lambda for AGOP-SC projection consistency")
    parser.add_argument("--agop-eval-batches", type=int, default=3, help="Batches used to estimate AGOP stats (eval)")
    parser.add_argument("--nmin", type=int, default=50, help="Nmin for worstG@Nmin metric")
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for task in args.tasks:
        # Spur mode
        outdir_spur = os.path.join(args.out_root, task, timestamp, "spur")
        ensure_dir(outdir_spur)
        print(f"\n=== Task: {task} | seeds={seeds} | epochs={args.epochs} | mode=spur ===")
        run_all(task=task, seeds=seeds, outdir=outdir_spur,
                epochs=args.epochs, p_train=args.p_train, p_test=args.p_test,
                train_fraction=args.train_fraction, lr=args.lr, upsample=args.upsample,
                k_agop=args.agop_k, lam_sc=args.lam_sc,
                agop_eval_batches=args.agop_eval_batches, nmin=args.nmin)

        if args.compare_nospur:
            outdir_ns = os.path.join(args.out_root, task, timestamp, "nospur")
            ensure_dir(outdir_ns)
            print(f"\n=== Task: {task} | seeds={seeds} | epochs={args.epochs} | mode=nospur ===")
            # nospur: independent (0.5 / 0.5)
            run_all(task=task, seeds=seeds, outdir=outdir_ns,
                    epochs=args.epochs, p_train=0.5, p_test=0.5,
                    train_fraction=args.train_fraction, lr=args.lr, upsample=args.upsample,
                    k_agop=args.agop_k, lam_sc=args.lam_sc,
                    agop_eval_batches=args.agop_eval_batches, nmin=args.nmin)

if __name__ == "__main__":
    main()
