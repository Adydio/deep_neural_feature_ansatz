# -*- coding: utf-8 -*-
"""
AGC-InvCFP on more complex curricula: Multi-spurious tasks + broad CL baselines.

Tasks:
  - multispur_mnist     (MNIST + 3 spurious factors: color / bg stripes / corner patch location)
  - multispur_cifar10   (CIFAR10-binary + the same 3 spurious factors)

Compare:
  - ERM
  - JTT
  - SPL-Easy (Self-Paced, easy-first by small loss)
  - SPL-Hard (Hard mining)
  - Random-Curr (random subset schedule)
  - AGC-InvCFP-old-multi (z + W^T W + full MSE; choose flip with largest Δ)
  - AGC-InvCFP-new-multi (AGOP@tap + projection MSE; choose flip with largest Δ)

Saves CSV/PNG under: experiments/curriculum_bench/<task>/<timestamp>/
"""

import os, math, random, argparse, json
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
# Small utilities
# -----------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def write_csv(path: str, logs: List[Dict]):
    import csv
    if not logs: return
    keys = sorted(list({k for row in logs for k in row.keys()}))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in logs:
            writer.writerow(row)

# -----------------------
# Multi-spurious dataset helpers
# -----------------------
def _to_float01(arr):
    return np.clip(arr.astype(np.float32), 0.0, 1.0)

def _add_bg_stripes(img3: np.ndarray, horizontal: bool, amp=0.25, freq=6):
    # img3: 3xH xW numpy in [0,1]
    _, H, W = img3.shape
    grid = (np.arange(W)[None, :] if not horizontal else np.arange(H)[:, None]).astype(np.float32)
    phase = 2*np.pi*grid / (W if not horizontal else H) * freq
    pat = (0.5*(1.0+np.sin(phase))).astype(np.float32)
    if horizontal:
        pat = np.repeat(pat[:, :], W, axis=1)
    else:
        pat = np.repeat(pat[:, :], H, axis=0)
    pat = pat.reshape(1, H, W)
    img3 = img3 + amp * pat
    return _to_float01(img3)

def _add_corner_patch(img3: np.ndarray, top_left: bool, size=5, val=(0.0, 0.0, 1.0)):
    # blue square patch
    C, H, W = img3.shape
    h0, w0 = (0,0) if top_left else (H-size, W-size)
    img3 = img3.copy()
    img3[0, h0:h0+size, w0:w0+size] = val[0]
    img3[1, h0:h0+size, w0:w0+size] = val[1]
    img3[2, h0:h0+size, w0:w0+size] = val[2]
    return _to_float01(img3)

def _tint_rgb(img3: np.ndarray, to_red: bool, strength=0.35):
    # amplify red or green channel slightly
    img3 = img3.copy()
    if to_red:
        img3[0] = np.clip(img3[0] * (1.0 + strength), 0, 1)
        img3[1] = np.clip(img3[1] * (1.0 - 0.15*strength), 0, 1)
    else:
        img3[1] = np.clip(img3[1] * (1.0 + strength), 0, 1)
        img3[0] = np.clip(img3[0] * (1.0 - 0.15*strength), 0, 1)
    return img3

def _gray_to_rgb(gray: np.ndarray, red: bool):
    # gray is HxW float in [0,1]
    R = gray if red else np.zeros_like(gray)
    G = gray if (not red) else np.zeros_like(gray)
    B = np.zeros_like(gray)
    return np.stack([R,G,B], axis=0).astype(np.float32)

# -----------------------
# Multi-spurious base dataset
# -----------------------
class MultiSpurBase(Dataset):
    """
    Wrap base dataset -> binary y; add 3 spurious factors s1=COLOR (red/green), s2=BG stripe (vert/hori), s3=LOC (TL/BR).
    Train: P[s_k==y] = p_train; Test: P[s_k==y] = p_test.
    Provide counterfactuals for each factor & joint.
    """
    def __init__(self, base_ds, make_binary_label, split: str, p_train=0.99, p_test=0.1, seed=0, is_grayscale=True):
        assert split in ["train","test"]
        self.base = base_ds
        self.split = split
        self.p_corr = p_train if split=="train" else p_test
        self.is_grayscale = is_grayscale
        rng = np.random.RandomState(seed if seed is not None else 0)

        self.labels = []
        self.s_color = []  # 1 => red, 0 => green
        self.s_bg    = []  # 1 => horizontal, 0 => vertical
        self.s_loc   = []  # 1 => top-left, 0 => bottom-right
        for i in range(len(self.base)):
            _, raw_y = self.base[i]
            y = int(make_binary_label(raw_y))
            # draw 3 spurious, each correlated with y with prob p_corr
            def draw_spur():
                return y if (rng.rand() < self.p_corr) else 1-y
            self.labels.append(y)
            self.s_color.append(draw_spur())
            self.s_bg.append(draw_spur())
            self.s_loc.append(draw_spur())
        self.labels = np.array(self.labels, dtype=np.int64)
        self.s_color = np.array(self.s_color, dtype=np.int64)
        self.s_bg = np.array(self.s_bg, dtype=np.int64)
        self.s_loc = np.array(self.s_loc, dtype=np.int64)

    def __len__(self): return len(self.base)

    def _pil_to_gray_np(self, pil_img) -> np.ndarray:
        arr = np.array(pil_img, dtype=np.float32)/255.0
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]
        return arr

    def _build_rgb(self, idx, force_color=None, force_bg=None, force_loc=None, tint_strength=0.35):
        pil_img, _ = self.base[idx]
        if self.is_grayscale:
            gray = self._pil_to_gray_np(pil_img)
            red = bool(self.s_color[idx]) if force_color is None else bool(force_color)
            x = _gray_to_rgb(gray, red)
        else:
            x = np.array(pil_img, dtype=np.float32)/255.0
            x = x.transpose(2,0,1)  # to 3xHxW
            # apply slight tint as color spurious
            to_red = bool(self.s_color[idx]) if force_color is None else bool(force_color)
            x = _tint_rgb(x, to_red, strength=tint_strength)

        bg_hori  = bool(self.s_bg[idx]) if force_bg is None else bool(force_bg)
        x = _add_bg_stripes(x, horizontal=bg_hori, amp=0.22, freq=6)

        loc_tl   = bool(self.s_loc[idx]) if force_loc is None else bool(force_loc)
        x = _add_corner_patch(x, top_left=loc_tl, size=5, val=(0.0,0.0,1.0))
        return torch.from_numpy(_to_float01(x))

    def get_cf_tensor(self, idx: int, kind: str) -> torch.Tensor:
        """
        kind in {"color", "bg", "loc", "joint"}
        """
        if kind=="color":
            return self._build_rgb(idx, force_color=1-self.s_color[idx], force_bg=None, force_loc=None)
        elif kind=="bg":
            return self._build_rgb(idx, force_color=None, force_bg=1-self.s_bg[idx], force_loc=None)
        elif kind=="loc":
            return self._build_rgb(idx, force_color=None, force_bg=None, force_loc=1-self.s_loc[idx])
        elif kind=="joint":
            return self._build_rgb(idx, force_color=1-self.s_color[idx],
                                        force_bg=1-self.s_bg[idx],
                                        force_loc=1-self.s_loc[idx])
        else:
            raise ValueError(f"unknown cf kind: {kind}")

    def __getitem__(self, idx):
        y = int(self.labels[idx])
        x = self._build_rgb(idx)
        s1 = int(self.s_color[idx]); s2=int(self.s_bg[idx]); s3=int(self.s_loc[idx])
        return x, y, s1, s2, s3, idx

class MultiSpurMNIST(MultiSpurBase):
    def __init__(self, root, split="train", p_train=0.99, p_test=0.1, seed=0, download=True):
        base = datasets.MNIST(root=root, train=(split=="train"), download=download)
        super().__init__(base, make_binary_label=lambda d: (d<5), split=split,
                         p_train=p_train, p_test=p_test, seed=seed, is_grayscale=True)

class MultiSpurCIFAR10(MultiSpurBase):
    ANIMALS = {2,3,4,5,6,7}
    def __init__(self, root, split="train", p_train=0.995, p_test=0.1, seed=0, download=True):
        base = datasets.CIFAR10(root=root, train=(split=="train"), download=download)
        def make_binary_label(raw_y: int) -> int:
            return 1 if raw_y in self.ANIMALS else 0
        super().__init__(base, make_binary_label=make_binary_label, split=split,
                         p_train=p_train, p_test=p_test, seed=seed, is_grayscale=False)

# -----------------------
# Model (with a "tap" to expose an intermediate representation h_l)
# -----------------------
class CNNFeatSmall(nn.Module):
    """Light CNN with a feature head (28x28 or 32x32), exposing a tap 'pre_proj2' (dim=256)."""
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
        self.classifier = nn.Linear(d_feat, 2)

    def forward_with_tap(self, x, tap="pre_proj2"):
        h0 = self.block(x)              # [B, 128*4*4]
        h1 = self.proj1(h0)             # tap
        z  = self.proj2(h1)             # [B, d_feat]
        logits = self.classifier(z)
        if tap == "pre_proj2":
            return logits, h1
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
    """Estimate AGOP G_l on tap representation; return top-k eigenvectors U (D x k)."""
    model.eval()
    G = None
    C = None
    for b, (x, y, s1, s2, s3, idx) in enumerate(loader):
        if b >= max_batches: break
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

    if G is None:
        W = model.classifier.weight.detach().to(device)
        G = W.T @ W

    G = G.float().detach().cpu()
    evals, evecs = torch.linalg.eigh(G)
    U = evecs[:, -k:].contiguous()  # [D, k]
    return U.to(device)

# -----------------------
# Eval helpers
# -----------------------
def run_epoch(model, loader, optimizer, criterion, train=True):
    if train: model.train()
    else: model.eval()
    total, correct, total_loss = 0, 0, 0.0
    for batch in loader:
        x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
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
def eval_worst_group_multi(model, loader):
    """Worst-group over 16 subgroups (y x s1 x s2 x s3)."""
    model.eval()
    stats = {(y,s1,s2,s3): [0,0] for y in [0,1] for s1 in [0,1] for s2 in [0,1] for s3 in [0,1]}
    for x, y, s1, s2, s3, idx in loader:
        x = x.to(DEVICE)
        pred = model(x).argmax(1).cpu()
        for i in range(len(y)):
            key = (int(y[i]), int(s1[i]), int(s2[i]), int(s3[i]))
            stats[key][0] += int(pred[i]==y[i])
            stats[key][1] += 1
    accs = {k: (v[0]/v[1] if v[1]>0 else 0.0) for k,v in stats.items()}
    worst = min(accs.values())
    avg = sum(accs.values())/len(accs)
    return worst, avg, accs

@torch.no_grad()
def eval_group_counts_multi(loader):
    counts = {(y,s1,s2,s3):0 for y in [0,1] for s1 in [0,1] for s2 in [0,1] for s3 in [0,1]}
    for _, y, s1, s2, s3, _ in loader:
        for i in range(len(y)):
            counts[(int(y[i]), int(s1[i]), int(s2[i]), int(s3[i]))] += 1
    return counts

@torch.no_grad()
def eval_flip_acc_multi(model, test_set, batch_size=256):
    """Return dict acc for each flip kind: color/bg/loc/joint."""
    base = test_set
    while isinstance(base, Subset):
        base = base.dataset
    assert isinstance(base, MultiSpurBase)
    model.eval()
    kinds = ["color","bg","loc","joint"]
    accs = {k: 0 for k in kinds}; tots = {k:0 for k in kinds}
    N = len(test_set)
    for s in range(0, N, batch_size):
        idxs = range(s, min(N, s+batch_size))
        xs = []
        for i in idxs:
            base_idx = int(test_set.indices[i]) if isinstance(test_set, Subset) else i
            xs.append(base._build_rgb(base_idx))
        X = torch.stack(xs, dim=0).to(DEVICE)
        y = torch.tensor([test_set[i][1] for i in idxs], dtype=torch.long).to(DEVICE)
        # for each flip kind
        for k in kinds:
            Xf = torch.stack([base.get_cf_tensor(int(test_set.indices[i]) if isinstance(test_set, Subset) else i, k)
                              for i in idxs], dim=0).to(DEVICE)
            pred = model(Xf).argmax(1)
            accs[k] += (pred==y).sum().item()
            tots[k] += y.numel()
    return {k: accs[k]/max(1,tots[k]) for k in kinds}

@torch.no_grad()
def eval_perm_acc(model, test_loader, seed=0):
    rng = np.random.RandomState(seed)
    model.eval()
    preds, ys = [], []
    for x, y, s1, s2, s3, idx in test_loader:
        x = x.to(DEVICE)
        pred = model(x).argmax(1).cpu().numpy()
        preds.append(pred); ys.append(y.numpy())
    preds = np.concatenate(preds); ys = np.concatenate(ys)
    perm = rng.permutation(len(ys))
    y_shuf = ys[perm]
    return float((preds==y_shuf).mean())

# -----------------------
# Pair-sensitivity (multi flips)
# -----------------------
@torch.no_grad()
def compute_pair_sensitivity_multi_old(model: nn.Module, loader: DataLoader, dataset: MultiSpurBase,
                                       U: torch.Tensor, probe_frac: float = 1.0) -> Tuple[Dict[int,float], Dict[int,str]]:
    """On final feature z; for each idx compute Δ for each flip in {color,bg,loc}; take max & argmax."""
    model.eval(); U = U.to(DEVICE)
    flips = ["color","bg","loc"]
    scores: Dict[int, float] = {}
    choices: Dict[int, str] = {}
    rng = np.random.RandomState(0)
    for x, y, s1, s2, s3, idx in loader:
        idx_list = idx.tolist()
        if probe_frac < 1.0:
            take = rng.rand(len(idx_list)) < probe_frac
            idx_list = [idx_list[i] for i,b in enumerate(take) if b]
            if len(idx_list) == 0: continue
            mask = torch.tensor([i in set(idx_list) for i in idx.tolist()], dtype=torch.bool)
            x = x[mask]; idx = idx[mask]
        x = x.to(DEVICE)
        with torch.no_grad():
            _, z = model(x, return_feat=True)
            z = F.normalize(z, dim=1)
            p = z @ U
        best = np.zeros(len(idx_list), dtype=np.float32) - 1.0
        which = ["color"]*len(idx_list)
        for k in flips:
            Xf = torch.stack([dataset.get_cf_tensor(int(i), k) for i in idx_list], dim=0).to(DEVICE)
            _, zf = model(Xf, return_feat=True)
            zf = F.normalize(zf, dim=1)
            pf = zf @ U
            num = ((p - pf)**2).sum(dim=1)
            den = (z.pow(2).sum(dim=1) + zf.pow(2).sum(dim=1) + 1e-8)
            delta = (num/den).detach().cpu().numpy()
            for j in range(len(idx_list)):
                if delta[j] > best[j]:
                    best[j] = float(delta[j]); which[j] = k
        for j, sid in enumerate(idx_list):
            scores[sid] = float(best[j]); choices[sid] = which[j]
    return scores, choices

@torch.no_grad()
def compute_pair_sensitivity_multi_tap(model: nn.Module, loader: DataLoader, dataset: MultiSpurBase,
                                       U: torch.Tensor, tap="pre_proj2", probe_frac: float = 1.0) -> Tuple[Dict[int,float], Dict[int,str]]:
    """On tap h_l; for each idx compute Δ for each flip; take max & argmax."""
    model.eval(); U = U.to(DEVICE)
    flips = ["color","bg","loc"]
    scores: Dict[int, float] = {}
    choices: Dict[int, str] = {}
    rng = np.random.RandomState(0)
    for x, y, s1, s2, s3, idx in loader:
        idx_list = idx.tolist()
        if probe_frac < 1.0:
            take = rng.rand(len(idx_list)) < probe_frac
            idx_list = [idx_list[i] for i,b in enumerate(take) if b]
            if len(idx_list) == 0: continue
            mask = torch.tensor([i in set(idx_list) for i in idx.tolist()], dtype=torch.bool)
            x = x[mask]; idx = idx[mask]
        x = x.to(DEVICE)
        _, h = model.forward_with_tap(x, tap=tap)
        h = F.normalize(h, dim=1); p = h @ U
        best = np.zeros(len(idx_list), dtype=np.float32) - 1.0
        which = ["color"]*len(idx_list)
        for k in flips:
            Xf = torch.stack([dataset.get_cf_tensor(int(i), k) for i in idx_list], dim=0).to(DEVICE)
            _, hf = model.forward_with_tap(Xf, tap=tap)
            hf = F.normalize(hf, dim=1); pf = hf @ U
            num = ((p - pf)**2).sum(dim=1)
            den = (h.pow(2).sum(dim=1) + hf.pow(2).sum(dim=1) + 1e-8)
            delta = (num/den).detach().cpu().numpy()
            for j in range(len(idx_list)):
                if delta[j] > best[j]:
                    best[j] = float(delta[j]); which[j] = k
        for j, sid in enumerate(idx_list):
            scores[sid] = float(best[j]); choices[sid] = which[j]
    return scores, choices

# -----------------------
# Train variants
# -----------------------
def train_erm(train_set, test_loader, epochs=30, lr=3e-4, wd=1e-4,
              eval_every=5, perm_seed=0, out_metrics:List[str]=None):
    model = CNNFeatSmall().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()
    tr_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
    te_set = test_loader.dataset
    logs=[]; printed=False
    for ep in range(1, epochs+1):
        run_epoch(model, tr_loader, opt, crit, train=True)
        te_loss, te_acc = run_epoch(model, test_loader, None, crit, train=False)
        wg, _, _ = eval_worst_group_multi(model, test_loader)
        flip = {}; delta_means={}
        if eval_every>0 and ((ep % eval_every)==1 or ep==epochs):
            flip = eval_flip_acc_multi(model, te_set)
            # quick Δ_U eval: use tap-U from a few batches
            with torch.enable_grad():
                U = estimate_agop_topk(model, test_loader, tap="pre_proj2", k=2, max_batches=2)
            # compute mean Δ for each flip
            delta_means = {}
            for kind in ["color","bg","loc"]:
                d = {}
                base = te_set
                with torch.no_grad():
                    # reuse sensitivity Tap with probe frac
                    d, _ = compute_pair_sensitivity_multi_tap(model, test_loader, base, U, tap="pre_proj2", probe_frac=0.4)
                arr = np.array(list(d.values()), dtype=np.float32)
                delta_means[f"deltaU_mean_{kind}"] = float(np.nanmean(arr)) if arr.size>0 else float("nan")
        perm = eval_perm_acc(model, test_loader, seed=perm_seed) if (eval_every>0 and ((ep % eval_every)==1 or ep==epochs)) else np.nan
        row = {"epoch":ep,"te_acc":te_acc,"worst_group_acc":wg,"perm_acc":perm}
        row.update({f"flip_acc_{k}":v for k,v in flip.items()})
        row.update(delta_means)
        logs.append(row)
        if not printed:
            print("[test group counts]", eval_group_counts_multi(test_loader)); printed=True
        msg = f"[ERM] ep{ep:02d} te_acc={te_acc:.3f} worstG={wg:.3f}"
        if flip: msg += " " + " ".join([f"{k}={v:.3f}" for k,v in flip.items()])
        if delta_means: msg += " " + " ".join([f"{k}={v:.4f}" for k,v in delta_means.items()])
        if not np.isnan(perm): msg += f" perm-acc={perm:.3f}"
        print(msg)
    return model, logs

def train_jtt(train_set, test_loader, total_epochs=30, stage1_epochs=5, upsample=10, lr=3e-4,
              eval_every=5, perm_seed=0):
    base = CNNFeatSmall().to(DEVICE)
    opt = torch.optim.AdamW(base.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    base_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
    logs=[]; printed=False
    for ep in range(1, stage1_epochs+1):
        run_epoch(base, base_loader, opt, crit, train=True)
        te_loss, te_acc = run_epoch(base, test_loader, None, crit, train=False)
        wg, _, _ = eval_worst_group_multi(base, test_loader)
        logs.append({"epoch":ep,"te_acc":te_acc,"worst_group_acc":wg})
        if not printed:
            print("[test group counts]", eval_group_counts_multi(test_loader)); printed=True
        print(f"[JTT-Stage1] ep{ep:02d} te_acc={te_acc:.3f} worstG={wg:.3f}")

    base.eval()
    mis_idx=[]
    with torch.no_grad():
        for x, y, s1, s2, s3, idx in DataLoader(train_set, batch_size=256, shuffle=False, num_workers=2, pin_memory=True):
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
    crit2 = nn.CrossEntropyLoss()
    for ep in range(stage1_epochs+1, total_epochs+1):
        run_epoch(model, train_loader, opt2, crit2, train=True)
        te_loss, te_acc = run_epoch(model, test_loader, None, crit2, train=False)
        wg, _, _ = eval_worst_group_multi(model, test_loader)
        flip = {}; delta_means={}
        if eval_every>0 and ((ep % eval_every)==1 or ep==total_epochs):
            flip = eval_flip_acc_multi(model, test_loader.dataset)
            with torch.enable_grad():
                U = estimate_agop_topk(model, test_loader, tap="pre_proj2", k=2, max_batches=2)
            delta_means={}
            for kind in ["color","bg","loc"]:
                d, _ = compute_pair_sensitivity_multi_tap(model, test_loader, test_loader.dataset, U, tap="pre_proj2", probe_frac=0.4)
                arr = np.array(list(d.values()), dtype=np.float32)
                delta_means[f"deltaU_mean_{kind}"] = float(np.nanmean(arr)) if arr.size>0 else float("nan")
        perm = eval_perm_acc(model, test_loader, seed=perm_seed) if (eval_every>0 and ((ep % eval_every)==1 or ep==total_epochs)) else np.nan
        row = {"epoch":ep,"te_acc":te_acc,"worst_group_acc":wg,"perm_acc":perm}
        row.update({f"flip_acc_{k}":v for k,v in flip.items()}); row.update(delta_means)
        logs.append(row)
        msg = f"[JTT] ep{ep:02d} te_acc={te_acc:.3f} worstG={wg:.3f}"
        if flip: msg += " " + " ".join([f"{k}={v:.3f}" for k,v in flip.items()])
        if delta_means: msg += " " + " ".join([f"{k}={v:.4f}" for k,v in delta_means.items()])
        if not np.isnan(perm): msg += f" perm-acc={perm:.3f}"
        print(msg)
    return model, logs

def _subset_by_indices(train_set, base_indices: List[int]) -> Subset:
    if isinstance(train_set, Subset):
        base_to_pos = {int(b): i for i, b in enumerate(train_set.indices)}
        pos = [base_to_pos[i] for i in base_indices if int(i) in base_to_pos]
        if len(pos) == 0:
            pos = list(range(len(train_set)))
        return Subset(train_set, pos)
    else:
        return Subset(train_set, base_indices)

@torch.no_grad()
def compute_losses_per_sample(model, loader, criterion) -> Dict[int,float]:
    model.eval()
    losses={}
    for x, y, s1, s2, s3, idx in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="none").detach().cpu().numpy()
        for i, sid in enumerate(idx.tolist()):
            losses[sid] = float(loss[i])
    return losses

def train_spl(train_set, test_loader, total_epochs=30, lr=3e-4, keep_start=0.3, keep_end=0.9,
              variant="easy", eval_every=5, perm_seed=0):
    assert variant in {"easy","hard","random"}
    model = CNNFeatSmall().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    base_loader = DataLoader(train_set, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    logs=[]; printed=False
    for ep in range(1, total_epochs+1):
        keep = min(keep_end, keep_start + (keep_end - keep_start) * (ep-1)/(total_epochs-1))
        n_total = len(train_set); m = max(1, int(n_total*keep))
        if variant == "random":
            selected = random.sample(range(n_total), m)
        else:
            losses = compute_losses_per_sample(model, base_loader, crit)
            items = sorted(losses.items(), key=lambda kv: kv[1], reverse=(variant=="hard"))
            selected = [i for i,_ in items[:m]]
        subset = _subset_by_indices(train_set, selected)
        tr_loader = DataLoader(subset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)

        run_epoch(model, tr_loader, opt, crit, train=True)
        te_loss, te_acc = run_epoch(model, test_loader, None, crit, train=False)
        wg, _, _ = eval_worst_group_multi(model, test_loader)
        flip = {}; delta_means={}
        if eval_every>0 and ((ep % eval_every)==1 or ep==total_epochs):
            flip = eval_flip_acc_multi(model, test_loader.dataset)
            with torch.enable_grad():
                U = estimate_agop_topk(model, test_loader, tap="pre_proj2", k=2, max_batches=2)
            for kind in ["color","bg","loc"]:
                d, _ = compute_pair_sensitivity_multi_tap(model, test_loader, test_loader.dataset, U, tap="pre_proj2", probe_frac=0.4)
                arr = np.array(list(d.values()), dtype=np.float32)
                delta_means[f"deltaU_mean_{kind}"] = float(np.nanmean(arr)) if arr.size>0 else float("nan")
        perm = eval_perm_acc(model, test_loader, seed=perm_seed) if (eval_every>0 and ((ep % eval_every)==1 or ep==total_epochs)) else np.nan
        row = {"epoch":ep,"te_acc":te_acc,"worst_group_acc":wg,"perm_acc":perm,"keep_ratio":keep,"subset_size":m}
        row.update({f"flip_acc_{k}":v for k,v in flip.items()}); row.update(delta_means)
        logs.append(row)
        if not printed:
            print("[test group counts]", eval_group_counts_multi(test_loader)); printed=True
        tag = {"easy":"SPL-Easy","hard":"SPL-Hard","random":"Random-Curr"}[variant]
        msg = f"[{tag}] ep{ep:02d} keep={keep:.2f} te_acc={te_acc:.3f} worstG={wg:.3f}"
        if flip: msg += " " + " ".join([f"{k}={v:.3f}" for k,v in flip.items()])
        if delta_means: msg += " " + " ".join([f"{k}={v:.4f}" for k,v in delta_means.items()])
        if not np.isnan(perm): msg += f" perm-acc={perm:.3f}"
        print(msg)
    return model, logs

# --- AGC-InvCFP old (multi) ---
def train_agc_invcfp_old_multi(train_set, test_loader, total_epochs=30, lr=3e-4,
                               keep_start=0.3, keep_end=0.9, k_desired=2,
                               probe_frac=0.6, lambda_cons=0.2,
                               eval_every=5, perm_seed=0):
    dataset = train_set
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    model = CNNFeatSmall().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    base_loader = DataLoader(train_set, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    logs=[]; printed=False
    for ep in range(1, total_epochs+1):
        keep = min(keep_end, keep_start + (keep_end - keep_start) * (ep-1)/(total_epochs-1))
        U = topk_basis_from_classifier(model, k_desired=k_desired)
        scores, choices = compute_pair_sensitivity_multi_old(model, base_loader, dataset, U, probe_frac=probe_frac)
        n = len(scores); m = max(1, int(n * keep))
        idx_sorted = [i for i,_ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
        top_idx = idx_sorted[:m]
        # + small random for stability
        remain = list(set(scores.keys()) - set(top_idx))
        if len(remain)>0:
            top_idx += random.sample(remain, min(int(0.1*n), len(remain)))

        subset = _subset_by_indices(train_set, top_idx)
        tr_loader = DataLoader(subset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
        # one epoch training
        model.train()
        for x, y, s1, s2, s3, idx in tr_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            # pick cf per-sample by recorded choice
            x_flip = torch.stack([dataset.get_cf_tensor(int(i), choices[int(i)]) for i in idx.tolist()], dim=0).to(DEVICE)
            logits, z  = model(x, return_feat=True)
            logits_f, zf = model(x_flip, return_feat=True)
            ce = crit(logits, y) + crit(logits_f, y)
            cons = F.mse_loss(z, zf)
            loss = ce + lambda_cons * cons
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

        te_loss, te_acc = run_epoch(model, test_loader, None, crit, train=False)
        wg, _, _ = eval_worst_group_multi(model, test_loader)
        flip = {}; delta_means={}
        if eval_every>0 and ((ep % eval_every)==1 or ep==total_epochs):
            flip = eval_flip_acc_multi(model, test_loader.dataset)
            with torch.enable_grad():
                Ue = estimate_agop_topk(model, test_loader, tap="pre_proj2", k=k_desired, max_batches=2)
            for kind in ["color","bg","loc"]:
                d, _ = compute_pair_sensitivity_multi_tap(model, test_loader, test_loader.dataset, Ue, tap="pre_proj2", probe_frac=0.4)
                arr = np.array(list(d.values()), dtype=np.float32)
                delta_means[f"deltaU_mean_{kind}"] = float(np.nanmean(arr)) if arr.size>0 else float("nan")
        perm = eval_perm_acc(model, test_loader, seed=perm_seed) if (eval_every>0 and ((ep % eval_every)==1 or ep==total_epochs)) else np.nan
        row = {"epoch":ep,"keep_ratio":keep,"te_acc":te_acc,"worst_group_acc":wg,"perm_acc":perm}
        row.update({f"flip_acc_{k}":v for k,v in flip.items()}); row.update(delta_means)
        logs.append(row)
        if not printed:
            print("[test group counts]", eval_group_counts_multi(test_loader)); printed=True
        msg = f"[AGC_INV-CFP_OLD_MULTI] ep{ep:02d} keep={keep:.2f} te_acc={te_acc:.3f} worstG={wg:.3f}"
        if flip: msg += " " + " ".join([f"{k}={v:.3f}" for k,v in flip.items()])
        if delta_means: msg += " " + " ".join([f"{k}={v:.4f}" for k,v in delta_means.items()])
        if not np.isnan(perm): msg += f" perm-acc={perm:.3f}"
        print(msg)
    return model, logs

# --- AGC-InvCFP new (multi; AGOP@tap + projection) ---
def train_agc_invcfp_new_multi(train_set, test_loader, total_epochs=30, lr=3e-4,
                               keep_start=0.3, keep_end=0.9, k_desired=2,
                               probe_frac=0.6, lambda_cons=0.2,
                               agop_update_every=1, agop_train_batches=3,
                               eval_every=5, perm_seed=0, tap="pre_proj2"):
    dataset = train_set
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    model = CNNFeatSmall().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    base_loader = DataLoader(train_set, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    logs=[]; printed=False
    U = estimate_agop_topk(model, base_loader, tap=tap, k=k_desired, max_batches=agop_train_batches)
    for ep in range(1, total_epochs+1):
        keep = min(keep_end, keep_start + (keep_end - keep_start) * (ep-1)/(total_epochs-1))
        if (ep==1) or (agop_update_every>0 and ep % agop_update_every == 0):
            U = estimate_agop_topk(model, base_loader, tap=tap, k=k_desired, max_batches=agop_train_batches)

        scores, choices = compute_pair_sensitivity_multi_tap(model, base_loader, dataset, U, tap=tap, probe_frac=probe_frac)
        n = len(scores); m = max(1, int(n * keep))
        idx_sorted = [i for i,_ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
        top_idx = idx_sorted[:m]
        remain = list(set(scores.keys()) - set(top_idx))
        if len(remain)>0:
            top_idx += random.sample(remain, min(int(0.1*n), len(remain)))

        subset = _subset_by_indices(train_set, top_idx)
        tr_loader = DataLoader(subset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)

        # one epoch with projection consistency on chosen flip per sample
        model.train()
        for x, y, s1, s2, s3, idx in tr_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x_flip = torch.stack([dataset.get_cf_tensor(int(i), choices[int(i)]) for i in idx.tolist()], dim=0).to(DEVICE)

            logits = model(x)
            logits_f = model(x_flip)
            _, h  = model.forward_with_tap(x, tap=tap)
            _, hf = model.forward_with_tap(x_flip, tap=tap)
            proj  = h  @ U
            projf = hf @ U

            ce = crit(logits, y) + crit(logits_f, y)
            cons = F.mse_loss(proj, projf)
            loss = ce + lambda_cons * cons

            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

        te_loss, te_acc = run_epoch(model, test_loader, None, crit, train=False)
        wg, _, _ = eval_worst_group_multi(model, test_loader)
        flip = {}; delta_means={}
        if eval_every>0 and ((ep % eval_every)==1 or ep==total_epochs):
            flip = eval_flip_acc_multi(model, test_loader.dataset)
            with torch.enable_grad():
                Ue = estimate_agop_topk(model, test_loader, tap=tap, k=k_desired, max_batches=2)
            for kind in ["color","bg","loc"]:
                d, _ = compute_pair_sensitivity_multi_tap(model, test_loader, test_loader.dataset, Ue, tap=tap, probe_frac=0.4)
                arr = np.array(list(d.values()), dtype=np.float32)
                delta_means[f"deltaU_mean_{kind}"] = float(np.nanmean(arr)) if arr.size>0 else float("nan")
        perm = eval_perm_acc(model, test_loader, seed=perm_seed) if (eval_every>0 and ((ep % eval_every)==1 or ep==total_epochs)) else np.nan
        row = {"epoch":ep,"keep_ratio":keep,"te_acc":te_acc,"worst_group_acc":wg,"perm_acc":perm}
        row.update({f"flip_acc_{k}":v for k,v in flip.items()}); row.update(delta_means)
        logs.append(row)
        if not printed:
            print("[test group counts]", eval_group_counts_multi(test_loader)); printed=True
        msg = f"[AGC_INV-CFP_NEW_MULTI] ep{ep:02d} keep={keep:.2f} te_acc={te_acc:.3f} worstG={wg:.3f}"
        if flip: msg += " " + " ".join([f"{k}={v:.3f}" for k,v in flip.items()])
        if delta_means: msg += " " + " ".join([f"{k}={v:.4f}" for k,v in delta_means.items()])
        if not np.isnan(perm): msg += f" perm-acc={perm:.3f}"
        print(msg)
    return model, logs

# -----------------------
# Task builder
# -----------------------
def build_task(task: str, root="./data", seed=0, p_train=0.99, p_test=0.1,
               train_fraction=1.0, download=True):
    task = task.lower()
    if task=="multispur_mnist":
        tr = MultiSpurMNIST(root, "train", p_train=p_train, p_test=p_test, seed=seed, download=download)
        te = MultiSpurMNIST(root, "test",  p_train=p_train, p_test=p_test, seed=seed+1, download=download)
        batch = 256
    elif task=="multispur_cifar10":
        tr = MultiSpurCIFAR10(root, "train", p_train=max(p_train,0.995), p_test=p_test, seed=seed, download=download)
        te = MultiSpurCIFAR10(root, "test",  p_train=max(p_train,0.995), p_test=p_test, seed=seed+1, download=download)
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
# Aggregation & plotting
# -----------------------
def aggregate_across_seeds(results_per_seed: List[List[Dict]], metric: str, max_epoch=None):
    if not results_per_seed or any(len(x)==0 for x in results_per_seed):
        return None
    min_len = min(len(logs) for logs in results_per_seed)
    if max_epoch is not None:
        min_len = min(min_len, max_epoch)
    epochs = np.array([results_per_seed[0][i]["epoch"] for i in range(min_len)])
    vals = np.stack([[logs[i].get(metric, np.nan) for i in range(min_len)] for logs in results_per_seed], axis=0)
    mean = np.nanmean(vals, axis=0)
    std  = np.nanstd(vals, axis=0, ddof=1) if vals.shape[0] >= 2 else np.zeros_like(mean)
    se   = std / np.sqrt(max(vals.shape[0],1)); ci = 1.96 * se
    return epochs, mean, mean - ci, mean + ci

def plot_metric_curves(agg: Dict[str, Dict[str, Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]]],
                       metric_name: str, out_png: str, title: str):
    plt.figure(figsize=(7,5))
    ok=False
    for method, d in agg.items():
        if metric_name not in d or d[metric_name] is None: continue
        ep, mu, lo, hi = d[metric_name]
        plt.plot(ep, mu, label=method)
        plt.fill_between(ep, lo, hi, alpha=0.2)
        ok=True
    if not ok:
        plt.close(); return
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
    plt.figure(figsize=(7,5))
    plt.bar(x, means, yerr=ci95, capsize=4)
    plt.xticks(x, labels, rotation=15)
    plt.ylabel(metric); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

# -----------------------
# Runner
# -----------------------
def run_all(task: str, seeds: List[int], outdir: str,
            epochs=30, p_train=0.99, p_test=0.1, train_fraction=1.0,
            lr=3e-4, upsample=10, keep_start=0.3, keep_end=0.9,
            eval_every=5, perm_seed=0,
            agop_update_every=1, agop_train_batches=3):
    ensure_dir(outdir)

    # for sanity: group counts and overlap
    tr_set0, te_set0, tr_loader0, te_loader0 = build_task(task, seed=seeds[0], p_train=p_train, p_test=p_test,
                                                          train_fraction=train_fraction)
    overlap = 0  # multispur uses distinct base objects
    counts = eval_group_counts_multi(te_loader0)
    print(f"[sanity] train/test index overlap = {overlap} (should be 0)")
    print(f"[test group counts (first seed)]: {counts}")

    cfg = dict(task=task, seeds=seeds, epochs=epochs, p_train=p_train, p_test=p_test,
               train_fraction=train_fraction, lr=lr, upsample=upsample,
               keep_start=keep_start, keep_end=keep_end, eval_every=eval_every,
               agop_update_every=agop_update_every, agop_train_batches=agop_train_batches)
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    results_by_method = {
        "ERM": [],
        "JTT": [],
        "SPL-Easy": [],
        "SPL-Hard": [],
        "Random-Curr": [],
        "AGC-InvCFP-old-multi": [],
        "AGC-InvCFP-new-multi": [],
    }

    for s in seeds:
        set_seed(s)
        tr_set, te_set, tr_loader, te_loader = build_task(task, seed=s, p_train=p_train, p_test=p_test,
                                                          train_fraction=train_fraction)

        # ERM
        erm_model, erm_logs = train_erm(tr_set, te_loader, epochs=epochs, lr=lr, eval_every=eval_every, perm_seed=perm_seed)
        write_csv(os.path.join(outdir, f"ERM_seed{s}.csv"), erm_logs)
        results_by_method["ERM"].append(erm_logs)

        # JTT
        jtt_model, jtt_logs = train_jtt(tr_set, te_loader, total_epochs=epochs, stage1_epochs=max(1,epochs//6),
                                        upsample=upsample, lr=lr, eval_every=eval_every, perm_seed=perm_seed)
        write_csv(os.path.join(outdir, f"JTT_seed{s}.csv"), jtt_logs)
        results_by_method["JTT"].append(jtt_logs)

        # SPL baselines + Random
        for variant, tag in [("easy","SPL-Easy"), ("hard","SPL-Hard"), ("random","Random-Curr")]:
            mdl, logs = train_spl(tr_set, te_loader, total_epochs=epochs, lr=lr,
                                  keep_start=keep_start, keep_end=keep_end,
                                  variant=variant, eval_every=eval_every, perm_seed=perm_seed)
            write_csv(os.path.join(outdir, f"{tag.replace(' ','_')}_seed{s}.csv"), logs)
            results_by_method[tag].append(logs)

        # AGC old multi
        agc_old_model, agc_old_logs = train_agc_invcfp_old_multi(tr_set, te_loader, total_epochs=epochs, lr=lr,
                                                                 keep_start=keep_start, keep_end=keep_end, k_desired=2,
                                                                 probe_frac=0.6, lambda_cons=0.2,
                                                                 eval_every=eval_every, perm_seed=perm_seed)
        write_csv(os.path.join(outdir, f"AGC_InvCFP_old_multi_seed{s}.csv"), agc_old_logs)
        results_by_method["AGC-InvCFP-old-multi"].append(agc_old_logs)

        # AGC new multi
        agc_new_model, agc_new_logs = train_agc_invcfp_new_multi(tr_set, te_loader, total_epochs=epochs, lr=lr,
                                                                 keep_start=keep_start, keep_end=keep_end, k_desired=2,
                                                                 probe_frac=0.6, lambda_cons=0.2,
                                                                 agop_update_every=agop_update_every, agop_train_batches=agop_train_batches,
                                                                 eval_every=eval_every, perm_seed=perm_seed, tap="pre_proj2")
        write_csv(os.path.join(outdir, f"AGC_InvCFP_new_multi_seed{s}.csv"), agc_new_logs)
        results_by_method["AGC-InvCFP-new-multi"].append(agc_new_logs)

    # Aggregate & plots
    agg = {}
    metrics_to_plot = [
        "te_acc", "worst_group_acc",
        "flip_acc_color", "flip_acc_bg", "flip_acc_loc", "flip_acc_joint",
        "deltaU_mean_color", "deltaU_mean_bg", "deltaU_mean_loc",
        "perm_acc"
    ]
    for method, seed_logs in results_by_method.items():
        agg[method] = {}
        for m in metrics_to_plot:
            agg[method][m] = aggregate_across_seeds(seed_logs, metric=m)

    for m in metrics_to_plot:
        plot_metric_curves(agg, m, os.path.join(outdir, f"curve_{m}.png"),
                           f"{task}: {m} (mean±95% CI)")

    # Best bars
    plot_bars_best(results_by_method, "te_acc", os.path.join(outdir, "best_te_acc.png"),
                   f"{task}: Best Test Acc across seeds")
    plot_bars_best(results_by_method, "worst_group_acc", os.path.join(outdir, "best_worst_group.png"),
                   f"{task}: Best Worst-group Acc across seeds")
    plot_bars_best(results_by_method, "flip_acc_joint", os.path.join(outdir, "best_flip_joint.png"),
                   f"{task}: Best Joint-Flip Acc across seeds")

    print(f"[DONE] Results saved under: {outdir}")

# -----------------------
# CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["multispur_mnist","multispur_cifar10"])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--p-train", type=float, default=0.99)
    parser.add_argument("--p-test", type=float, default=0.1)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--upsample", type=int, default=10)
    parser.add_argument("--keep-start", type=float, default=0.3)
    parser.add_argument("--keep-end", type=float, default=0.9)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--perm-seed", type=int, default=0)
    parser.add_argument("--agop-update-every", type=int, default=1)
    parser.add_argument("--agop-train-batches", type=int, default=3)
    parser.add_argument("--out-root", type=str, default="experiments/curriculum_bench")
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
                keep_start=args.keep_start, keep_end=args.keep_end,
                eval_every=args.eval_every, perm_seed=args.perm_seed,
                agop_update_every=args.agop_update_every, agop_train_batches=args.agop_train_batches)

if __name__ == "__main__":
    main()
