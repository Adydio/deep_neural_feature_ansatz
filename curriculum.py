# -*- coding: utf-8 -*-
# AGOP-aware Curriculum on Colored MNIST
# Requires: torch, torchvision, numpy, tqdm
import math, random, os, copy, time
from dataclasses import dataclass
from typing import Tuple, List, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, Sampler
from torchvision import datasets, transforms
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

# ---------------------------
# 1) Colored MNIST dataset
# ---------------------------
class ColoredMNIST(Dataset):
    """
    Binary label: y = (digit < 5)
    Color c correlates with y in train (p_train), weakly in test (p_test).
    """
    def __init__(self, root, split="train", p_train=0.99, p_test=0.1, download=True):
        assert split in ["train","test"]
        self.split = split
        self.base = datasets.MNIST(root=root, train=(split=="train"),
                                   download=download, transform=None)
        self.p_corr = p_train if split=="train" else p_test

    def __len__(self): return len(self.base)

    def _colorize(self, img_np, c):
        # img_np: HxW uint8 [0..255], produce 3xHxW float [0..1]
        img = img_np.astype(np.float32)/255.0
        R = img if c==0 else np.zeros_like(img)
        G = img if c==1 else np.zeros_like(img)
        B = np.zeros_like(img)
        arr = np.stack([R,G,B], axis=0)  # 3xHxW
        return torch.from_numpy(arr)

    def __getitem__(self, idx):
        img, digit = self.base[idx]
        img = np.array(img, dtype=np.uint8)
        y = 1 if digit < 5 else 0  # binary label
        # assign color correlated with y with prob p_corr
        if np.random.rand() < self.p_corr:
            c = y
        else:
            c = 1 - y
        x = self._colorize(img, c)
        return x, y, c, idx  # return color for group metrics

# ---------------------------
# 2) Small CNN with feature head
# ---------------------------
class CNNFeat(nn.Module):
    def __init__(self, d_feat=256, num_classes=2):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4,4)),
            nn.Flatten(),
            nn.Linear(128*4*4, d_feat), nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(d_feat, num_classes)

    def forward(self, x, return_feat=False):
        z = self.feat(x)
        logits = self.classifier(z)
        if return_feat:
            return logits, z
        return logits

# ---------------------------
# 3) Utilities: AGOP subspace, difficulty, spectral entropy
# ---------------------------
@torch.no_grad()
def topk_basis_from_classifier(model: CNNFeat, k: int) -> torch.Tensor:
    W = model.classifier.weight.detach()  # [C, d]
    M = W.T @ W                           # [d, d]
    # eigh returns ascending order
    eigvals, eigvecs = torch.linalg.eigh(M)  # eigvecs: [d, d]
    U = eigvecs[:, -k:]                     # top-k
    return U  # [d, k]

@torch.no_grad()
def spectral_entropy(model: CNNFeat) -> float:
    W = model.classifier.weight.detach()
    M = W.T @ W
    evals = torch.linalg.eigvalsh(M).clamp(min=1e-12)
    p = (evals / evals.sum()).cpu().numpy()
    return float(-(p * np.log(p)).sum())

@torch.no_grad()
def compute_align_scores(model: CNNFeat, loader: DataLoader, U: torch.Tensor) -> Dict[int, float]:
    """
    Return s(i) for each index in dataset: 1 - ||U^T z||^2 / ||z||^2
    """
    model.eval()
    scores = {}
    U = U.to(DEVICE)
    for x, y, c, idx in loader:
        x = x.to(DEVICE)
        _, z = model(x, return_feat=True)
        z = F.normalize(z, dim=1)  # stabilize
        proj = (z @ U)             # [B, k]
        num = (proj ** 2).sum(dim=1)
        s = (1 - num).clamp(min=0.0, max=1.0).detach().cpu().numpy()
        for i, sid in enumerate(idx.tolist()):
            scores[sid] = float(s[i])
    return scores

def select_indices_by_curriculum(scores: Dict[int,float],
                                 labels: Dict[int,int],
                                 keep_ratio: float,
                                 variant: str = "agc_despur",
                                 min_random_frac: float = 0.1) -> List[int]:
    """
    variant in {"agc_easy", "agc_despur"}
    keep_ratio: fraction of data to keep this epoch
    """
    n = len(scores)
    m = max(1, int(n * keep_ratio))
    pairs = sorted(scores.items(), key=lambda kv: kv[1], reverse=(variant=="agc_despur"))
    selected = [i for i,_ in pairs[:m]]

    # mix-in random for stability
    remain = list(set(scores.keys()) - set(selected))
    rnd_k = int(n * min_random_frac)
    if rnd_k > 0:
        selected += random.sample(remain, min(rnd_k, len(remain)))
    random.shuffle(selected)

    # optional: class-balance (simple stratified cap)
    by_y = {0:[], 1:[]}
    for i in selected:
        by_y[labels[i]].append(i)
    half = len(selected)//2
    k0 = min(len(by_y[0]), half)
    k1 = min(len(by_y[1]), len(selected)-k0)
    balanced = by_y[0][:k0] + by_y[1][:k1]
    if len(balanced) >= int(0.6*len(selected)):
        selected = balanced
    return selected

# ---------------------------
# 4) Training loops (ERM, JTT, AGC)
# ---------------------------
def run_epoch(model, loader, optim, criterion, train=True):
    if train: model.train()
    else: model.eval()
    total, correct, total_loss = 0, 0, 0.0
    for x, y, c, idx in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
        pred = logits.argmax(1)
        total += y.size(0)
        correct += (pred==y).sum().item()
        total_loss += float(loss.item()) * y.size(0)
    return total_loss/total, correct/total

@torch.no_grad()
def eval_worst_group(model, loader):
    # groups: (y, color) in {0,1}x{0,1}
    model.eval()
    stats = {(y,c): [0,0] for y in [0,1] for c in [0,1]}  # [correct, total]
    for x, y, c, idx in loader:
        x = x.to(DEVICE)
        logits = model(x)
        pred = logits.argmax(1).cpu()
        for i in range(len(y)):
            key = (int(y[i]), int(c[i]))
            stats[key][0] += int(pred[i]==y[i])
            stats[key][1] += 1
    accs = {k: (v[0]/v[1] if v[1]>0 else 0.0) for k,v in stats.items()}
    worst = min(accs.values())
    avg = sum(accs.values())/len(accs)
    return worst, avg, accs

def train_erm(model, train_loader, test_loader, epochs=20, lr=1e-3, wd=1e-4):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()
    history=[]
    for ep in range(1, epochs+1):
        tr_loss, tr_acc = run_epoch(model, train_loader, opt, crit, train=True)
        te_loss, te_acc = run_epoch(model, test_loader, None, crit, train=False)
        wg, _, _ = eval_worst_group(model, test_loader)
        H = spectral_entropy(model)
        history.append((ep,tr_loss,tr_acc,te_loss,te_acc,wg,H))
        print(f"[ERM] ep{ep:02d} tr_acc={tr_acc:.3f} te_acc={te_acc:.3f} worstG={wg:.3f} H={H:.3f}")
    return model, history

def train_jtt(train_set, test_loader, epochs1=10, epochs2=20, upsample=10, lr=1e-3):
    # Stage 1: ERM short
    base = CNNFeat().to(DEVICE)
    opt = torch.optim.AdamW(base.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
    for ep in range(epochs1):
        run_epoch(base, loader, opt, crit, train=True)

    # find misclassified indices
    base.eval()
    mis_idx=[]
    with torch.no_grad():
        for x, y, c, idx in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = base(x).argmax(1)
            mask = (pred!=y).cpu().numpy()
            mis_idx += list(np.array(idx)[mask])
    # Stage 2: reweight those
    weights = np.ones(len(train_set))
    for i in mis_idx:
        weights[i] *= upsample

    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(train_set), replacement=True)
    train_loader = DataLoader(train_set, batch_size=256, sampler=sampler, num_workers=2, pin_memory=True)
    model = CNNFeat().to(DEVICE)
    opt2 = torch.optim.AdamW(model.parameters(), lr=lr)
    hist=[]
    for ep in range(1, epochs2+1):
        tr_loss, tr_acc = run_epoch(model, train_loader, opt2, crit, train=True)
        te_loss, te_acc = run_epoch(model, test_loader, None, crit, train=False)
        wg, _, _ = eval_worst_group(model, test_loader)
        H = spectral_entropy(model)
        hist.append((ep,tr_loss,tr_acc,te_loss,te_acc,wg,H))
        print(f"[JTT] ep{ep:02d} tr_acc={tr_acc:.3f} te_acc={te_acc:.3f} worstG={wg:.3f} H={H:.3f}")
    return model, hist

def train_agc(train_set, test_loader, variant="agc_despur",
              total_epochs=30, lr=1e-3, keep_start=0.2, keep_end=1.0, k_frac=0.25):
    """
    variant: "agc_easy" or "agc_despur"
    """
    model = CNNFeat().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    base_loader = DataLoader(train_set, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    # for labels dict (stratification)
    labels = {}
    for _, y, _, idx in base_loader:
        for i, sid in enumerate(idx.tolist()):
            labels[sid] = int(y[i])

    hist=[]
    for ep in range(1, total_epochs+1):
        # 1) compute AGOP subspace from classifier
        d = model.classifier.in_features
        k = max(1, int(k_frac * d))
        U = topk_basis_from_classifier(model, k)

        # 2) compute alignment scores on current features
        scores = compute_align_scores(model, base_loader, U)

        # 3) decide keep ratio (linear pacing; 可改为谱熵自适应)
        keep_ratio = keep_start + (keep_end - keep_start) * (ep-1)/(total_epochs-1)

        # 4) select indices
        selected_idx = select_indices_by_curriculum(scores, labels, keep_ratio, variant=variant)

        # 5) train one epoch on selected subset
        subset = Subset(train_set, selected_idx)
        train_loader = DataLoader(subset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
        tr_loss, tr_acc = run_epoch(model, train_loader, opt, crit, train=True)

        # eval
        te_loss, te_acc = run_epoch(model, test_loader, None, crit, train=False)
        wg, _, _ = eval_worst_group(model, test_loader)
        H = spectral_entropy(model)
        hist.append((ep, keep_ratio, tr_loss, tr_acc, te_loss, te_acc, wg, H))
        print(f"[{variant.upper()}] ep{ep:02d} keep={keep_ratio:.2f} te_acc={te_acc:.3f} worstG={wg:.3f} H={H:.3f}")

    return model, hist

# ---------------------------
# 5) Main: build data & run
# ---------------------------
def build_loaders(root="./data", batch=256):
    tr = ColoredMNIST(root, split="train", p_train=0.99, p_test=0.1, download=True)
    te = ColoredMNIST(root, split="test",  p_train=0.99, p_test=0.1, download=True)
    train_loader = DataLoader(tr, batch_size=batch, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(te, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)
    return tr, te, train_loader, test_loader

if __name__ == "__main__":
    tr_set, te_set, tr_loader, te_loader = build_loaders()

    # 1) ERM
    erm_model, erm_hist = train_erm(CNNFeat(), tr_loader, te_loader, epochs=20, lr=3e-4)

    # 2) JTT
    jtt_model, jtt_hist = train_jtt(tr_set, te_loader, epochs1=5, epochs2=15, upsample=10, lr=3e-4)

    # 3) AGC-Despur (ours)
    agc_model, agc_hist = train_agc(tr_set, te_loader, variant="agc_despur",
                                    total_epochs=25, lr=3e-4, keep_start=0.3, keep_end=1.0, k_frac=0.25)

    # 4) (Optional) AGC-Easy
    agc_easy_model, agc_easy_hist = train_agc(tr_set, te_loader, variant="agc_easy",
                                              total_epochs=25, lr=3e-4, keep_start=0.3, keep_end=1.0, k_frac=0.25)
