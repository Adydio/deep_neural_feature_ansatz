#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NanoGPT-style Tiny Shakespeare experiments:
AGOP Spectral Curriculum (AGOP-SC) vs baselines (random, loss-curriculum, self-paced, anti-curriculum).

- Saves logs, figures, tables into /experiments/nanogpt_curriculum/
- Minimal GPT implementation (no external nanogpt dependency).
- Candidate-pool selection to realize curriculum sampling per step.

Usage examples:
  python run_nanogpt_curriculum.py --method agop_sc --steps 4000
  python run_nanogpt_curriculum.py --method all --steps 4000

If /experiments/ is not writable on your system, the script will fallback to ./experiments/nanogpt_curriculum/.
"""

import os
import math
import time
import json
import argparse
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List

# ----------------------------- utils: I/O paths -----------------------------

def get_exp_dir():
    preferred = "/experiments/nanogpt_curriculum/"
    fallback = "./experiments/nanogpt_curriculum/"
    try:
        os.makedirs(preferred, exist_ok=True)
        # test write
        testfile = os.path.join(preferred, ".write_test")
        with open(testfile, "w") as f:
            f.write("ok")
        os.remove(testfile)
        return preferred
    except Exception:
        os.makedirs(fallback, exist_ok=True)
        return fallback

EXP_DIR = get_exp_dir()
LOG_DIR = os.path.join(EXP_DIR, "logs")
FIG_DIR = os.path.join(EXP_DIR, "figs")
CKPT_DIR = os.path.join(EXP_DIR, "ckpts")
TAB_DIR = os.path.join(EXP_DIR, "tables")
for d in [LOG_DIR, FIG_DIR, CKPT_DIR, TAB_DIR]:
    os.makedirs(d, exist_ok=True)

# ----------------------------- data: Tiny Shakespeare -----------------------------

TS_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def maybe_download_tiny_shakespeare(data_dir: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, "input.txt")
    if os.path.exists(data_path):
        return data_path
    # try to download
    try:
        import urllib.request
        print("Downloading Tiny Shakespeare...")
        urllib.request.urlretrieve(TS_URL, data_path)
        print(f"Saved to {data_path}")
    except Exception as e:
        print("WARNING: download failed, creating a tiny synthetic fallback dataset.")
        # minimal fallback to allow the script to run
        sample = ("To be, or not to be, that is the question:\n"
                  "Whether 'tis nobler in the mind to suffer\n"
                  "The slings and arrows of outrageous fortune,\n") * 512
        with open(data_path, "w", encoding="utf-8") as f:
            f.write(sample)
    return data_path

def build_char_dataset(path: str, split_ratio: float = 0.9):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(list(set(text)))
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for ch,i in stoi.items()}
    data = np.array([stoi[c] for c in text], dtype=np.int64)

    n = int(len(data) * split_ratio)
    train_data = data[:n]
    val_data = data[n:]
    vocab_size = len(chars)
    return train_data, val_data, vocab_size, stoi, itos

def get_batch(data: np.ndarray, block_size: int, batch_size: int, device: str):
    # random contiguous blocks
    ix = np.random.randint(0, len(data) - block_size - 1, size=(batch_size,))
    x = np.stack([data[i:i+block_size] for i in ix])
    y = np.stack([data[i+1:i+block_size+1] for i in ix])
    x = torch.from_numpy(x).long().to(device)
    y = torch.from_numpy(y).long().to(device)
    return x, y

# ----------------------------- model: a tiny GPT -----------------------------

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # causal mask
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1,config.block_size,config.block_size))

    def forward(self, x):
        B,T,C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # (B, nh, T, hs)
        q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)

        att = (q @ k.transpose(-2,-1)) * (1.0/ math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.resid_drop(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.dropout(self.c_proj(x))
        return x

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)  # (b,t,n_embd)
        pos_emb = self.transformer.wpe(pos)[None, :, :]  # (1,t,n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

# ----------------------------- AGOP-SC selector -----------------------------

def _orthonormalize(U: torch.Tensor) -> torch.Tensor:
    Q, _ = torch.linalg.qr(U, mode='reduced')
    return Q

class AGOPSpectralSelector:
    def __init__(self, vocab_size:int, m:int=64, k:int=8, beta:float=0.98,
                 c_target:float=0.35, alpha:float=20.0, device='cuda'):
        self.V = vocab_size
        self.m = m
        self.k = k
        self.beta = beta
        self.c_target = c_target
        self.alpha = alpha
        self.device = device

        self.R = torch.randn(self.V, self.m, device=device) / math.sqrt(self.m)
        self.S = torch.zeros(self.m, self.m, device=device)
        self.U = torch.eye(self.m, self.k, device=device)
        self.eigs = torch.ones(self.k, device=device)

    @torch.no_grad()
    def _update_spectrum(self, phi_batch: torch.Tensor):
        C = (phi_batch.T @ phi_batch) / max(1, phi_batch.size(0))
        self.S = self.beta * self.S + (1 - self.beta) * C
        evals, evecs = torch.linalg.eigh(self.S)
        topk = torch.argsort(evals, descending=True)[:self.k]
        self.eigs = evals[topk]
        self.U = _orthonormalize(evecs[:, topk])

    @torch.no_grad()
    def project_logits_grad(self, logits: torch.Tensor, targets: torch.Tensor, topk: Optional[int]=None):
        Bc, T, V = logits.shape
        assert V == self.V
        # For tiny Shakespeare, V is small, use full softmax
        p = F.softmax(logits, dim=-1)  # [B_c, T, V]
        pR = torch.matmul(p, self.R)   # [B_c, T, m]
        Rt = self.R[targets]           # [B_c, T, m]
        phi = (pR - Rt).sum(dim=1)     # [B_c, m]
        return phi

    @torch.no_grad()
    def score(self, phi: torch.Tensor, loss_per_seq: Optional[torch.Tensor]=None, gamma: float=0.0):
        proj = phi @ self.U
        num = (proj ** 2).sum(dim=1)
        den = (phi ** 2).sum(dim=1) + 1e-12
        a = (num / den).clamp(0, 1)
        n = 1.0 - a

        trace = torch.trace(self.S) + 1e-12
        c = float(self.eigs[0] / trace) if trace > 0 else 0.0
        w = 1.0 / (1.0 + torch.exp(-self.alpha*(self.c_target - c)))

        s = w * a + (1 - w) * n
        if loss_per_seq is not None and gamma > 0:
            s = s * (loss_per_seq.detach() ** gamma)
        return s, a, n, w, c

    def step_select(self, model: nn.Module, x_cand: torch.Tensor, y_cand: torch.Tensor,
                    gamma: float=0.0):
        with torch.no_grad():
            logits = model(x_cand)
            loss_tok = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                       y_cand.view(-1), reduction='none')
            loss_seq = loss_tok.view(x_cand.size(0), x_cand.size(1)).mean(dim=1)

        phi = self.project_logits_grad(logits, y_cand)
        self._update_spectrum(phi)
        scores, a, n, w, c = self.score(phi, loss_per_seq=loss_seq, gamma=gamma)
        order = torch.argsort(scores, descending=True)
        return order, {"scores":scores, "align":a, "novel":n, "w":w, "c":c,
                       "loss_seq":loss_seq, "phi":phi}

# ----------------------------- training & evaluation -----------------------------

def evaluate_ppl(model: nn.Module, data: np.ndarray, block_size: int, device: str, max_batches:int=50, batch_size:int=64):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(max_batches):
            xb, yb = get_batch(data, block_size, batch_size, device)
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   yb.view(-1), reduction='mean')
            losses.append(loss.item())
    mean_loss = float(np.mean(losses))
    ppl = math.exp(mean_loss)
    model.train()
    return mean_loss, ppl

def cosine_lr(step, warmup, total_steps, base_lr):
    if step < warmup:
        return base_lr * (step / max(1, warmup))
    # cosine decay to 10% of base_lr
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.1 * base_lr + 0.9 * base_lr * 0.5 * (1 + math.cos(math.pi * (1 - progress)))

def train_one_method(method: str,
                     train_data: np.ndarray,
                     val_data: np.ndarray,
                     vocab_size: int,
                     args) -> Dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout
    )
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    selector = None
    if method == "agop_sc":
        selector = AGOPSpectralSelector(vocab_size, m=args.m, k=args.k, beta=args.beta,
                                        c_target=args.c_target, alpha=args.alpha, device=device)

    # logs
    log_path = os.path.join(LOG_DIR, f"{method}.csv")
    with open(log_path, "w") as f:
        f.write("step,train_loss,val_loss,val_ppl,lr,spec_c,align_w\n")

    best_val = 1e9
    step_to_threshold = None
    threshold = args.ppl_threshold

    for step in range(1, args.steps + 1):
        # cosine lr
        lr = cosine_lr(step, args.warmup, args.steps, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # candidate pool
        x_cand, y_cand = get_batch(train_data, args.block_size, args.candidate_batch, device)

        if method == "random":
            order = torch.randperm(args.candidate_batch, device=device)
            select_idx = order[:args.batch_size]

        elif method in ("loss_curriculum", "self_paced", "anti_curriculum"):
            with torch.no_grad():
                logits = model(x_cand)
                loss_tok = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                           y_cand.view(-1), reduction='none')
                loss_seq = loss_tok.view(args.candidate_batch, args.block_size).mean(dim=1)
            if method == "loss_curriculum":
                order = torch.argsort(loss_seq, descending=False) # easy first
                select_idx = order[:args.batch_size]
            elif method == "anti_curriculum":
                order = torch.argsort(loss_seq, descending=True) # hard first
                select_idx = order[:args.batch_size]
            else:  # self_paced
                # linear schedule on percentile
                p0, p1 = 0.2, 0.9
                frac = min(1.0, step / max(1, int(0.6 * args.steps)))
                pct = p0 * (1 - frac) + p1 * frac
                kth = int(max(1, math.floor(pct * args.candidate_batch)))
                thresh = torch.kthvalue(loss_seq, k=kth).values.item()
                mask = loss_seq <= thresh
                idx = torch.nonzero(mask, as_tuple=False).view(-1)
                if idx.numel() >= args.batch_size:
                    select_idx = idx[:args.batch_size]
                else:
                    # pad with easiest
                    order = torch.argsort(loss_seq, descending=False)
                    select_idx = order[:args.batch_size]

        elif method == "agop_sc":
            order, info = selector.step_select(model, x_cand, y_cand, gamma=args.gamma)
            select_idx = order[:args.batch_size]
        else:
            raise ValueError(f"Unknown method {method}")

        # true update
        x_sel = x_cand[select_idx]
        y_sel = y_cand[select_idx]
        logits = model(x_sel)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               y_sel.view(-1), reduction='mean')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.eval_interval == 0 or step == 1 or step == args.steps:
            val_loss, val_ppl = evaluate_ppl(model, val_data, args.block_size, device,
                                             max_batches=args.eval_batches, batch_size=64)
            spec_c = ""
            align_w = ""
            if method == "agop_sc":
                # record spectral concentration and weight
                trace = float(torch.trace(selector.S).item())
                c = float(selector.eigs[0].item() / max(trace, 1e-12)) if trace>0 else 0.0
                # compute w for logging (use last step value)
                align_w = 1.0 / (1.0 + math.exp(-selector.alpha*(selector.c_target - c)))
                spec_c = c

            with open(log_path, "a") as f:
                f.write(f"{step},{loss.item():.6f},{val_loss:.6f},{val_ppl:.6f},{lr:.6g},{spec_c},{align_w}\n")

            if val_loss < best_val:
                best_val = val_loss
                ckpt = {
                    "model": model.state_dict(),
                    "config": config.__dict__,
                    "step": step,
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    "method": method
                }
                torch.save(ckpt, os.path.join(CKPT_DIR, f"{method}_best.pt"))

            if step_to_threshold is None and val_ppl <= threshold:
                step_to_threshold = step

    # final save
    ckpt = {
        "model": model.state_dict(),
        "config": config.__dict__,
        "step": args.steps,
        "method": method
    }
    torch.save(ckpt, os.path.join(CKPT_DIR, f"{method}_final.pt"))

    # return stats for table
    # read best val ppl from best checkpoint
    best_ckpt = torch.load(os.path.join(CKPT_DIR, f"{method}_best.pt"), map_location="cpu")
    return {
        "method": method,
        "best_step": int(best_ckpt["step"]),
        "best_val_ppl": float(best_ckpt["val_ppl"]),
        "step_to_threshold": int(step_to_threshold) if step_to_threshold is not None else -1
    }

# ----------------------------- plotting & tables -----------------------------

def summarize_and_plot(methods: List[str], args):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Load logs and plot val ppl curves
    plt.figure()
    for m in methods:
        log_path = os.path.join(LOG_DIR, f"{m}.csv")
        if not os.path.exists(log_path):
            continue
        df = pd.read_csv(log_path)
        plt.plot(df["step"], df["val_ppl"], label=m)
    plt.xlabel("step")
    plt.ylabel("val perplexity")
    plt.legend()
    plt.title("Validation PPL vs step (Tiny Shakespeare)")
    fig_path = os.path.join(FIG_DIR, "val_ppl_vs_step.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Plot spectral concentration for agop_sc if exists
    m = "agop_sc"
    log_path = os.path.join(LOG_DIR, f"{m}.csv")
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        if "spec_c" in df.columns:
            # filter non-empty
            cc = []
            for v in df["spec_c"].values:
                try:
                    cc.append(float(v))
                except:
                    cc.append(np.nan)
            df["spec_c_float"] = cc
            plt.figure()
            plt.plot(df["step"], df["spec_c_float"])
            plt.xlabel("step")
            plt.ylabel("spectral concentration (lambda1 / trace)")
            plt.title("AGOP-SC spectral concentration over training")
            fig2 = os.path.join(FIG_DIR, "agop_sc_spectral_concentration.png")
            plt.savefig(fig2, dpi=150, bbox_inches="tight")
            plt.close()

    # Build summary table
    rows = []
    for m in methods:
        ckpt_path = os.path.join(CKPT_DIR, f"{m}_best.pt")
        log_path = os.path.join(LOG_DIR, f"{m}.csv")
        if os.path.exists(ckpt_path):
            ck = torch.load(ckpt_path, map_location="cpu")
            # try to find step_to_threshold from logs table
            step_to_threshold = -1
            if os.path.exists(log_path):
                df = pd.read_csv(log_path)
                # first step where val_ppl <= threshold
                S = df[df["val_ppl"] <= args.ppl_threshold]
                if len(S) > 0:
                    step_to_threshold = int(S["step"].iloc[0])
            rows.append({
                "method": m,
                "best_step": int(ck["step"]),
                "best_val_ppl": float(ck["val_ppl"]),
                "step_to_threshold": int(step_to_threshold)
            })
    if rows:
        summary = pd.DataFrame(rows)
        tab_path = os.path.join(TAB_DIR, "summary.csv")
        summary.to_csv(tab_path, index=False)

        # Also write a simple Markdown table
        md_path = os.path.join(TAB_DIR, "summary.md")
        with open(md_path, "w") as f:
            f.write(summary.to_markdown(index=False))

# ----------------------------- main -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=6)
    parser.add_argument("--n_embd", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_batches", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--candidate_batch", type=int, default=128)
    parser.add_argument("--method", type=str, default="agop_sc",
                        choices=["agop_sc", "random", "loss_curriculum", "self_paced", "anti_curriculum", "all"])

    # AGOP-SC selector hyperparams
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.98)
    parser.add_argument("--c_target", type=float, default=0.35)
    parser.add_argument("--alpha", type=float, default=20.0)
    parser.add_argument("--gamma", type=float, default=0.0)  # loss^gamma mixing (0 to disable)

    # metrics
    parser.add_argument("--ppl_threshold", type=float, default=1.8)

    args = parser.parse_args()

    data_path = maybe_download_tiny_shakespeare(args.data_dir)
    train_data, val_data, vocab_size, stoi, itos = build_char_dataset(data_path)

    methods = ["agop_sc"] if args.method != "all" else ["random", "loss_curriculum", "self_paced", "anti_curriculum", "agop_sc"]

    results = []
    for m in methods:
        print(f"\n===== Training method: {m} =====")
        stats = train_one_method(m, train_data, val_data, vocab_size, args)
        results.append(stats)
        print("Stats:", stats)

    # summarize & plot
    summarize_and_plot(methods, args)

    # save results json
    with open(os.path.join(EXP_DIR, "run_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nAll done. Artifacts saved under:", EXP_DIR)

if __name__ == "__main__":
    main()
