#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NanoGPT-style experiments (Tiny Shakespeare / TinyStories) — v3
AGOP-SC vs baselines with several robustness improvements:

New in v3:
- Larger TinyStories cap by default (--ts_max_chars 8M)
- AGOP score gating by phi-norm (robust energy scaling)
- Diversity re-ranking (repulsion) inside AGOP selection
- Dynamic c_target schedule (early alignment -> later novelty)
- Lower default weight_decay (0.05) and longer cosine warmup option
- Helpful auto-fixes (n_head divisibility; BPE Top-K default)

Artifacts: /experiments/nanogpt_curriculum/ (fallback: ./experiments/nanogpt_curriculum/)
"""

import os, math, json, argparse
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------- utils: I/O paths -----------------------------

def get_exp_dir():
    preferred = "/experiments/nanogpt_curriculum/"
    fallback = "./experiments/nanogpt_curriculum/"
    try:
        os.makedirs(preferred, exist_ok=True)
        with open(os.path.join(preferred, ".write_test"), "w") as f: f.write("ok")
        os.remove(os.path.join(preferred, ".write_test"))
        return preferred
    except Exception:
        os.makedirs(fallback, exist_ok=True)
        return fallback

EXP_DIR = get_exp_dir()
LOG_DIR = os.path.join(EXP_DIR, "logs")
FIG_DIR = os.path.join(EXP_DIR, "figs")
CKPT_DIR = os.path.join(EXP_DIR, "ckpts")
TAB_DIR = os.path.join(EXP_DIR, "tables")
for d in (LOG_DIR, FIG_DIR, CKPT_DIR, TAB_DIR):
    os.makedirs(d, exist_ok=True)

# ----------------------------- data -----------------------------

TS_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def maybe_download_tiny_shakespeare(data_dir: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, "input.txt")
    if os.path.exists(data_path): return data_path
    try:
        import urllib.request
        print("Downloading Tiny Shakespeare...")
        urllib.request.urlretrieve(TS_URL, data_path)
        print(f"Saved to {data_path}")
    except Exception as e:
        print("WARNING: download failed; using small fallback text.")
        sample = ("To be, or not to be, that is the question:\n"
                  "Whether 'tis nobler in the mind to suffer\n"
                  "The slings and arrows of outrageous fortune,\n") * 512
        with open(data_path, "w", encoding="utf-8") as f: f.write(sample)
    return data_path

def build_char_dataset(path: str, split_ratio: float = 0.99):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(list(set(text)))
    stoi = {ch:i for i,ch in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    n = int(len(data) * split_ratio)
    return data[:n], data[n:], len(chars)

def build_tinystories_dataset(tokenizer: str="byte", max_chars:int=8_000_000,
                              val_ratio:float=0.01, seed:int=1337):
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError("Please `pip install datasets` for --dataset tinystories") from e
    print("Loading TinyStories (train split)...")
    ds = load_dataset("roneneldan/TinyStories", split="train")
    texts, chars = [], 0
    for rec in ds:
        s = rec["text"] + "\n"
        texts.append(s); chars += len(s)
        if chars >= max_chars: break
    full_text = "".join(texts)
    if tokenizer == "gpt2":
        try:
            import tiktoken
        except Exception as e:
            raise RuntimeError("Please `pip install tiktoken` for --tokenizer gpt2") from e
        enc = tiktoken.get_encoding("gpt2")
        ids = np.array(enc.encode_ordinary(full_text), dtype=np.int64)
        vocab_size = enc.n_vocab
    elif tokenizer == "byte":
        ids = np.frombuffer(full_text.encode("utf-8"), dtype=np.uint8).astype(np.int64)
        vocab_size = 256
    else:
        raise ValueError("tokenizer must be 'byte' or 'gpt2'")
    n = int(len(ids) * (1 - val_ratio))
    print(f"TinyStories tokens: total={len(ids)}, train={n}, val={len(ids)-n}, vocab={vocab_size}, tokenizer={tokenizer}")
    return ids[:n], ids[n:], vocab_size

def get_batch(data: np.ndarray, block_size: int, batch_size: int, device: str):
    ix = np.random.randint(0, len(data) - block_size - 1, size=(batch_size,))
    x = np.stack([data[i:i+block_size] for i in ix])
    y = np.stack([data[i+1:i+block_size+1] for i in ix])
    return torch.from_numpy(x).long().to(device), torch.from_numpy(y).long().to(device)

# ----------------------------- model -----------------------------

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512
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
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1,config.block_size,config.block_size))
    def forward(self, x):
        B,T,C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
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
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, idx):
        device = idx.device
        b, t = idx.size(); assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)[None, :, :]
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

# ----------------------------- AGOP-SC -----------------------------

def _orthonormalize(U: torch.Tensor) -> torch.Tensor:
    Q, _ = torch.linalg.qr(U, mode='reduced')
    return Q

class AGOPSpectralSelector:
    def __init__(self, vocab_size:int, m:int=128, k:int=16, beta:float=0.995,
                 c_target:float=0.5, alpha:float=15.0, device='cuda',
                 phi_norm_gamma:float=0.25, phi_norm_clip:float=3.0,
                 div_lambda:float=0.2, preselect_factor:float=2.0):
        self.V = vocab_size; self.m=m; self.k=k; self.beta=beta
        self.c_target = c_target; self.alpha=alpha; self.device=device
        self.phi_norm_gamma = phi_norm_gamma; self.phi_norm_clip = phi_norm_clip
        self.div_lambda = div_lambda; self.preselect_factor = preselect_factor
        self.R = torch.randn(self.V, self.m, device=device) / math.sqrt(self.m)
        self.S = torch.zeros(self.m, self.m, device=device)
        self.U = torch.eye(self.m, self.k, device=device)
        self.eigs = torch.ones(self.k, device=device)
        self.norm_med = 1.0

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
        Bc, T, V = logits.shape; assert V == self.V
        if (topk is None) or (topk <= 0) or (topk >= V):
            p = F.softmax(logits, dim=-1)
            pR = torch.matmul(p, self.R)
            Rt = self.R[targets]
            phi = (pR - Rt).sum(dim=1)
            return phi
        topv, topi = torch.topk(logits, k=topk, dim=-1)
        p_top = F.softmax(topv, dim=-1)
        R_top = self.R[topi]
        pR_top = (p_top.unsqueeze(-1) * R_top).sum(dim=2)
        Rt = self.R[targets]
        phi = (pR_top - Rt).sum(dim=1)
        return phi

    @torch.no_grad()
    def score(self, phi: torch.Tensor, loss_per_seq: torch.Tensor=None, gamma: float=0.0, c_target_eff: Optional[float]=None):
        proj = phi @ self.U
        num = (proj ** 2).sum(dim=1)
        den = (phi ** 2).sum(dim=1).clamp_min(1e-12)
        a = (num / den).clamp(0, 1)
        n = 1.0 - a

        # energy gating by phi-norm (robust)
        phi_norm = den.sqrt()
        med = torch.median(phi_norm).item()
        if med > 0:
            self.norm_med = 0.99*self.norm_med + 0.01*med
        scale = (phi_norm / (self.norm_med + 1e-12)).clamp(1/self.phi_norm_clip, self.phi_norm_clip)
        s = a  # will mix with novelty after computing w below
        if self.phi_norm_gamma > 0:
            s = s * (scale ** self.phi_norm_gamma)

        # dynamic c_target schedule support
        trace = float(torch.trace(self.S).item())
        lam1 = float(self.eigs[0].item())
        c = (lam1 / max(trace, 1e-12)) if trace > 0.0 else 0.0
        c_t = self.c_target if c_target_eff is None else float(c_target_eff)
        w = 1.0 / (1.0 + math.exp(-self.alpha * (c_t - c)))
        s = w * s + (1.0 - w) * (1.0 - a)  # mix novelty

        if loss_per_seq is not None and gamma > 0:
            s = s * (loss_per_seq.detach() ** gamma)
        return s, a, (1.0 - a), w, c

    def _diverse_rerank(self, idx_sorted: torch.Tensor, phi: torch.Tensor, B: int):
        # greedy DPP-ish re-ranking with cosine repulsion
        m = min(int(self.preselect_factor * B), idx_sorted.numel())
        pool = idx_sorted[:m]
        phi_sel = []
        chosen = []
        # precompute normalized
        phi_n = phi / (phi.norm(dim=1, keepdim=True).clamp_min(1e-12))
        for _ in range(min(B, m)):
            if not chosen:
                chosen_idx = pool[0].item()
                chosen.append(chosen_idx); phi_sel.append(phi_n[chosen_idx])
                continue
            # penalize by max cosine similarity to chosen
            sims = torch.stack([ (phi_n[pool] @ psel).squeeze(-1) for psel in phi_sel ], dim=1).abs().max(dim=1).values
            penal = self.div_lambda * sims
            # choose with minimum penalty (we want lower similarity)
            # here we simply pick next with smallest penal (equivalently highest (orig rank - penalty))
            # since orig rank already sorted by score, we subtract penalty to re-rank
            # implement by argmin of penal
            best_local = torch.argmin(penal).item()
            chosen_idx = pool[best_local].item()
            chosen.append(chosen_idx); phi_sel.append(phi_n[chosen_idx])
            # remove chosen from pool
            mask = torch.ones_like(pool, dtype=torch.bool)
            mask[best_local] = False
            pool = pool[mask]
            if pool.numel() == 0: break
        return torch.tensor(chosen, device=idx_sorted.device, dtype=idx_sorted.dtype)

    def step_select(self, model: nn.Module, x_cand: torch.Tensor, y_cand: torch.Tensor,
                    gamma: float=0.0, topk_logits: Optional[int]=None, c_target_eff: Optional[float]=None,
                    rand_frac: float=0.2, B:int=32):
        with torch.no_grad():
            logits = model(x_cand)
            loss_tok = F.cross_entropy(logits.view(-1, logits.size(-1)), y_cand.view(-1), reduction='none')
            loss_seq = loss_tok.view(x_cand.size(0), x_cand.size(1)).mean(dim=1)
        phi = self.project_logits_grad(logits, y_cand, topk=topk_logits)
        self._update_spectrum(phi)
        scores, a, n, w, c = self.score(phi, loss_per_seq=loss_seq, gamma=gamma, c_target_eff=c_target_eff)
        order = torch.argsort(scores, descending=True)

        # diversity re-ranking on the top subset
        pick_reranked = self._diverse_rerank(order, phi, B=max(1, int((1.0 - rand_frac) * B)))

        # mix with random for diversity floor
        num_pick = pick_reranked.numel()
        rest = torch.randperm(x_cand.size(0), device=x_cand.device)
        if num_pick > 0:
            rest = rest[~torch.isin(rest, pick_reranked)]
        need = max(0, B - num_pick)
        select_idx = torch.cat([pick_reranked, rest[:need]])
        return select_idx, {"scores":scores, "align":a, "novel":n, "w":w, "c":c, "phi":phi, "loss_seq":loss_seq}

# ----------------------------- train & eval -----------------------------

def evaluate_ppl(model: nn.Module, data: np.ndarray, block_size: int, device: str, max_batches:int=50, batch_size:int=64):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(max_batches):
            xb, yb = get_batch(data, block_size, batch_size, device)
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), reduction='mean')
            losses.append(loss.item())
    mean_loss = float(np.mean(losses)) if losses else 0.0
    ppl = math.exp(mean_loss) if mean_loss < 20 else float('inf')
    model.train()
    return mean_loss, ppl

def cosine_lr(step, warmup, total_steps, base_lr):
    if step < warmup: return base_lr * (step / max(1, warmup))
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.1 * base_lr + 0.9 * base_lr * 0.5 * (1 + math.cos(math.pi * (1 - progress)))

def train_one_method(method: str, train_data: np.ndarray, val_data: np.ndarray, vocab_size: int, args) -> Dict:
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = GPTConfig(vocab_size=vocab_size, block_size=args.block_size, n_layer=args.n_layer,
                       n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout)
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    selector = None
    if method == 'agop_sc':
        selector = AGOPSpectralSelector(vocab_size, m=args.m, k=args.k, beta=args.beta,
                                        c_target=args.c_target_min, alpha=args.alpha, device=device,
                                        phi_norm_gamma=args.phi_norm_gamma, phi_norm_clip=args.phi_norm_clip,
                                        div_lambda=args.div_lambda, preselect_factor=args.preselect_factor)

    log_path = os.path.join(LOG_DIR, f"{args.dataset}_{args.tokenizer}_{method}.csv")
    with open(log_path, "w") as f: f.write("step,train_loss,val_loss,val_ppl,lr,spec_c,align_w\n")

    best_val = 1e9; step_to_threshold = None

    for step in range(1, args.steps + 1):
        lr = cosine_lr(step, args.warmup, args.steps, args.lr)
        for pg in optimizer.param_groups: pg["lr"] = lr

        x_cand, y_cand = get_batch(train_data, args.block_size, args.candidate_batch, device)

        if method == 'random':
            order = torch.randperm(args.candidate_batch, device=device)
            select_idx = order[:args.batch_size]

        elif method in ('loss_curriculum', 'self_paced', 'anti_curriculum'):
            with torch.no_grad():
                logits = model(x_cand)
                loss_tok = F.cross_entropy(logits.view(-1, logits.size(-1)), y_cand.view(-1), reduction='none')
                loss_seq = loss_tok.view(args.candidate_batch, args.block_size).mean(dim=1)
            if method == 'loss_curriculum':
                frac = min(1.0, step / max(1, int(0.7 * args.steps)))
                pct = 0.3 + 0.6 * frac
                kth = max(1, int(math.floor(pct * args.candidate_batch)))
                thresh = torch.kthvalue(loss_seq, k=kth).values
                eligible = torch.nonzero(loss_seq <= thresh, as_tuple=False).view(-1)
                num_rand = int(args.batch_size * args.baseline_rand_frac)
                num_pick = args.batch_size - num_rand
                pick = eligible[torch.randperm(eligible.numel(), device=device)[:max(0, min(num_pick, eligible.numel()))]] if eligible.numel() > 0 else eligible
                rest = torch.randperm(args.candidate_batch, device=device)
                if pick.numel() > 0: rest = rest[~torch.isin(rest, pick)]
                select_idx = torch.cat([pick, rest[:max(0, args.batch_size - pick.numel())]])
            elif method == 'anti_curriculum':
                order = torch.argsort(loss_seq, descending=True); select_idx = order[:args.batch_size]
            else:
                p0, p1 = 0.2, 0.9
                frac = min(1.0, step / max(1, int(0.6 * args.steps)))
                pct = p0 * (1 - frac) + p1 * frac
                kth = int(max(1, math.floor(pct * args.candidate_batch)))
                thresh = torch.kthvalue(loss_seq, k=kth).values
                eligible = torch.nonzero(loss_seq <= thresh, as_tuple=False).view(-1)
                num_rand = int(args.batch_size * args.baseline_rand_frac)
                num_pick = args.batch_size - num_rand
                pick = eligible[torch.randperm(eligible.numel(), device=device)[:max(0, min(num_pick, eligible.numel()))]] if eligible.numel() > 0 else eligible
                rest = torch.randperm(args.candidate_batch, device=device)
                if pick.numel() > 0: rest = rest[~torch.isin(rest, pick)]
                select_idx = torch.cat([pick, rest[:max(0, args.batch_size - pick.numel())]])

        elif method == 'agop_sc':
            if step < args.agop_start_step:
                order = torch.randperm(args.candidate_batch, device=device)
                num_rand = int(args.batch_size * args.agop_rand_frac)
                pick = order[:args.batch_size - num_rand]
                rest = order[args.batch_size - num_rand:args.batch_size]
                select_idx = torch.cat([pick, rest])
            else:
                # dynamic c_target schedule: linear from min->max across training
                frac = step / max(1, args.steps)
                c_target_eff = args.c_target_min + (args.c_target_max - args.c_target_min) * frac
                topk = args.topk_logits if args.topk_logits > 0 else None
                select_idx, info = selector.step_select(model, x_cand, y_cand, gamma=args.gamma,
                                                        topk_logits=topk, c_target_eff=c_target_eff,
                                                        rand_frac=args.agop_rand_frac, B=args.batch_size)
        else:
            raise ValueError(f"Unknown method {method}")

        x_sel = x_cand[select_idx]; y_sel = y_cand[select_idx]
        logits = model(x_sel)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y_sel.view(-1), reduction='mean')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.eval_interval == 0 or step == 1 or step == args.steps:
            val_loss, val_ppl = evaluate_ppl(model, val_data, args.block_size, device,
                                             max_batches=args.eval_batches, batch_size=64)
            spec_c = ""; align_w = ""
            if method == 'agop_sc' and step >= args.agop_start_step:
                trace = float(torch.trace(selector.S).item())
                c = (float(selector.eigs[0].item()) / max(trace, 1e-12)) if trace>0 else 0.0
                # compute current c_target_eff for log
                frac = step / max(1, args.steps)
                c_target_eff = args.c_target_min + (args.c_target_max - args.c_target_min) * frac
                align_w = 1.0 / (1.0 + math.exp(-selector.alpha * (c_target_eff - c)))
                spec_c = c
            with open(log_path, "a") as f:
                f.write(f"{step},{loss.item():.6f},{val_loss:.6f},{val_ppl:.6f},{lr:.6g},{spec_c},{align_w}\n")

            if val_loss < best_val:
                best_val = val_loss
                ckpt = {"model": model.state_dict(), "config": config.__dict__, "step": step,
                        "val_loss": val_loss, "val_ppl": val_ppl, "method": method,
                        "dataset": args.dataset, "tokenizer": args.tokenizer}
                torch.save(ckpt, os.path.join(CKPT_DIR, f"{args.dataset}_{args.tokenizer}_{method}_best.pt"))

            if args.ppl_threshold > 0 and val_ppl <= args.ppl_threshold and step_to_threshold is None:
                step_to_threshold = step

    # save final
    torch.save({"model": model.state_dict(), "config": config.__dict__, "step": args.steps, "method": method},
               os.path.join(CKPT_DIR, f"{args.dataset}_{args.tokenizer}_{method}_final.pt"))
    best_ckpt = torch.load(os.path.join(CKPT_DIR, f"{args.dataset}_{args.tokenizer}_{method}_best.pt"), map_location="cpu")
    return {"method": method, "best_step": int(best_ckpt["step"]),
            "best_val_ppl": float(best_ckpt["val_ppl"]),
            "step_to_threshold": int(step_to_threshold) if step_to_threshold is not None else -1}

# ----------------------------- plotting -----------------------------

def summarize_and_plot(methods: List[str], args):
    import pandas as pd
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # ppl curves
    plt.figure()
    for m in methods:
        log_path = os.path.join(LOG_DIR, f"{args.dataset}_{args.tokenizer}_{m}.csv")
        if not os.path.exists(log_path): continue
        df = pd.read_csv(log_path)
        plt.plot(df["step"], df["val_ppl"], label=m)
    plt.xlabel("step"); plt.ylabel("val perplexity"); plt.legend()
    plt.title(f"Validation PPL vs step ({args.dataset}, {args.tokenizer})")
    plt.savefig(os.path.join(FIG_DIR, f"{args.dataset}_{args.tokenizer}_val_ppl_vs_step.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # spectral concentration
    log_path = os.path.join(LOG_DIR, f"{args.dataset}_{args.tokenizer}_agop_sc.csv")
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        if "spec_c" in df.columns:
            cc = []
            for v in df["spec_c"].values:
                try: cc.append(float(v))
                except: cc.append(np.nan)
            df["spec_c_float"] = cc
            plt.figure()
            plt.plot(df["step"], df["spec_c_float"])
            plt.xlabel("step"); plt.ylabel("spectral concentration (lambda1/trace)")
            plt.title(f"AGOP-SC spectral concentration ({args.dataset}, {args.tokenizer})")
            plt.savefig(os.path.join(FIG_DIR, f"{args.dataset}_{args.tokenizer}_agop_sc_spectral_concentration.png"),
                        dpi=150, bbox_inches='tight')
            plt.close()

    # table
    rows = []
    for m in methods:
        ck = os.path.join(CKPT_DIR, f"{args.dataset}_{args.tokenizer}_{m}_best.pt")
        if os.path.exists(ck):
            ckpt = torch.load(ck, map_location="cpu")
            rows.append({"method": m, "best_step": int(ckpt["step"]), "best_val_ppl": float(ckpt["val_ppl"])})
    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(TAB_DIR, f"{args.dataset}_{args.tokenizer}_summary.csv"), index=False)
        with open(os.path.join(TAB_DIR, f"{args.dataset}_{args.tokenizer}_summary.md"), "w") as f:
            f.write(df.to_markdown(index=False))

# ----------------------------- main -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="tinystories", choices=["tinyshakespeare", "tinystories"])
    parser.add_argument("--tokenizer", type=str, default="gpt2", choices=["char", "byte", "gpt2"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--ts_max_chars", type=int, default=8_000_000)
    parser.add_argument("--val_ratio", type=float, default=0.01)

    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_embd", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_batches", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--candidate_batch", type=int, default=256)
    parser.add_argument("--method", type=str, default="agop_sc",
                        choices=["agop_sc", "random", "loss_curriculum", "self_paced", "anti_curriculum", "all"])
    parser.add_argument("--baseline_rand_frac", type=float, default=0.25)

    # AGOP
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--beta", type=float, default=0.995)
    parser.add_argument("--c_target_min", type=float, default=0.4)
    parser.add_argument("--c_target_max", type=float, default=0.6)
    parser.add_argument("--alpha", type=float, default=15.0)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--agop_rand_frac", type=float, default=0.2)
    parser.add_argument("--agop_start_step", type=int, default=300)
    parser.add_argument("--phi_norm_gamma", type=float, default=0.25)
    parser.add_argument("--phi_norm_clip", type=float, default=3.0)
    parser.add_argument("--div_lambda", type=float, default=0.2)
    parser.add_argument("--preselect_factor", type=float, default=2.0)

    parser.add_argument("--topk_logits", type=int, default=64)
    parser.add_argument("--ppl_threshold", type=float, default=-1.0)

    args = parser.parse_args()

    # auto-fix heads
    if args.n_embd % args.n_head != 0:
        preferred = max(1, args.n_embd // 64)
        divs = [d for d in range(1, args.n_embd+1) if args.n_embd % d == 0]
        args.n_head = min(divs, key=lambda d: abs(d - preferred))
        print(f"[auto-fix] n_head={args.n_head} (head_dim={args.n_embd//args.n_head})")
    # BPE defaults
    if args.dataset == "tinystories" and args.tokenizer == "gpt2":
        if args.block_size < 256:
            print("Increasing block_size to 256 for gpt2 tokenizer."); args.block_size = 256
        if args.topk_logits <= 0:
            args.topk_logits = 64
            print("[auto] set --topk_logits 64 for BPE vocab.")

    # build data
    if args.dataset == "tinyshakespeare":
        if args.tokenizer != "char":
            print("For tinyshakespeare, tokenizer forced to 'char'"); args.tokenizer = "char"
        path = maybe_download_tiny_shakespeare(args.data_dir)
        train_data, val_data, vocab_size = build_char_dataset(path, split_ratio=1.0 - args.val_ratio)
    else:
        if args.tokenizer == "char":
            print("For tinystories, prefer byte/gpt2; switching to byte")
            args.tokenizer = "byte"
        train_data, val_data, vocab_size = build_tinystories_dataset(
            tokenizer=args.tokenizer, max_chars=args.ts_max_chars, val_ratio=args.val_ratio, seed=args.seed
        )

    methods = ["agop_sc"] if args.method != "all" else ["random","loss_curriculum","self_paced","anti_curriculum","agop_sc"]
    results = []
    for m in methods:
        print(f"\n===== Training method: {m}  ({args.dataset}, {args.tokenizer}) =====")
        stats = train_one_method(m, train_data, val_data, vocab_size, args)
        results.append(stats)
        print("Stats:", stats)

    summarize_and_plot(methods, args)
    with open(os.path.join(EXP_DIR, f"{args.dataset}_{args.tokenizer}_run_summary_v3.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nAll done. Artifacts saved under:", EXP_DIR)

if __name__ == "__main__":
    main()
