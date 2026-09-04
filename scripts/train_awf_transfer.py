"""
AWF LLM with knowledge transfer from a pre-trained dense model.

Key insight: instead of training AWF from scratch (which takes many epochs to
bootstrap the generator), we initialize the AWF's generator + low-rank
to approximately reproduce a pre-trained dense model's weights.

Strategy:
  1. Train dense LLM (already done)
  2. For each AWF layer, fit the generator + low-rank to reproduce dense's weight matrix
  3. Then continue training end-to-end to refine
"""
import os, sys, time, json, math
sys.path.insert(0, "/home/z/my-project/awf")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from core import AWFTransformer, DenseTransformer, num_params

DATA = "/home/z/my-project/data/corpus.txt"
OUT_DIR = "/home/z/my-project/download/AWF_LLM"

with open(DATA) as f:
    text = f.read()
chars = sorted(set(text))
VOCAB = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda ids: "".join(itos[i] for i in ids)

BLOCK = 64
BATCH = 32
class TextDataset(Dataset):
    def __init__(self, text, block_size):
        self.data = encode(text); self.block_size = block_size
    def __len__(self): return max(0, len(self.data) - self.block_size - 1)
    def __getitem__(self, i):
        chunk = self.data[i:i + self.block_size + 1]
        return torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)

n_train = int(0.9 * len(text))
train_ds = TextDataset(text[:n_train], BLOCK)
val_ds = TextDataset(text[n_train:], BLOCK)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False)

def evaluate(model, loader):
    model.eval()
    loss_sum, n = 0.0, 0
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction="sum")
            loss_sum += loss.item(); n += y.numel()
            pred = logits.argmax(-1)
            correct += (pred == y).sum().item(); total += y.numel()
    return loss_sum / n, correct / total

# Load dense model
dense = DenseTransformer(vocab_size=VOCAB, d_model=96, n_layers=3, n_heads=4, block_size=BLOCK)
dense.load_state_dict(torch.load(os.path.join(OUT_DIR, "dense_llm.pt")))
dense.eval()
dense_val_loss, dense_val_acc = evaluate(dense, val_loader)
print(f"Dense: val_loss={dense_val_loss:.4f}, val_acc={dense_val_acc*100:.2f}%")

# Build AWF model
torch.manual_seed(0)
awf = AWFTransformer(vocab_size=VOCAB, d_model=96, n_layers=3, n_heads=4, block_size=BLOCK,
                     residual_rank=8, gen_kwargs=dict(n_fourier=16, hidden=96, n_layers=3), gen_grid=16)
awf_params = num_params(awf)
awf_storage = awf.total_storage_bytes(2)
print(f"AWF params: {awf_params:,} (gen: {num_params(awf.generator):,})  storage: {awf_storage/1024:.1f}KB")

# --- Knowledge transfer: fit each AWF layer to reproduce dense's weights ---
# Match by SHAPE — only transfer where dense has matching (in, out) Linear.
# nn.MultiheadAttention uses combined qkv; skip those (AWF has separate q/k/v/o).
# We focus on FFN (up, down) and head — these are pure nn.Linear.
print("\n--- Knowledge transfer: fitting AWF to dense weights via SVD ---")

# Collect dense pure-Linear layers
dense_linears = []
for blk in dense.blocks:
    dense_linears.append(("ffn_up", blk.ff[0].weight))      # (4d, d)
    dense_linears.append(("ffn_down", blk.ff[2].weight))    # (d, 4d)
dense_linears.append(("head", dense.head.weight))           # (vocab, d)

# AWF pure-Linear layers in same order (FFN up/down + head)
awf_linears = []
for blk in awf.blocks:
    awf_linears.append(blk.up)
    awf_linears.append(blk.down)
awf_linears.append(awf.head)

for i, (awf_layer, (name, dense_W)) in enumerate(zip(awf_linears, dense_linears)):
    # dense_W shape (out, in); AWF produces W of (in, out) then transposes in forward
    # So target for awf is dense_W.T of shape (in, out)
    target = dense_W.detach().T.clone()  # (in, out)
    M, N = target.shape
    if min(M, N) < awf_layer.cfg.residual_rank:
        print(f"  Layer {i} ({name}): skip (too small)")
        continue
    U, S, Vh = torch.linalg.svd(target, full_matrices=False)
    r = awf_layer.cfg.residual_rank
    awf_layer.U.data = (U[:, :r] * S[:r].unsqueeze(0)).clone()
    awf_layer.V.data = Vh[:r, :].clone()
    # Initialize generator output to small (so low-rank dominates)
    awf_layer.generator.out_scale.data[awf_layer.cfg.layer_id] = 0.001
    awf_layer.generator.out_bias.data[awf_layer.cfg.layer_id] = 0.0
    print(f"  Layer {i} ({name}): target {target.shape}, top-{r} SVs: {[f'{s:.3f}' for s in S[:r].tolist()]}")

# Evaluate after transfer (no training yet)
awf.eval()
transfer_val_loss, transfer_val_acc = evaluate(awf, val_loader)
print(f"\nAfter SVD transfer (no training): val_loss={transfer_val_loss:.4f}, val_acc={transfer_val_acc*100:.2f}%")
print(f"  (dense was {dense_val_loss:.4f} / {dense_val_acc*100:.2f}%)")

# Fine-tune end-to-end
print("\n--- Fine-tuning AWF end-to-end ---")
opt = torch.optim.AdamW(awf.parameters(), lr=3e-4, weight_decay=0.01)
t0 = time.time()
for ep in range(5):
    awf.train()
    ep_loss, n_seen = 0.0, 0
    for x, y in train_loader:
        opt.zero_grad()
        logits = awf(x)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(awf.parameters(), 1.0)
        opt.step()
        ep_loss += loss.item() * x.size(0); n_seen += x.size(0)
    val_loss, val_acc = evaluate(awf, val_loader)
    ppl = math.exp(min(val_loss, 20))
    dt = time.time() - t0
    print(f"  [awf-ft] ep{ep+1:02d} train_loss={ep_loss/n_seen:.4f} val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}% ppl={ppl:.1f} ({dt:.0f}s)")

torch.save(awf.state_dict(), os.path.join(OUT_DIR, "awf_llm.pt"))

# Final
awf_val_loss, awf_val_acc = evaluate(awf, val_loader)
print(f"\n=== FINAL ===")
print(f"Dense: {num_params(dense):,} params, {num_params(dense)*4/1024:.1f}KB, val_loss={dense_val_loss:.4f}, val_acc={dense_val_acc*100:.2f}%")
print(f"AWF:   {awf_params:,} params, {awf_storage/1024:.1f}KB, val_loss={awf_val_loss:.4f}, val_acc={awf_val_acc*100:.2f}%")
print(f"Param compression: {num_params(dense)/awf_params:.2f}x")
print(f"Storage compression: {num_params(dense)*4/awf_storage:.2f}x")
