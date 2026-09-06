"""Continue training AWF LLM to push accuracy higher and generate better samples.

Designed to run on a CPU laptop with 8GB RAM. Takes ~5-10 minutes.
"""
import os, sys, time, math, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from awf.core import AWFTransformer, DenseTransformer, num_params

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "corpus.txt")
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")

BLOCK = 64
BATCH = 32
D_MODEL = 64
N_LAYERS = 2
N_HEADS = 4
N_EPOCHS = 6

with open(DATA) as f: text = f.read()
chars = sorted(set(text))
VOCAB = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda ids: "".join(itos[i] for i in ids)


class TextDataset(Dataset):
    def __init__(self, text, bs):
        self.data = encode(text); self.bs = bs
    def __len__(self): return max(0, len(self.data) - self.bs - 1)
    def __getitem__(self, i):
        c = self.data[i:i+self.bs+1]
        return torch.tensor(c[:-1], dtype=torch.long), torch.tensor(c[1:], dtype=torch.long)


n_train = int(0.9 * len(text))
train_ds = TextDataset(text[:n_train], BLOCK)
val_ds = TextDataset(text[n_train:], BLOCK)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH)
print(f"Corpus: {len(text)} chars | Train: {len(train_ds)} | Val: {len(val_ds)}")


def evaluate(model, loader):
    model.eval()
    loss_sum, n, correct, total = 0.0, 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            V = logits.size(-1)
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction="sum")
            loss_sum += loss.item(); n += y.numel()
            correct += (logits.argmax(-1) == y).sum().item(); total += y.numel()
    return loss_sum/n, correct/total


def generate(model, prompt, n_tokens=150, temperature=0.6, top_k=5):
    model.eval()
    ids = encode(prompt)
    if not ids: ids = [0]
    with torch.no_grad():
        for _ in range(n_tokens):
            x = torch.tensor(ids[-BLOCK:], dtype=torch.long).unsqueeze(0)
            logits = model(x)
            nl = logits[0, -1] / max(temperature, 0.01)
            if top_k > 0:
                v, _ = torch.topk(nl, min(top_k, VOCAB))
                nl[nl < v[-1]] = -float("inf")
            probs = F.softmax(nl, dim=-1)
            ids.append(torch.multinomial(probs, 1).item())
    return decode(ids)


# Load existing AWF checkpoint
torch.manual_seed(0)
awf = AWFTransformer(vocab_size=VOCAB, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
                    block_size=BLOCK, residual_rank=8,
                    gen_kwargs=dict(n_fourier=16, hidden=96, n_layers=3), gen_grid=16)
if os.path.exists(os.path.join(CKPT_DIR, "awf_llm.pt")):
    awf.load_state_dict(torch.load(os.path.join(CKPT_DIR, "awf_llm.pt")))
    print(f"Loaded existing AWF checkpoint: {num_params(awf):,} params")
else:
    print("No existing checkpoint, training from scratch")
val_loss, val_acc = evaluate(awf, val_loader)
print(f"Initial: val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%")

# Continue training
opt = torch.optim.AdamW(awf.parameters(), lr=6e-4, weight_decay=0.01)
t0 = time.time()
for ep in range(N_EPOCHS):
    awf.train()
    ep_loss, n = 0.0, 0
    for x, y in train_loader:
        opt.zero_grad()
        logits = awf(x)
        V = logits.size(-1)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(awf.parameters(), 1.0)
        opt.step()
        ep_loss += loss.item() * x.size(0); n += x.size(0)
    val_loss, val_acc = evaluate(awf, val_loader)
    ppl = math.exp(min(val_loss, 20))
    print(f"  [awf] ep{ep+1:02d} train={ep_loss/n:.4f} val={val_loss:.4f} acc={val_acc*100:.2f}% ppl={ppl:.1f} ({time.time()-t0:.0f}s)")
    if val_acc > 0.85:
        print("  Reached 85%+ val acc — stopping early")
        break

torch.save(awf.state_dict(), os.path.join(CKPT_DIR, "awf_llm.pt"))

# Generate samples
print("\n=== Text Generation ===")
prompts = ["To be, or not", "The Sun is", "Once upon", "Hello, how", "Friends, Romans"]
samples = {}
for p in prompts:
    sample = generate(awf, p, n_tokens=150)
    print(f"\n--- {p!r} ---\n{sample}")
    samples[p] = sample

with open(os.path.join(CKPT_DIR, "..", "benchmarks", "final_samples.json"), "w") as f:
    json.dump(samples, f, indent=2)

# Final stats
print(f"\n=== FINAL ===")
print(f"AWF params: {num_params(awf):,}, storage: {awf.total_storage_bytes(2)/1024:.1f}KB")
print(f"Val loss: {val_loss:.4f}, Val acc: {val_acc*100:.2f}%")
print(f"Perplexity: {math.exp(min(val_loss, 20)):.2f}")
