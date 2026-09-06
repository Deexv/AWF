"""
Train and benchmark AWF-LLM vs Dense-LLM on a real text corpus.

Goal: prove AWF can generate coherent text rivaling a dense baseline
at much smaller param count and storage.

Pipeline:
  1. Load corpus, build char tokenizer
  2. Train Dense-LLM (4-layer d=128 transformer)
  3. Train AWF-LLM (matched architecture)
  4. Generate text samples from both
  5. Compute perplexity, params, storage
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
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cpu"
SEED = 0
torch.manual_seed(SEED)

# Read corpus
with open(DATA) as f:
    text = f.read()
print(f"Corpus: {len(text):,} chars")

# Char tokenizer
chars = sorted(set(text))
VOCAB = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda ids: "".join(itos[i] for i in ids)
print(f"Vocab: {VOCAB} chars")

# Save tokenizer
with open(os.path.join(OUT_DIR, "tokenizer.json"), "w") as f:
    json.dump({"chars": chars}, f)

# Dataset
class TextDataset(Dataset):
    def __init__(self, text, block_size):
        self.data = encode(text)
        self.block_size = block_size
    def __len__(self):
        return max(0, len(self.data) - self.block_size - 1)
    def __getitem__(self, i):
        chunk = self.data[i:i + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

BLOCK = 64
BATCH = 32
N_EPOCHS = 4
LR = 1e-3
D_MODEL = 96
N_LAYERS = 3
N_HEADS = 4

# Split 90/10 — use a smaller subset for tractable CPU training
n_train = int(0.9 * len(text))
train_ds = TextDataset(text[:n_train], BLOCK)
val_ds = TextDataset(text[n_train:], BLOCK)
print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False)


def evaluate(model, loader):
    model.eval()
    loss_sum, n = 0.0, 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction="sum")
            loss_sum += loss.item()
            n += y.numel()
            pred = logits.argmax(-1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return loss_sum / n, correct / total


def train(model, name, n_epochs, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    best_val = float("inf")
    t0 = time.time()
    for ep in range(n_epochs):
        model.train()
        ep_loss, n_seen = 0.0, 0
        for x, y in train_loader:
            opt.zero_grad()
            logits = model(x)
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * x.size(0)
            n_seen += x.size(0)
        val_loss, val_acc = evaluate(model, val_loader)
        ppl = math.exp(min(val_loss, 20))  # clamp for safety
        dt = time.time() - t0
        print(f"  [{name}] ep{ep+1:02d} train_loss={ep_loss/n_seen:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}% ppl={ppl:.1f} ({dt:.0f}s)")
        best_val = min(best_val, val_loss)
    return best_val


def generate(model, prompt, n_tokens=100, temperature=0.8, top_k=10):
    """Generate text from a prompt."""
    model.eval()
    ids = encode(prompt)
    if len(ids) == 0:
        ids = [0]
    if len(ids) > BLOCK:
        ids = ids[-BLOCK:]
    with torch.no_grad():
        for _ in range(n_tokens):
            x = torch.tensor(ids[-BLOCK:], dtype=torch.long).unsqueeze(0)
            logits = model(x)
            next_logits = logits[0, -1] / temperature
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, VOCAB))
                next_logits[next_logits < v[-1]] = -float("inf")
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            ids.append(next_id)
    return decode(ids)


# ---- Train Dense LLM ----
print(f"\n=== Training Dense LLM ({N_LAYERS}-layer d={D_MODEL}) ===")
torch.manual_seed(SEED)
dense = DenseTransformer(vocab_size=VOCAB, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS, block_size=BLOCK)
dense_params = num_params(dense)
print(f"Dense params: {dense_params:,}  storage fp32: {dense_params*4/1024:.1f}KB")
dense_val_loss = train(dense, "dense", N_EPOCHS, LR)
torch.save(dense.state_dict(), os.path.join(OUT_DIR, "dense_llm.pt"))


# ---- Train AWF LLM ----
print(f"\n=== Training AWF LLM ({N_LAYERS}-layer d={D_MODEL}, rank=8, gen h=96) ===")
torch.manual_seed(SEED)
awf = AWFTransformer(vocab_size=VOCAB, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS, block_size=BLOCK,
                     residual_rank=8, gen_kwargs=dict(n_fourier=16, hidden=96, n_layers=3), gen_grid=16)
awf_params = num_params(awf)
awf_storage = awf.total_storage_bytes(2)
print(f"AWF params: {awf_params:,} (gen: {num_params(awf.generator):,})  storage fp16: {awf_storage/1024:.1f}KB")
print(f"Param compression: {dense_params/awf_params:.2f}x")
print(f"Storage compression: {dense_params*4/awf_storage:.2f}x")
# AWF needs more epochs to bootstrap the generator
awf_val_loss = train(awf, "awf", N_EPOCHS + 1, LR * 1.5)
torch.save(awf.state_dict(), os.path.join(OUT_DIR, "awf_llm.pt"))


# ---- Generate text samples ----
print(f"\n=== Text Generation ===")
prompts = [
    "To be, or not to be",
    "The Sun is the star",
    "Once upon a time",
    "Hello, how are you",
]

samples = {}
for prompt in prompts:
    print(f"\n--- Prompt: {prompt!r} ---")
    dense_sample = generate(dense, prompt, n_tokens=120)
    print(f"DENSE: {dense_sample}")
    print()
    awf_sample = generate(awf, prompt, n_tokens=120)
    print(f"AWF:   {awf_sample}")
    samples[prompt] = {"dense": dense_sample, "awf": awf_sample}

# Save samples
with open(os.path.join(OUT_DIR, "generated_samples.json"), "w") as f:
    json.dump(samples, f, indent=2)

# Save results
results = {
    "dense": {
        "params": dense_params,
        "storage_bytes": dense_params * 4,
        "val_loss": dense_val_loss,
        "val_ppl": math.exp(min(dense_val_loss, 20)),
    },
    "awf": {
        "params": awf_params,
        "storage_bytes": awf_storage,
        "generator_params": num_params(awf.generator),
        "val_loss": awf_val_loss,
        "val_ppl": math.exp(min(awf_val_loss, 20)),
    },
    "param_compression": dense_params / awf_params,
    "storage_compression": (dense_params * 4) / awf_storage,
}
with open(os.path.join(OUT_DIR, "llm_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\n=== FINAL RESULTS ===")
print(f"Dense: {dense_params:,} params, {dense_params*4/1024:.1f}KB, val_loss={dense_val_loss:.4f} (ppl={math.exp(min(dense_val_loss,20)):.1f})")
print(f"AWF:   {awf_params:,} params, {awf_storage/1024:.1f}KB, val_loss={awf_val_loss:.4f} (ppl={math.exp(min(awf_val_loss,20)):.1f})")
print(f"Param compression: {dense_params/awf_params:.2f}x")
print(f"Storage compression: {dense_params*4/awf_storage:.2f}x")
print(f"\nSee {OUT_DIR}/generated_samples.json for text samples")
