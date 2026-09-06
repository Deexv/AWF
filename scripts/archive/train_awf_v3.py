"""Train only the AWF LLM (dense already trained)."""
import os, sys, time, math, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from awf.core import AWFTransformer, DenseTransformer, num_params

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "corpus.txt")
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")
BENCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks")

BLOCK = 48              # smaller block for faster training
BATCH = 64              # bigger batch = fewer steps
D_MODEL = 64
N_LAYERS = 2
N_HEADS = 4
RESIDUAL_RANK = 8
SPARSE_K = 64           # smaller sparse for speed
N_EPOCHS_BOOTSTRAP = 2
N_EPOCHS_TOTAL = 6

with open(DATA) as f: text = f.read()
chars = sorted(set(text))
VOCAB = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda ids: "".join(itos[i] for i in ids)
print(f"Corpus: {len(text):,} chars, vocab: {VOCAB}")


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


def generate(model, prompt, n_tokens=150, temperature=0.6, top_k=5, seed=None):
    if seed is not None: torch.manual_seed(seed)
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


# Dense comparison (load existing checkpoint to save time)
print("\n=== Loading Dense LLM (already trained) ===")
dense = DenseTransformer(vocab_size=VOCAB, d_model=D_MODEL, n_layers=N_LAYERS,
                         n_heads=N_HEADS, block_size=BLOCK)
dense.load_state_dict(torch.load(os.path.join(CKPT_DIR, "dense_llm.pt")))
dense.eval()
dense_params = num_params(dense)
dense_val_loss, dense_val_acc = evaluate(dense, val_loader)
print(f"Dense: {dense_params:,} params, val_acc={dense_val_acc*100:.2f}%, ppl={math.exp(min(dense_val_loss,20)):.2f}")


# Build and train AWF
print(f"\n=== Training AWF LLM ===")
torch.manual_seed(0)
awf = AWFTransformer(vocab_size=VOCAB, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
                    block_size=BLOCK, residual_rank=RESIDUAL_RANK, sparse_k=SPARSE_K,
                    gen_kwargs=dict(n_fourier=16, hidden=96, n_layers=3), gen_grid=16)
awf_params = num_params(awf)
awf_storage_fp16 = awf.total_storage_bytes(2)
print(f"AWF params: {awf_params:,} (gen: {num_params(awf.generator):,})")
print(f"AWF storage fp16: {awf_storage_fp16/1024:.1f}KB")
print(f"Param compression vs dense: {dense_params/awf_params:.2f}x")
print(f"Storage compression vs dense fp32: {dense_params*4/awf_storage_fp16:.2f}x")

opt = torch.optim.AdamW(awf.parameters(), lr=1.5e-3, weight_decay=0.01)
t0 = time.time()
for ep in range(N_EPOCHS_TOTAL):
    if ep == N_EPOCHS_BOOTSTRAP:
        print(f"  Activating sparse corrections (k={SPARSE_K}/layer)")
        awf.activate_sparse_corrections()
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
    # Save after every epoch so we make progress even if timeout
    torch.save(awf.state_dict(), os.path.join(CKPT_DIR, "awf_llm.pt"))
awf_val_loss, awf_val_acc = val_loss, val_acc
print(f"\nAWF final: val_acc={awf_val_acc*100:.2f}%, ppl={math.exp(min(awf_val_loss,20)):.2f}")

# Apply int8 quantization
print("\nApplying int8 per-row quantization...")
awf_q = AWFTransformer(vocab_size=VOCAB, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
                       block_size=BLOCK, residual_rank=RESIDUAL_RANK, sparse_k=SPARSE_K,
                       gen_kwargs=dict(n_fourier=16, hidden=96, n_layers=3), gen_grid=16)
awf_q.load_state_dict(torch.load(os.path.join(CKPT_DIR, "awf_llm.pt")))
awf_q.activate_sparse_corrections()
awf_storage_int8 = awf_q.quantize_for_inference("int8_per_row")
awf_val_loss_q, awf_val_acc_q = evaluate(awf_q, val_loader)
print(f"int8: val_acc={awf_val_acc_q*100:.2f}%, storage={awf_storage_int8/1024:.1f}KB")

# Dense comparison - already trained above, just re-evaluate
print(f"\nDense: {dense_params:,} params, val_acc={dense_val_acc*100:.2f}%, ppl={math.exp(min(dense_val_loss,20)):.2f}")

# Generate samples
print("\n=== Text Generation ===")
prompts = ["To be, or not", "The Sun is", "Once upon a time", "Hello, how are", "Friends, Romans", "The ocean covers"]
samples = {"dense": {}, "awf_fp16": {}, "awf_int8": {}}
for p in prompts:
    d = generate(dense, p, n_tokens=120, seed=42)
    a = generate(awf, p, n_tokens=120, seed=42)
    a_q = generate(awf_q, p, n_tokens=120, seed=42)
    print(f"\n--- {p!r} ---")
    print(f"DENSE:    {d}")
    print(f"AWF fp16: {a}")
    print(f"AWF int8: {a_q}")
    samples["dense"][p] = d
    samples["awf_fp16"][p] = a
    samples["awf_int8"][p] = a_q

with open(os.path.join(BENCH_DIR, "final_samples.json"), "w") as f:
    json.dump(samples, f, indent=2)

# Save results
results = {
    "dense": {"params": dense_params, "storage_bytes": dense_params*4,
              "val_loss": dense_val_loss, "val_acc": dense_val_acc,
              "val_ppl": math.exp(min(dense_val_loss, 20))},
    "awf_fp16": {"params": awf_params, "storage_bytes": awf_storage_fp16,
                 "generator_params": num_params(awf.generator),
                 "val_loss": awf_val_loss, "val_acc": awf_val_acc,
                 "val_ppl": math.exp(min(awf_val_loss, 20))},
    "awf_int8": {"params": awf_params, "storage_bytes": awf_storage_int8,
                 "val_loss": awf_val_loss_q, "val_acc": awf_val_acc_q,
                 "val_ppl": math.exp(min(awf_val_loss_q, 20))},
    "param_compression": dense_params / awf_params,
    "storage_compression_fp16": (dense_params * 4) / awf_storage_fp16,
    "storage_compression_int8": (dense_params * 4) / awf_storage_int8,
    "awf_beats_dense_on_acc": awf_val_acc > dense_val_acc,
    "config": {"d_model": D_MODEL, "n_layers": N_LAYERS, "n_heads": N_HEADS,
               "residual_rank": RESIDUAL_RANK, "sparse_k": SPARSE_K,
               "block_size": BLOCK, "n_epochs": N_EPOCHS_TOTAL,
               "corpus_chars": len(text)},
}
with open(os.path.join(BENCH_DIR, "llm_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\n=== FINAL ===")
print(f"Dense:     {dense_params:,} params, {dense_params*4/1024:.1f}KB, val_acc={dense_val_acc*100:.2f}%")
print(f"AWF fp16:  {awf_params:,} params, {awf_storage_fp16/1024:.1f}KB, val_acc={awf_val_acc*100:.2f}%")
print(f"AWF int8:  {awf_params:,} params, {awf_storage_int8/1024:.1f}KB, val_acc={awf_val_acc_q*100:.2f}%")
print(f"Param compression: {dense_params/awf_params:.2f}x")
print(f"Storage compression fp16: {(dense_params*4)/awf_storage_fp16:.2f}x")
print(f"Storage compression int8: {(dense_params*4)/awf_storage_int8:.2f}x")
