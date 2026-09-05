"""Final benchmark: AWF vs Dense on TinyStories with enhanced sampling."""
import os, sys, json, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn.functional as F
from awf.core import AWFTransformer, DenseTransformer, num_params
from chat_v2 import ByteTokenizer, generate, BLOCK

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")
BENCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks")
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "tinystories_train.txt")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load validation data
with open(DATA_FILE) as f:
    text = f.read()
text = text[:3_000_000]
n_train = int(0.95 * len(text))
val_text = text[n_train:]
tokenizer = ByteTokenizer()


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, text, bs):
        self.data = tokenizer.encode(text); self.bs = bs
    def __len__(self): return max(0, len(self.data) - self.bs - 1)
    def __getitem__(self, i):
        c = self.data[i:i+self.bs+1]
        return torch.tensor(c[:-1]), torch.tensor(c[1:])

val_ds = TextDataset(val_text, BLOCK)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16)


def evaluate(model, loader, max_batches=80):
    model.eval()
    loss_sum, n, correct, total = 0.0, 0, 0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_batches: break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            V = logits.size(-1)
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction="sum")
            loss_sum += loss.item(); n += y.numel()
            correct += (logits.argmax(-1) == y).sum().item(); total += y.numel()
    return loss_sum/max(n,1), correct/max(total,1)


def diversity_metrics(text):
    if not text: return {}
    chars = list(text)
    bigrams = [text[i:i+2] for i in range(len(text)-1)]
    trigrams = [text[i:i+3] for i in range(len(text)-2)]
    repeat_bigrams = sum(1 for b in bigrams if len(b)==2 and b[0]==b[1]) / max(len(bigrams),1)
    from collections import Counter
    char_counts = Counter(chars)
    total = sum(char_counts.values())
    entropy = -sum((c/total) * math.log2(c/total) for c in char_counts.values() if c > 0)
    longest_run = 1; cur = 1
    for i in range(1, len(chars)):
        if chars[i] == chars[i-1]:
            cur += 1; longest_run = max(longest_run, cur)
        else: cur = 1
    return {
        "unique_bigrams": round(len(set(bigrams))/max(len(bigrams),1), 3),
        "unique_trigrams": round(len(set(trigrams))/max(len(trigrams),1), 3),
        "repetition_2gram": round(repeat_bigrams, 3),
        "char_entropy": round(entropy, 2),
        "longest_run": longest_run,
    }


# Load AWF
print("Loading AWF 10M...")
awf = AWFTransformer(vocab_size=256, d_model=256, n_layers=6, n_heads=8, block_size=BLOCK,
                    residual_rank=16, sparse_k=256,
                    gen_kwargs=dict(n_fourier=16, hidden=128, n_layers=3), gen_grid=16)
state = torch.load(os.path.join(CKPT_DIR, "awf_10m.pt"), map_location=device)
awf.load_state_dict(state["model"])
awf.activate_sparse_corrections()
awf = awf.to(device)
awf.eval()
awf_params = num_params(awf)
awf_storage = awf.total_storage_bytes(2)
awf_step = state.get("step", "?")
awf_epoch = state.get("epoch", "?")
print(f"  AWF: {awf_params:,} params ({awf_params/1e6:.2f}M), storage {awf_storage/1024:.1f}KB, step {awf_step}")

# Load Dense
print("Loading Dense 10M...")
dense = DenseTransformer(vocab_size=256, d_model=256, n_layers=6, n_heads=8, block_size=BLOCK)
dense_ckpt = os.path.join(CKPT_DIR, "dense_10m.pt")
if os.path.exists(dense_ckpt):
    state = torch.load(dense_ckpt, map_location=device)
    dense.load_state_dict(state["model"])
    dense_step = state.get("step", "?")
else:
    dense_step = "?"
dense = dense.to(device)
dense.eval()
dense_params = num_params(dense)
print(f"  Dense: {dense_params:,} params ({dense_params/1e6:.2f}M), step {dense_step}")

# Evaluate
print("\nEvaluating on validation set...")
awf_loss, awf_acc = evaluate(awf, val_loader)
dense_loss, dense_acc = evaluate(dense, val_loader) if os.path.exists(dense_ckpt) else (0, 0)
print(f"  AWF:   val_loss={awf_loss:.4f}, val_acc={awf_acc*100:.2f}%, ppl={math.exp(min(awf_loss,20)):.1f}")
if os.path.exists(dense_ckpt):
    print(f"  Dense: val_loss={dense_loss:.4f}, val_acc={dense_acc*100:.2f}%, ppl={math.exp(min(dense_loss,20)):.1f}")

# Generate samples with enhanced sampling
print("\n=== Text Generation (enhanced sampling: temp=0.7, top_k=30, top_p=0.9, rep_penalty=1.3) ===")
prompts = [
    "Once upon a time",
    "The little girl",
    "A boy named Tom",
    "In the forest",
    "Today I learned",
    "The dog ran",
]

results = {"dense": {}, "awf": {}}
for p in prompts:
    print(f"\n--- {p!r} ---")
    a = generate(awf, tokenizer, p, n_tokens=200, temperature=0.7, top_k=30, top_p=0.9,
                 repetition_penalty=1.3, seed=42, device=device)
    print(f"AWF: {a[:200]}")
    if os.path.exists(dense_ckpt):
        d = generate(dense, tokenizer, p, n_tokens=200, temperature=0.7, top_k=30, top_p=0.9,
                     repetition_penalty=1.3, seed=42, device=device)
        print(f"DENSE: {d[:200]}")
        results["dense"][p] = {"text": d, "metrics": diversity_metrics(d)}
    results["awf"][p] = {"text": a, "metrics": diversity_metrics(a)}

# Average diversity
print("\n=== AVERAGE DIVERSITY METRICS ===")
metrics_to_avg = ["unique_bigrams", "unique_trigrams", "repetition_2gram", "char_entropy", "longest_run"]
awf_avg = {m: sum(results["awf"][p]["metrics"][m] for p in prompts) / len(prompts) for m in metrics_to_avg}
print(f"{'Metric':<25} {'AWF':>12}")
for m in metrics_to_avg:
    print(f"{m:<25} {awf_avg[m]:>12.3f}")
if os.path.exists(dense_ckpt):
    dense_avg = {m: sum(results["dense"][p]["metrics"][m] for p in prompts) / len(prompts) for m in metrics_to_avg}
    print(f"\n{'Metric':<25} {'Dense':>12} {'AWF':>12} {'Winner':>10}")
    print("-" * 60)
    for m in metrics_to_avg:
        d_val = dense_avg[m]; a_val = awf_avg[m]
        if m in ["unique_bigrams", "unique_trigrams", "char_entropy"]:
            winner = "AWF" if a_val > d_val else "Dense"
        else:
            winner = "AWF" if a_val < d_val else "Dense"
        print(f"{m:<25} {d_val:>12.3f} {a_val:>12.3f} {winner:>10}")

# Save
summary = {
    "awf": {"params": awf_params, "val_loss": awf_loss, "val_acc": awf_acc,
            "val_ppl": math.exp(min(awf_loss, 20)), "storage_bytes": awf_storage,
            "generator_params": num_params(awf.generator), "step": awf_step, "epoch": awf_epoch},
    "param_compression": dense_params / awf_params,
    "storage_compression": (dense_params * 4) / awf_storage,
    "awf_avg_metrics": awf_avg,
    "samples": results,
}
if os.path.exists(dense_ckpt):
    summary["dense"] = {"params": dense_params, "val_loss": dense_loss, "val_acc": dense_acc,
                        "val_ppl": math.exp(min(dense_loss, 20)), "storage_bytes": dense_params * 4,
                        "step": dense_step}
    summary["dense_avg_metrics"] = dense_avg

with open(os.path.join(BENCH_DIR, "10m_benchmark.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n=== FINAL SUMMARY ===")
print(f"AWF: {awf_params:,} params ({awf_params/1e6:.2f}M), {awf_storage/1024:.1f}KB, val_acc={awf_acc*100:.2f}%, step {awf_step}")
if os.path.exists(dense_ckpt):
    print(f"Dense: {dense_params:,} params ({dense_params/1e6:.2f}M), {dense_params*4/1024:.1f}KB, val_acc={dense_acc*100:.2f}%")
    print(f"Param compression: {dense_params/awf_params:.2f}x")
    print(f"Storage compression: {(dense_params*4)/awf_storage:.2f}x")
