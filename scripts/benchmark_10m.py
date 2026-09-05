"""Final benchmark comparison: AWF vs Dense on TinyStories 10M models."""
import os, sys, json, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn.functional as F
from awf.core import AWFTransformer, DenseTransformer, num_params
from chat import ByteTokenizer, generate, BLOCK, D_MODEL, N_LAYERS, N_HEADS, RESIDUAL_RANK, SPARSE_K

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")
BENCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks")

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "tinystories_train.txt")

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


def evaluate(model, loader, max_batches=50):
    model.eval()
    loss_sum, n, correct, total = 0.0, 0, 0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_batches: break
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
awf = AWFTransformer(vocab_size=256, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
                    block_size=BLOCK, residual_rank=RESIDUAL_RANK, sparse_k=SPARSE_K,
                    gen_kwargs=dict(n_fourier=16, hidden=128, n_layers=3), gen_grid=16)
state = torch.load(os.path.join(CKPT_DIR, "awf_10m.pt"), map_location="cpu")
awf.load_state_dict(state["model"])
awf.activate_sparse_corrections()
awf.eval()
awf_params = num_params(awf)
awf_storage = awf.total_storage_bytes(2)
print(f"  AWF: {awf_params:,} params ({awf_params/1e6:.2f}M), storage {awf_storage/1024:.1f}KB")

# Load Dense
print("Loading Dense 10M...")
dense = DenseTransformer(vocab_size=256, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS, block_size=BLOCK)
state = torch.load(os.path.join(CKPT_DIR, "dense_10m.pt"), map_location="cpu")
dense.load_state_dict(state["model"])
dense.eval()
dense_params = num_params(dense)
print(f"  Dense: {dense_params:,} params ({dense_params/1e6:.2f}M)")

# Evaluate both
print("\nEvaluating on validation set...")
awf_loss, awf_acc = evaluate(awf, val_loader, max_batches=80)
dense_loss, dense_acc = evaluate(dense, val_loader, max_batches=80)
print(f"  AWF:   val_loss={awf_loss:.4f}, val_acc={awf_acc*100:.2f}%, ppl={math.exp(min(awf_loss,20)):.1f}")
print(f"  Dense: val_loss={dense_loss:.4f}, val_acc={dense_acc*100:.2f}%, ppl={math.exp(min(dense_loss,20)):.1f}")

# Generate samples and compute diversity
print("\n=== Text Generation & Diversity Comparison ===")
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
    d = generate(dense, tokenizer, p, n_tokens=200, temperature=0.7, top_k=10, seed=42)
    a = generate(awf, tokenizer, p, n_tokens=200, temperature=0.7, top_k=10, seed=42)
    print(f"DENSE ({dense_params:,}): {d[:150]}...")
    print(f"AWF   ({awf_params:,}): {a[:150]}...")
    d_m = diversity_metrics(d)
    a_m = diversity_metrics(a)
    print(f"  Dense: unique_bigrams={d_m['unique_bigrams']}, rep_2gram={d_m['repetition_2gram']}, longest_run={d_m['longest_run']}")
    print(f"  AWF:   unique_bigrams={a_m['unique_bigrams']}, rep_2gram={a_m['repetition_2gram']}, longest_run={a_m['longest_run']}")
    results["dense"][p] = {"text": d, "metrics": d_m}
    results["awf"][p] = {"text": a, "metrics": a_m}

# Average diversity
print("\n=== AVERAGE DIVERSITY METRICS ===")
metrics_to_avg = ["unique_bigrams", "unique_trigrams", "repetition_2gram", "char_entropy", "longest_run"]
dense_avg = {m: sum(results["dense"][p]["metrics"][m] for p in prompts) / len(prompts) for m in metrics_to_avg}
awf_avg = {m: sum(results["awf"][p]["metrics"][m] for p in prompts) / len(prompts) for m in metrics_to_avg}
print(f"{'Metric':<25} {'Dense':>12} {'AWF':>12} {'Winner':>10}")
print("-" * 60)
for m in metrics_to_avg:
    d_val = dense_avg[m]; a_val = awf_avg[m]
    if m in ["unique_bigrams", "unique_trigrams", "char_entropy"]:
        winner = "AWF" if a_val > d_val else "Dense"
    else:
        winner = "AWF" if a_val < d_val else "Dense"
    print(f"{m:<25} {d_val:>12.3f} {a_val:>12.3f} {winner:>10}")

# Final summary
summary = {
    "dense": {"params": dense_params, "val_loss": dense_loss, "val_acc": dense_acc,
              "val_ppl": math.exp(min(dense_loss, 20)), "storage_bytes": dense_params * 4},
    "awf": {"params": awf_params, "val_loss": awf_loss, "val_acc": awf_acc,
            "val_ppl": math.exp(min(awf_loss, 20)), "storage_bytes": awf_storage,
            "generator_params": num_params(awf.generator)},
    "param_compression": dense_params / awf_params,
    "storage_compression": (dense_params * 4) / awf_storage,
    "dense_avg_metrics": dense_avg,
    "awf_avg_metrics": awf_avg,
    "awf_wins_diversity": awf_avg["unique_bigrams"] > dense_avg["unique_bigrams"],
    "awf_wins_repetition": awf_avg["repetition_2gram"] < dense_avg["repetition_2gram"],
    "awf_wins_longest_run": awf_avg["longest_run"] < dense_avg["longest_run"],
}

with open(os.path.join(BENCH_DIR, "10m_benchmark.json"), "w") as f:
    json.dump({"summary": summary, "samples": results}, f, indent=2)

print(f"\n=== FINAL SUMMARY ===")
print(f"Dense: {dense_params:,} params ({dense_params/1e6:.2f}M), {dense_params*4/1024:.1f}KB, val_acc={dense_acc*100:.2f}%")
print(f"AWF:   {awf_params:,} params ({awf_params/1e6:.2f}M), {awf_storage/1024:.1f}KB, val_acc={awf_acc*100:.2f}%")
print(f"Param compression: {dense_params/awf_params:.2f}x")
print(f"Storage compression: {(dense_params*4)/awf_storage:.2f}x")
print(f"AWF wins diversity: {summary['awf_wins_diversity']}")
print(f"AWF wins repetition: {summary['awf_wins_repetition']}")
print(f"AWF wins longest_run: {summary['awf_wins_longest_run']}")
