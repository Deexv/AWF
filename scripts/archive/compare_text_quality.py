"""Demonstrate that AWF generates better text than Dense, even when Dense has higher val accuracy.

The key insight: Dense memorizes the small corpus and gets 99% val accuracy,
but its GENERATED TEXT collapses to repetition ("ssssss..."). AWF's structural
regularization produces more varied, language-like output.

This script runs a "Turing test" style evaluation:
1. Generate text from both models with the same prompts
2. Compute diversity metrics (unique n-grams, repetition rate)
3. Show side-by-side samples
"""
import os, sys, json, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn.functional as F
from awf.core import AWFTransformer, DenseTransformer, num_params

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "corpus.txt")
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")
BENCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks")

with open(DATA) as f: text = f.read()
chars = sorted(set(text))
VOCAB = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda ids: "".join(itos[i] for i in ids)

BLOCK = 48

def generate(model, prompt, n_tokens=200, temperature=0.6, top_k=5, seed=None):
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


def diversity_metrics(text):
    """Compute text diversity metrics.
    Returns dict with:
    - unique_chars: fraction of unique characters
    - unique_bigrams: fraction of unique bigrams (KEY METRIC for repetition collapse)
    - unique_trigrams: fraction of unique trigrams
    - repetition_2gram: rate of repeated bigrams (lower is better)
    - char_diversity: how varied the character distribution is (entropy)
    - longest_repeat: longest run of the same character (lower is better)
    """
    if not text:
        return {}
    chars = list(text)
    unique_chars = len(set(chars)) / max(len(chars), 1)

    # Character-level bigrams and trigrams (catches "ssss..." repetition)
    bigrams = [text[i:i+2] for i in range(len(text)-1)]
    unique_bigrams = len(set(bigrams)) / max(len(bigrams), 1)
    trigrams = [text[i:i+3] for i in range(len(text)-2)]
    unique_trigrams = len(set(trigrams)) / max(len(trigrams), 1)

    # Repetition rate: fraction of bigrams that are the same char repeated ("ss", "ee", etc.)
    repeat_bigrams = sum(1 for b in bigrams if len(b) == 2 and b[0] == b[1]) / max(len(bigrams), 1)

    # Character entropy (higher = more varied)
    from collections import Counter
    char_counts = Counter(chars)
    total = sum(char_counts.values())
    entropy = -sum((c/total) * math.log2(c/total) for c in char_counts.values() if c > 0)

    # Longest run of same character
    longest_run = 1
    current_run = 1
    for i in range(1, len(chars)):
        if chars[i] == chars[i-1]:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 1

    return {
        "unique_chars": round(unique_chars, 3),
        "unique_bigrams": round(unique_bigrams, 3),
        "unique_trigrams": round(unique_trigrams, 3),
        "repetition_2gram": round(repeat_bigrams, 3),
        "char_entropy": round(entropy, 2),
        "longest_run": longest_run,
    }


# Load both models
print("Loading AWF LLM...")
awf = AWFTransformer(vocab_size=VOCAB, d_model=64, n_layers=2, n_heads=4, block_size=BLOCK,
                    residual_rank=8, sparse_k=64,
                    gen_kwargs=dict(n_fourier=16, hidden=96, n_layers=3), gen_grid=16)
awf.load_state_dict(torch.load(os.path.join(CKPT_DIR, "awf_llm.pt")))
print(f"  AWF: {num_params(awf):,} params, {awf.total_storage_bytes(2)/1024:.1f}KB")

print("Loading Dense LLM...")
dense = DenseTransformer(vocab_size=VOCAB, d_model=64, n_layers=2, n_heads=4, block_size=BLOCK)
dense.load_state_dict(torch.load(os.path.join(CKPT_DIR, "dense_llm.pt")))
print(f"  Dense: {num_params(dense):,} params, {num_params(dense)*4/1024:.1f}KB")

# Generate longer samples for fair comparison
prompts = [
    "To be, or not",
    "The Sun is",
    "Once upon a time",
    "Hello, how are",
    "Friends, Romans",
    "The ocean covers",
    "A computer is",
    "The heart is",
]

print("\n" + "="*80)
print("TEXT GENERATION COMPARISON: Dense vs AWF")
print("="*80)

results = {"dense": {}, "awf": {}}
for p in prompts:
    print(f"\n--- Prompt: {p!r} ---")
    d = generate(dense, p, n_tokens=200, seed=42)
    a = generate(awf, p, n_tokens=200, seed=42)
    print(f"DENSE ({num_params(dense):,} params):")
    print(f"  {d}")
    print(f"AWF   ({num_params(awf):,} params):")
    print(f"  {a}")
    # Compute diversity metrics
    d_metrics = diversity_metrics(d)
    a_metrics = diversity_metrics(a)
    print(f"Diversity metrics:")
    print(f"  Dense: unique_bigrams={d_metrics['unique_bigrams']}, repetition_2gram={d_metrics['repetition_2gram']}, longest_run={d_metrics['longest_run']}")
    print(f"  AWF:   unique_bigrams={a_metrics['unique_bigrams']}, repetition_2gram={a_metrics['repetition_2gram']}, longest_run={a_metrics['longest_run']}")
    results["dense"][p] = {"text": d, "metrics": d_metrics}
    results["awf"][p] = {"text": a, "metrics": a_metrics}

# Compute average diversity
print("\n" + "="*80)
print("AVERAGE DIVERSITY METRICS")
print("(Higher unique_bigrams/trigrams = better. Lower repetition_2gram = better.")
print(" Lower longest_run = better. Higher char_entropy = better.)")
print("="*80)

metrics_to_avg = ["unique_bigrams", "unique_trigrams", "repetition_2gram",
                  "char_entropy", "longest_run"]
dense_avg = {m: sum(results["dense"][p]["metrics"][m] for p in prompts) / len(prompts)
             for m in metrics_to_avg}
awf_avg = {m: sum(results["awf"][p]["metrics"][m] for p in prompts) / len(prompts)
           for m in metrics_to_avg}

print(f"\n{'Metric':<25} {'Dense':>12} {'AWF':>12} {'Winner':>10}")
print("-" * 60)
for m in metrics_to_avg:
    d_val = dense_avg[m]
    a_val = awf_avg[m]
    # For unique_bigrams, unique_trigrams, char_entropy: higher is better
    # For repetition_2gram, longest_run: lower is better
    if m in ["unique_bigrams", "unique_trigrams", "char_entropy"]:
        winner = "AWF" if a_val > d_val else "Dense"
    else:
        winner = "AWF" if a_val < d_val else "Dense"
    print(f"{m:<25} {d_val:>12.3f} {a_val:>12.3f} {winner:>10}")

# AWF wins if it has more unique bigrams AND less repetition
awf_wins_diversity = awf_avg["unique_bigrams"] > dense_avg["unique_bigrams"]
awf_wins_repetition = awf_avg["repetition_2gram"] < dense_avg["repetition_2gram"]
awf_wins_longest_run = awf_avg["longest_run"] < dense_avg["longest_run"]

print(f"\n=== KEY FINDINGS ===")
print(f"AWF generates MORE diverse text (unique bigrams): {awf_wins_diversity}")
print(f"AWF has LESS character repetition: {awf_wins_repetition}")
print(f"AWF has SHORTER character runs (no 'sssss...' collapse): {awf_wins_longest_run}")

# Save
results["summary"] = {
    "dense_avg": dense_avg,
    "awf_avg": awf_avg,
    "awf_wins_diversity": awf_wins_diversity,
    "awf_wins_repetition": awf_wins_repetition,
    "awf_wins_longest_run": awf_wins_longest_run,
}
with open(os.path.join(BENCH_DIR, "diversity_comparison.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to {os.path.join(BENCH_DIR, 'diversity_comparison.json')}")
