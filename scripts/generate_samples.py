"""Generate side-by-side text samples from Dense LLM and AWF LLM.

Use this as a demo to show AWF generates more coherent, less repetitive text.
"""
import os, sys, json, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn.functional as F
from awf.core import AWFTransformer, DenseTransformer, num_params

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "corpus.txt")
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")

with open(DATA) as f: text = f.read()
chars = sorted(set(text))
VOCAB = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda ids: "".join(itos[i] for i in ids)

BLOCK = 64

def generate(model, prompt, n_tokens=150, temperature=0.6, top_k=5, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
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


# Load models
print("Loading AWF LLM...")
awf = AWFTransformer(vocab_size=VOCAB, d_model=64, n_layers=2, n_heads=4, block_size=BLOCK,
                    residual_rank=8, gen_kwargs=dict(n_fourier=16, hidden=96, n_layers=3), gen_grid=16)
awf.load_state_dict(torch.load(os.path.join(CKPT_DIR, "awf_llm.pt")))
print(f"  AWF: {num_params(awf):,} params, {awf.total_storage_bytes(2)/1024:.1f}KB")

print("Loading Dense LLM...")
dense = DenseTransformer(vocab_size=VOCAB, d_model=64, n_layers=2, n_heads=4, block_size=BLOCK)
dense.load_state_dict(torch.load(os.path.join(CKPT_DIR, "dense_llm.pt")))
print(f"  Dense: {num_params(dense):,} params, {num_params(dense)*4/1024:.1f}KB")

# Generate samples
prompts = [
    "To be, or not",
    "The Sun is",
    "Once upon",
    "Hello, how",
    "Friends, Romans",
    "The ocean covers",
]

print("\n=== Text Generation: Dense vs AWF ===\n")
samples = {}
for p in prompts:
    print(f"--- Prompt: {p!r} ---")
    d = generate(dense, p, n_tokens=120, seed=42)
    a = generate(awf, p, n_tokens=120, seed=42)
    print(f"DENSE ({num_params(dense):,} params): {d}")
    print(f"AWF   ({num_params(awf):,} params): {a}")
    print()
    samples[p] = {"dense": d, "awf": a}

out = os.path.join(CKPT_DIR, "..", "benchmarks", "final_samples.json")
with open(out, "w") as f:
    json.dump(samples, f, indent=2)
print(f"\nSaved to {out}")
