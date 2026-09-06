"""Tune event-driven v2 parameters to find maximum speedup.

Tests multiple configurations to find the one that achieves the best
val_loss in the same wall-clock time.
"""
import os, sys, time, json, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from awf.core import AWFTransformer, num_params
from awf.event_training import StandardTrainer
from awf.event_training_v2 import GradientEventTrainerV2
from benchmark_event_v2 import ByteTokenizer, TextDataset, evaluate, build_model, BLOCK, BATCH, DATA_FILE

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--time_budget", type=int, default=120)
args = parser.parse_args()

# Load data
with open(DATA_FILE) as f: text = f.read()
text = text[:1_000_000]
tokenizer = ByteTokenizer()
n_train = int(0.95 * len(text))
train_ds = TextDataset(text[:n_train], tokenizer, BLOCK)
val_ds = TextDataset(text[n_train:], tokenizer, BLOCK)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH)

# Initial
init_model = build_model()
init_loss, init_acc = evaluate(init_model, val_loader)
print(f"Init: val_loss={init_loss:.4f}")
del init_model

# Standard baseline
print(f"\n--- STANDARD ({args.time_budget}s) ---")
m = build_model()
t = StandardTrainer(m, lr=1e-3)
t0 = time.time()
n_std = 0
while time.time() - t0 < args.time_budget:
    for x, y in train_loader:
        if time.time() - t0 >= args.time_budget: break
        t.step(x, y); n_std += 1
std_loss, std_acc = evaluate(m, val_loader)
std_delta = init_loss - std_loss
print(f"  {n_std} steps, val_loss={std_loss:.4f} delta={std_delta:.4f}")
t.close(); del m, t

# Test multiple v2 configs
configs = [
    {"name": "v2_conservative", "warmup_fraction": 0.10, "gradient_norm_threshold": 0.01,
     "reuse_threshold": 0.08, "max_staleness": 6, "max_steps": 500},
    {"name": "v2_moderate", "warmup_fraction": 0.05, "gradient_norm_threshold": 0.05,
     "reuse_threshold": 0.12, "max_staleness": 8, "max_steps": 500},
    {"name": "v2_aggressive", "warmup_fraction": 0.03, "gradient_norm_threshold": 0.10,
     "reuse_threshold": 0.15, "max_staleness": 10, "max_steps": 500},
    {"name": "v2_max", "warmup_fraction": 0.0, "gradient_norm_threshold": 0.20,
     "reuse_threshold": 0.20, "max_staleness": 12, "max_steps": 500},
]

results = [{"name": "standard", "steps": n_std, "val_loss": std_loss, "delta": std_delta}]

for cfg in configs:
    print(f"\n--- {cfg['name']} ({args.time_budget}s) ---")
    m = build_model()
    t = GradientEventTrainerV2(m, lr=1e-3, **{k:v for k,v in cfg.items() if k != "name"})
    t0 = time.time()
    n = 0
    while time.time() - t0 < args.time_budget:
        for x, y in train_loader:
            if time.time() - t0 >= args.time_budget: break
            t.step(x, y); n += 1
    val_loss, val_acc = evaluate(m, val_loader)
    cum = t.get_cumulative_stats()
    delta = init_loss - val_loss
    ratio = delta / max(std_delta, 1e-8)
    print(f"  {n} steps, val_loss={val_loss:.4f} delta={delta:.4f} L0%={cum['L0_reuse_pct']*100:.0f} L2%={cum['L2_full_pct']*100:.0f} ratio={ratio:.2f}x")
    results.append({"name": cfg["name"], "steps": n, "val_loss": val_loss,
                    "delta": delta, "ratio": ratio,
                    "L0_pct": cum["L0_reuse_pct"], "L2_pct": cum["L2_full_pct"],
                    "speedup": cum["estimated_speedup"]})
    t.close(); del m, t

print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
print(f"{'Config':<25} {'Steps':>6} {'ValLoss':>8} {'Delta':>8} {'Ratio':>7} {'L0%':>6} {'L2%':>6}")
for r in results:
    ratio_str = f"{r.get('ratio', 1.0):.2f}x" if 'ratio' in r else "1.00x"
    l0 = f"{r.get('L0_pct', 0)*100:.0f}%" if 'L0_pct' in r else "0%"
    l2 = f"{r.get('L2_pct', 1)*100:.0f}%" if 'L2_pct' in r else "100%"
    print(f"{r['name']:<25} {r['steps']:>6} {r['val_loss']:>8.4f} {r['delta']:>8.4f} {ratio_str:>7} {l0:>6} {l2:>6}")

with open(os.path.join(os.path.dirname(__file__), "..", "benchmarks", "event_tuning.json"), "w") as f:
    json.dump({"init_loss": init_loss, "results": results}, f, indent=2)
