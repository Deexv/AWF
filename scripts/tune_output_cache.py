"""Tune output caching to maximize skip rate while maintaining learning quality."""
import os, sys, time, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from torch.utils.data import DataLoader
from awf.event_training import StandardTrainer
from awf.output_caching_trainer import OutputCachingTrainer
from benchmark_event_v2 import ByteTokenizer, TextDataset, evaluate, build_model, BLOCK, BATCH, DATA_FILE

parser = argparse.ArgumentParser()
parser.add_argument("--time_budget", type=int, default=60)
args = parser.parse_args()

with open(DATA_FILE) as f: text = f.read()
text = text[:1_000_000]
tok = ByteTokenizer()
n_train = int(0.95 * len(text))
train_ds = TextDataset(text[:n_train], tok, BLOCK)
val_ds = TextDataset(text[n_train:], tok, BLOCK)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH)

init_model = build_model()
init_loss, _ = evaluate(init_model, val_loader)
print(f"Init: val_loss={init_loss:.4f}")
del init_model

# Standard baseline
m = build_model(); t = StandardTrainer(m, lr=1e-3)
t0 = time.time(); n_std = 0
while time.time() - t0 < args.time_budget:
    for x, y in train_loader:
        if time.time() - t0 >= args.time_budget: break
        t.step(x, y); n_std += 1
std_loss, _ = evaluate(m, val_loader)
std_delta = init_loss - std_loss
std_thru = n_std * BATCH / args.time_budget
print(f"Standard: {n_std} steps val={std_loss:.4f} delta={std_delta:.4f} thru={std_thru:.1f}")
t.close(); del m, t

# Test configs: vary threshold, max_staleness, min_full_blocks
configs = [
    # name, threshold, max_staleness, min_full, warmup
    ("agg_s2_m1", 0.30, 2, 1, 10),    # aggressive: skip more, staleness=2, min_full=1
    ("agg_s3_m1", 0.30, 3, 1, 10),    # aggressive: staleness=3
    ("v_agg_s2", 0.50, 2, 1, 10),     # very aggressive threshold
    ("v_agg_s4", 0.50, 4, 1, 10),     # very aggressive, staleness=4
    ("max_agg", 0.80, 5, 1, 5),       # maximum aggressiveness
    ("conservative", 0.05, 3, 2, 30), # conservative baseline
]

results = []
for name, thresh, ms, mf, warmup in configs:
    m = build_model()
    t = OutputCachingTrainer(m, lr=1e-3, reuse_threshold=thresh,
                               max_staleness=ms, warmup_steps=warmup,
                               min_full_blocks=mf)
    t0 = time.time(); n = 0
    while time.time() - t0 < args.time_budget:
        for x, y in train_loader:
            if time.time() - t0 >= args.time_budget: break
            t.step(x, y); n += 1
    vl, _ = evaluate(m, val_loader)
    cum = t.get_cumulative_stats()
    d = init_loss - vl; ratio = d / max(std_delta, 1e-8)
    thru = n * BATCH / args.time_budget
    combined = ratio * (thru / max(std_thru, 1e-8))
    print(f"{name}: {n} steps val={vl:.4f} d={d:.4f} skip%={cum['skip_pct']*100:.0f} ratio={ratio:.2f}x thru={thru:.1f} combined={combined:.2f}x")
    results.append({"name": name, "steps": n, "val_loss": vl, "delta": d,
                    "skip_pct": cum["skip_pct"], "ratio": ratio, "thru": thru, "combined": combined})
    t.close(); del m, t

print(f"\n{'='*70}\nFINAL RANKING\n{'='*70}")
print(f"{'Config':<20} {'Skip%':>6} {'Ratio':>7} {'Thru':>6} {'Combined':>9}")
for r in sorted(results, key=lambda r: r["combined"], reverse=True):
    print(f"{r['name']:<20} {r['skip_pct']*100:>5.0f}% {r['ratio']:>6.2f}x {r['thru']:>5.1f} {r['combined']:>8.2f}x")

best = max(results, key=lambda r: r["combined"])
print(f"\n>>> BEST: {best['name']} combined={best['combined']:.2f}x (skip={best['skip_pct']*100:.0f}% ratio={best['ratio']:.2f}x)")

with open(os.path.join(os.path.dirname(__file__), "..", "benchmarks", "output_cache_tuning.json"), "w") as f:
    json.dump({"init_loss": init_loss, "std_delta": std_delta, "results": results, "best": best}, f, indent=2)
