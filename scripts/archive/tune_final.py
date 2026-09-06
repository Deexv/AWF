"""Final tuning: find the maximum speedup by testing multiple v3 configs."""
import os, sys, time, json, math, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from torch.utils.data import DataLoader
from awf.event_training import StandardTrainer
from awf.event_training_v3 import GradientEventTrainerV3
from benchmark_event_v2 import ByteTokenizer, TextDataset, evaluate, build_model, BLOCK, BATCH, DATA_FILE

parser = argparse.ArgumentParser()
parser.add_argument("--time_budget", type=int, default=90)
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

def run(name, trainer, time_budget):
    t0 = time.time()
    n = 0
    while time.time() - t0 < time_budget:
        for x, y in train_loader:
            if time.time() - t0 >= time_budget: break
            trainer.step(x, y); n += 1
    vl, va = evaluate(trainer.model, val_loader)
    cum = trainer.get_cumulative_stats()
    delta = init_loss - vl
    thru = n * BATCH / time_budget
    print(f"  [{name}] {n} steps val={vl:.4f} delta={delta:.4f} L0%={cum['L0_reuse_pct']*100:.0f} thru={thru:.1f} cache={cum.get('cache_hits',0)}")
    trainer.close(); del trainer.model; del trainer
    return {"name": name, "steps": n, "val_loss": vl, "delta": delta, "thru": thru,
            "L0_pct": cum["L0_reuse_pct"], "L2_pct": cum["L2_full_pct"], "cache_hits": cum.get("cache_hits",0)}

# Standard
m = build_model(); t = StandardTrainer(m, lr=1e-3)
r_std = run("standard", t, args.time_budget)

# v3 configs — tune staleness and thresholds
configs = [
    # name, reuse_thresh, grad_norm_thresh, max_staleness, min_full, warmup
    ("v3_s4", 0.15, 0.10, 4, 6, 0.0),    # staleness=4, more full layers
    ("v3_s6", 0.15, 0.10, 6, 4, 0.0),    # staleness=6
    ("v3_s3", 0.12, 0.05, 3, 8, 0.0),    # staleness=3, very conservative
    ("v3_s4_loose", 0.20, 0.15, 4, 4, 0.0),  # loose thresholds
]

results = [r_std]
for name, rt, gnt, ms, mf, wf in configs:
    m = build_model()
    t = GradientEventTrainerV3(m, lr=1e-3, reuse_threshold=rt,
                               gradient_norm_threshold=gnt, max_staleness=ms,
                               min_full_layers=mf, warmup_fraction=wf, max_steps=500)
    r = run(name, t, args.time_budget)
    r["ratio"] = r["delta"] / max(r_std["delta"], 1e-8)
    results.append(r)

print(f"\n{'='*70}\nFINAL\n{'='*70}")
print(f"{'Config':<15} {'Steps':>6} {'ValLoss':>8} {'Delta':>8} {'Ratio':>7} {'Thru':>6} {'L0%':>5}")
for r in results:
    ratio = r.get("ratio", 1.0)
    print(f"{r['name']:<15} {r['steps']:>6} {r['val_loss']:>8.4f} {r['delta']:>8.4f} {ratio:>6.2f}x {r['thru']:>5.1f} {r['L0_pct']*100:>4.0f}%")

best = max(results[1:], key=lambda r: r.get("ratio", 0))
print(f"\nBest: {best['name']} with ratio={best.get('ratio',0):.2f}x")
print(f"  Throughput improvement: {best['thru']/r_std['thru']:.2f}x")
print(f"  Combined speedup: {best.get('ratio',0) * best['thru']/r_std['thru']:.2f}x")

with open(os.path.join(os.path.dirname(__file__), "..", "benchmarks", "event_final_tuning.json"), "w") as f:
    json.dump({"init_loss": init_loss, "results": results}, f, indent=2)
