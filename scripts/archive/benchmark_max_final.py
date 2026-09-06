"""Final maximum speedup benchmark: Standard vs v3(layer) vs Block-v3 vs GII + Block-v3."""
import os, sys, time, json, math, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from torch.utils.data import DataLoader
from awf.core import AWFTransformer, num_params
from awf.event_training import StandardTrainer
from awf.event_training_v3 import GradientEventTrainerV3
from awf.block_v3_trainer import BlockV3Trainer
from awf.gii_trainer import GIITrainer
from benchmark_event_v2 import ByteTokenizer, TextDataset, evaluate, build_model, BLOCK, BATCH, DATA_FILE

parser = argparse.ArgumentParser()
parser.add_argument("--time_budget", type=int, default=100)
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

def run(name, trainer, tb):
    t0 = time.time(); n = 0
    while time.time() - t0 < tb:
        for x, y in train_loader:
            if time.time() - t0 >= tb: break
            trainer.step(x, y); n += 1
            if n % 100 == 0:
                cum = trainer.get_cumulative_stats()
                l0 = cum.get("L0_reuse_pct", cum.get("skip_rate", 0))
                print(f"  [{name}] {n} steps L0%={l0*100:.0f} ({time.time()-t0:.0f}s)")
    vl, va = evaluate(trainer.model, val_loader)
    cum = trainer.get_cumulative_stats()
    d = init_loss - vl; thru = n * BATCH / tb
    print(f"  [{name}] {n} steps val={vl:.4f} delta={d:.4f} thru={thru:.1f}")
    trainer.close(); del trainer.model; del trainer
    return {"name": name, "steps": n, "val_loss": vl, "delta": d, "thru": thru,
            "L0_pct": cum.get("L0_reuse_pct", cum.get("skip_rate", 0)),
            "L2_pct": cum.get("L2_full_pct", 1 - cum.get("skip_rate", 0)),
            "cache_hits": cum.get("cache_hits", 0)}

results = []

# 1. Standard
print(f"\n=== STANDARD ({args.time_budget}s) ===")
m = build_model(); t = StandardTrainer(m, lr=1e-3)
results.append(run("standard", t, args.time_budget))

# 2. v3 (layer-level weight caching)
print(f"\n=== v3-LAYER ({args.time_budget}s) ===")
m = build_model(); t = GradientEventTrainerV3(m, lr=1e-3, reuse_threshold=0.15,
    warmup_fraction=0.0, gradient_norm_threshold=0.10, max_staleness=6, max_steps=500)
results.append(run("v3_layer", t, args.time_budget))

# 3. Block-v3 (block-level weight caching)
print(f"\n=== BLOCK-v3 ({args.time_budget}s) ===")
m = build_model(); t = BlockV3Trainer(m, lr=1e-3, reuse_threshold=0.12,
    max_staleness=4, warmup_steps=30, min_full_blocks=1)
results.append(run("block_v3", t, args.time_budget))

# Summary
print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
std = results[0]
print(f"{'Trainer':<15} {'Steps':>6} {'ValLoss':>8} {'Delta':>8} {'Ratio':>7} {'Thru':>6} {'Combined':>9} {'L0%':>5} {'Cache':>7}")
print("-" * 75)
for r in results:
    ratio = r["delta"] / max(std["delta"], 1e-8)
    thru_ratio = r["thru"] / max(std["thru"], 1e-8)
    combined = ratio * thru_ratio
    r["ratio"] = ratio; r["combined"] = combined
    print(f"{r['name']:<15} {r['steps']:>6} {r['val_loss']:>8.4f} {r['delta']:>8.4f} "
          f"{ratio:>6.2f}x {r['thru']:>5.1f} {combined:>8.2f}x {r['L0_pct']*100:>4.0f}% {r['cache_hits']:>7}")

best = max(results[1:], key=lambda r: r["combined"])
print(f"\n>>> BEST: {best['name']} with combined speedup {best['combined']:.2f}x")
print(f"    Learning ratio: {best['ratio']:.2f}x")
print(f"    Throughput ratio: {best['thru']/std['thru']:.2f}x")

with open(os.path.join(os.path.dirname(__file__), "..", "benchmarks", "max_speedup_benchmark.json"), "w") as f:
    json.dump({"init_loss": init_loss, "results": results, "best": best["name"]}, f, indent=2)
