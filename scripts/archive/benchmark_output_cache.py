"""Benchmark: Standard vs Block-v3 vs Output-Caching (skip entire blocks)."""
import os, sys, time, json, math, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from torch.utils.data import DataLoader
from awf.core import AWFTransformer, num_params
from awf.event_training import StandardTrainer
from awf.block_v3_trainer import BlockV3Trainer
from awf.output_caching_trainer import OutputCachingTrainer
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
                skip = cum.get("skip_pct", cum.get("L0_reuse_pct", 0))
                print(f"  [{name}] {n} steps skip%={skip*100:.0f} ({time.time()-t0:.0f}s)")
    vl, va = evaluate(trainer.model, val_loader)
    cum = trainer.get_cumulative_stats()
    d = init_loss - vl; thru = n * BATCH / tb
    print(f"  [{name}] {n} steps val={vl:.4f} delta={d:.4f} thru={thru:.1f} skip%={cum.get('skip_pct', cum.get('L0_reuse_pct',0))*100:.0f}")
    trainer.close(); del trainer.model; del trainer
    return {"name": name, "steps": n, "val_loss": vl, "delta": d, "thru": thru,
            "skip_pct": cum.get("skip_pct", cum.get("L0_reuse_pct", 0)),
            "compute_pct": cum.get("compute_pct", cum.get("L2_full_pct", 1)),
            "est_speedup": cum.get("estimated_speedup", 1)}

results = []

# 1. Standard
print(f"\n=== STANDARD ({args.time_budget}s) ===")
m = build_model(); t = StandardTrainer(m, lr=1e-3)
results.append(run("standard", t, args.time_budget))

# 2. Block-v3 (weight caching)
print(f"\n=== BLOCK-V3 ({args.time_budget}s) ===")
m = build_model(); t = BlockV3Trainer(m, lr=1e-3, reuse_threshold=0.12, max_staleness=4, warmup_steps=30)
results.append(run("block_v3", t, args.time_budget))

# 3. Output-caching (skip entire blocks)
# Test multiple thresholds
for thresh in [0.05, 0.10, 0.20, 0.30]:
    print(f"\n=== OUTPUT-CACHE thresh={thresh} ({args.time_budget}s) ===")
    m = build_model()
    t = OutputCachingTrainer(m, lr=1e-3, reuse_threshold=thresh, max_staleness=3,
                              warmup_steps=20, min_full_blocks=1)
    r = run(f"out_cache_t{thresh}", t, args.time_budget)
    results.append(r)

# Summary
print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
std = results[0]
print(f"{'Trainer':<20} {'Steps':>6} {'ValLoss':>8} {'Delta':>8} {'Ratio':>7} {'Thru':>6} {'Combined':>9} {'Skip%':>6} {'EstSpeedup':>11}")
print("-" * 85)
for r in results:
    ratio = r["delta"] / max(std["delta"], 1e-8)
    thru_ratio = r["thru"] / max(std["thru"], 1e-8)
    combined = ratio * thru_ratio
    r["ratio"] = ratio; r["combined"] = combined
    print(f"{r['name']:<20} {r['steps']:>6} {r['val_loss']:>8.4f} {r['delta']:>8.4f} "
          f"{ratio:>6.2f}x {r['thru']:>5.1f} {combined:>8.2f}x {r['skip_pct']*100:>5.0f}% {r['est_speedup']:>10.1f}x")

best = max(results[1:], key=lambda r: r["combined"])
print(f"\n>>> BEST: {best['name']} with combined speedup {best['combined']:.2f}x")
print(f"    Learning ratio: {best['ratio']:.2f}x, Throughput ratio: {best['thru']/std['thru']:.2f}x")
print(f"    Skip rate: {best['skip_pct']*100:.0f}%, Est backward speedup: {best['est_speedup']:.1f}x")

with open(os.path.join(os.path.dirname(__file__), "..", "benchmarks", "output_cache_benchmark.json"), "w") as f:
    json.dump({"init_loss": init_loss, "results": results, "best": best["name"]}, f, indent=2)
