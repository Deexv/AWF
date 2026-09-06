"""Benchmark: Standard vs GII vs Block-level vs Combined — find maximum speedup."""
import os, sys, time, json, math, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from torch.utils.data import DataLoader
from awf.core import AWFTransformer, num_params
from awf.event_training import StandardTrainer
from awf.gii_trainer import GIITrainer
from awf.block_event_trainer import BlockLevelEventTrainer
from awf.combined_trainer import CombinedTrainer
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
                skip_rate = cum.get("batch_skip_rate", cum.get("L0_reuse_pct", 0))
                print(f"  [{name}] {n} steps skip%={skip_rate*100:.0f} ({time.time()-t0:.0f}s)")
    vl, va = evaluate(trainer.model, val_loader)
    cum = trainer.get_cumulative_stats()
    d = init_loss - vl; thru = n * BATCH / tb
    print(f"  [{name}] {n} steps val={vl:.4f} delta={d:.4f} thru={thru:.1f}")
    print(f"       stats: {cum}")
    trainer.close(); del trainer.model; del trainer
    return {"name": name, "steps": n, "val_loss": vl, "delta": d, "thru": thru, "stats": cum}

results = []

# 1. Standard
print(f"\n=== STANDARD ({args.time_budget}s) ===")
m = build_model(); t = StandardTrainer(m, lr=1e-3)
results.append(run("standard", t, args.time_budget))

# 2. GII only (batch-level filtering)
print(f"\n=== GII ({args.time_budget}s) ===")
m = build_model(); t = GIITrainer(m, lr=1e-3, gii_config={"skip_threshold": 0.92, "warmup_steps": 30})
results.append(run("gii", t, args.time_budget))

# 3. Block-level event-driven
print(f"\n=== BLOCK-EVENT ({args.time_budget}s) ===")
m = build_model(); t = BlockLevelEventTrainer(m, lr=1e-3, reuse_threshold=0.12, max_staleness=4, use_learned_gate=True)
results.append(run("block_evt", t, args.time_budget))

# 4. Combined: GII + Block-level
print(f"\n=== COMBINED ({args.time_budget}s) ===")
m = build_model(); t = CombinedTrainer(m, lr=1e-3, gii_skip_threshold=0.92, gii_warmup=30,
                                         block_reuse_threshold=0.12, block_max_staleness=4,
                                         use_learned_gate=True)
results.append(run("combined", t, args.time_budget))

# Summary
print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
std = results[0]
std_delta = std["delta"]
std_thru = std["thru"]
print(f"{'Trainer':<15} {'Steps':>6} {'ValLoss':>8} {'Delta':>8} {'Ratio':>7} {'Thru':>6} {'Combined':>9}")
print("-" * 65)
for r in results:
    ratio = r["delta"] / max(std_delta, 1e-8)
    thru_ratio = r["thru"] / max(std_thru, 1e-8)
    combined = ratio * thru_ratio
    r["ratio"] = ratio
    r["combined"] = combined
    print(f"{r['name']:<15} {r['steps']:>6} {r['val_loss']:>8.4f} {r['delta']:>8.4f} {ratio:>6.2f}x {r['thru']:>5.1f} {combined:>8.2f}x")

print(f"\nDetailed stats:")
for r in results:
    s = r.get("stats", {})
    print(f"  {r['name']}:")
    for k, v in s.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")

best = max(results[1:], key=lambda r: r["combined"])
print(f"\n>>> BEST: {best['name']} with combined speedup {best['combined']:.2f}x")
print(f"    Learning ratio: {best['ratio']:.2f}x")
print(f"    Throughput ratio: {best['thru']/std_thru:.2f}x")

with open(os.path.join(os.path.dirname(__file__), "..", "benchmarks", "combined_benchmark.json"), "w") as f:
    json.dump({"init_loss": init_loss, "results": results, "best": best["name"]}, f, indent=2)
