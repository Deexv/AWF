"""Final benchmark: Standard vs v2 vs v3 vs Gradient-Only. Find maximum speedup."""
import os, sys, time, json, math, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from torch.utils.data import DataLoader
from awf.event_training import StandardTrainer
from awf.event_training_v2 import GradientEventTrainerV2
from awf.event_training_v3 import GradientEventTrainerV3
from awf.grad_only_trainer import GradientOnlyTrainer
from benchmark_event_v2 import ByteTokenizer, TextDataset, evaluate, build_model, BLOCK, BATCH, DATA_FILE

parser = argparse.ArgumentParser()
parser.add_argument("--time_budget", type=int, default=120)
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
                print(f"  [{name}] {n} steps L0%={cum['L0_reuse_pct']*100:.0f} ({time.time()-t0:.0f}s)")
    vl, va = evaluate(trainer.model, val_loader)
    cum = trainer.get_cumulative_stats()
    d = init_loss - vl; thru = n * BATCH / tb
    print(f"  [{name}] {n} steps val={vl:.4f} delta={d:.4f} thru={thru:.1f}")
    trainer.close(); del trainer.model; del trainer
    return {"name": name, "steps": n, "val_loss": vl, "delta": d, "thru": thru,
            "L0_pct": cum["L0_reuse_pct"], "L2_pct": cum["L2_full_pct"]}

results = []

# 1. Standard
m = build_model(); t = StandardTrainer(m, lr=1e-3)
results.append(run("standard", t, args.time_budget))

# 2. v2 (best from earlier tuning)
m = build_model(); t = GradientEventTrainerV2(m, lr=1e-3, reuse_threshold=0.15,
    warmup_fraction=0.0, gradient_norm_threshold=0.10, max_staleness=10, max_steps=500)
results.append(run("v2", t, args.time_budget))

# 3. v3 (weight caching)
m = build_model(); t = GradientEventTrainerV3(m, lr=1e-3, reuse_threshold=0.15,
    warmup_fraction=0.0, gradient_norm_threshold=0.10, max_staleness=6, max_steps=500)
results.append(run("v3", t, args.time_budget))

# 4. Gradient-only (minimal overhead)
m = build_model(); t = GradientOnlyTrainer(m, lr=1e-3, skip_threshold=0.01,
    max_staleness=6, min_full=6, warmup_steps=50)
results.append(run("grad_only", t, args.time_budget))

# Summary
print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
print(f"{'Trainer':<15} {'Steps':>6} {'ValLoss':>8} {'Delta':>8} {'Ratio':>7} {'Thru':>6} {'L0%':>5} {'L2%':>5}")
std_delta = results[0]["delta"]
for r in results:
    ratio = r["delta"] / max(std_delta, 1e-8)
    r["ratio"] = ratio
    print(f"{r['name']:<15} {r['steps']:>6} {r['val_loss']:>8.4f} {r['delta']:>8.4f} {ratio:>6.2f}x {r['thru']:>5.1f} {r['L0_pct']*100:>4.0f}% {r['L2_pct']*100:>4.0f}%")

print(f"\nCombined speedup (ratio × throughput improvement):")
for r in results[1:]:
    combined = r["ratio"] * (r["thru"] / results[0]["thru"])
    print(f"  {r['name']:<15} ratio={r['ratio']:.2f}x thru_gain={r['thru']/results[0]['thru']:.2f}x combined={combined:.2f}x")

best = max(results[1:], key=lambda r: r["ratio"] * r["thru"] / results[0]["thru"])
print(f"\n>>> BEST: {best['name']} with combined speedup {best['ratio'] * best['thru']/results[0]['thru']:.2f}x")

with open(os.path.join(os.path.dirname(__file__), "..", "benchmarks", "event_max_benchmark.json"), "w") as f:
    json.dump({"init_loss": init_loss, "results": results, "best": best["name"]}, f, indent=2)
