"""Benchmark: Standard vs Event v2 vs Event v3 (weight caching)."""
import os, sys, time, json, math, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from awf.core import AWFTransformer, num_params
from awf.event_training import StandardTrainer
from awf.event_training_v2 import GradientEventTrainerV2
from awf.event_training_v3 import GradientEventTrainerV3
from benchmark_event_v2 import ByteTokenizer, TextDataset, evaluate, build_model, BLOCK, BATCH, DATA_FILE

parser = argparse.ArgumentParser()
parser.add_argument("--time_budget", type=int, default=120)
parser.add_argument("--lr", type=float, default=1e-3)
args = parser.parse_args()

with open(DATA_FILE) as f: text = f.read()
text = text[:1_000_000]
tokenizer = ByteTokenizer()
n_train = int(0.95 * len(text))
train_ds = TextDataset(text[:n_train], tokenizer, BLOCK)
val_ds = TextDataset(text[n_train:], tokenizer, BLOCK)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH)

init_model = build_model()
init_loss, init_acc = evaluate(init_model, val_loader)
print(f"Init: val_loss={init_loss:.4f}")
del init_model

def run_trainer(trainer, name, time_budget):
    t0 = time.time()
    n = 0
    while time.time() - t0 < time_budget:
        for x, y in train_loader:
            if time.time() - t0 >= time_budget: break
            loss, _ = trainer.step(x, y)
            n += 1
            if n % 100 == 0:
                cum = trainer.get_cumulative_stats()
                print(f"  [{name}] step {n} loss={loss:.4f} L0%={cum['L0_reuse_pct']*100:.0f} ({time.time()-t0:.0f}s)")
    val_loss, val_acc = evaluate(trainer.model, val_loader)
    cum = trainer.get_cumulative_stats()
    delta = init_loss - val_loss
    ratio = delta / max(init_loss - 2.0, 0.001)  # normalized
    throughput = n * BATCH / time_budget
    print(f"  [{name}] {n} steps, val_loss={val_loss:.4f} delta={delta:.4f} L0%={cum['L0_reuse_pct']*100:.0f} throughput={throughput:.1f}")
    trainer.close()
    del trainer.model; del trainer
    return {"name": name, "steps": n, "val_loss": val_loss, "delta": delta, "throughput": throughput,
            "L0_pct": cum["L0_reuse_pct"], "L2_pct": cum["L2_full_pct"],
            "speedup": cum.get("estimated_speedup", 1), "cache_hits": cum.get("cache_hits", 0)}

results = []

# Standard
print(f"\n=== STANDARD ({args.time_budget}s) ===")
m = build_model()
t = StandardTrainer(m, lr=args.lr)
r = run_trainer(t, "standard", args.time_budget)
results.append(r)

# v2
print(f"\n=== EVENT v2 ({args.time_budget}s) ===")
m = build_model()
t = GradientEventTrainerV2(m, lr=args.lr, reuse_threshold=0.15, warmup_fraction=0.0,
                           gradient_norm_threshold=0.10, max_staleness=10, max_steps=500)
r = run_trainer(t, "evt_v2", args.time_budget)
results.append(r)

# v3 (weight caching)
print(f"\n=== EVENT v3 ({args.time_budget}s) ===")
m = build_model()
t = GradientEventTrainerV3(m, lr=args.lr, reuse_threshold=0.15, warmup_fraction=0.0,
                           gradient_norm_threshold=0.10, max_staleness=10, max_steps=500)
r = run_trainer(t, "evt_v3", args.time_budget)
results.append(r)

# Summary
print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
print(f"{'Trainer':<15} {'Steps':>6} {'ValLoss':>8} {'Delta':>8} {'Thru':>6} {'L0%':>5} {'L2%':>5} {'Cache':>7}")
for r in results:
    print(f"{r['name']:<15} {r['steps']:>6} {r['val_loss']:>8.4f} {r['delta']:>8.4f} "
          f"{r['throughput']:>5.1f} {r['L0_pct']*100:>4.0f}% {r['L2_pct']*100:>4.0f}% {r.get('cache_hits',0):>7}")

std_delta = results[0]["delta"]
for r in results[1:]:
    ratio = r["delta"] / max(std_delta, 1e-8)
    print(f"\n{r['name']} vs standard: ratio={ratio:.2f}x (val_loss {r['val_loss']:.4f} vs {results[0]['val_loss']:.4f})")

with open(os.path.join(os.path.dirname(__file__), "..", "benchmarks", "event_v3_benchmark.json"), "w") as f:
    json.dump({"init_loss": init_loss, "results": results}, f, indent=2)
