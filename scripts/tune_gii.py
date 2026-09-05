"""Tune GII skip threshold to find the sweet spot where learning ratio >= 1.0."""
import os, sys, time, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from torch.utils.data import DataLoader
from awf.event_training import StandardTrainer
from awf.gii_trainer import GIITrainer
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
t0 = time.time(); n = 0
while time.time() - t0 < args.time_budget:
    for x, y in train_loader:
        if time.time() - t0 >= args.time_budget: break
        t.step(x, y); n += 1
std_loss, _ = evaluate(m, val_loader)
std_delta = init_loss - std_loss
print(f"Standard: {n} steps val={std_loss:.4f} delta={std_delta:.4f}")
t.close(); del m, t

# Test multiple GII thresholds
thresholds = [0.99, 0.98, 0.97, 0.96, 0.95, 0.90]
results = []

for thresh in thresholds:
    m = build_model()
    t = GIITrainer(m, lr=1e-3, gii_config={"skip_threshold": thresh, "warmup_steps": 20, "history_size": 32})
    t0 = time.time(); n = 0
    while time.time() - t0 < args.time_budget:
        for x, y in train_loader:
            if time.time() - t0 >= args.time_budget: break
            t.step(x, y); n += 1
    vl, _ = evaluate(m, val_loader)
    cum = t.get_cumulative_stats()
    d = init_loss - vl; ratio = d / max(std_delta, 1e-8)
    thru = n * BATCH / args.time_budget
    print(f"GII thresh={thresh}: {n} steps val={vl:.4f} delta={d:.4f} skip%={cum['skip_rate']*100:.0f} ratio={ratio:.2f}x thru={thru:.1f}")
    results.append({"threshold": thresh, "steps": n, "val_loss": vl, "delta": d,
                    "skip_rate": cum["skip_rate"], "ratio": ratio, "thru": thru})
    t.close(); del m, t

# Find best: highest combined (ratio × thru / std_thru)
std_thru = n * BATCH / args.time_budget  # last n is from standard
print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
print(f"{'Threshold':>10} {'Steps':>6} {'ValLoss':>8} {'Delta':>8} {'Skip%':>6} {'Ratio':>7} {'Thru':>6} {'Combined':>9}")
for r in results:
    combined = r["ratio"] * (r["thru"] / max(std_thru, 1e-8))
    print(f"{r['threshold']:>10.2f} {r['steps']:>6} {r['val_loss']:>8.4f} {r['delta']:>8.4f} "
          f"{r['skip_rate']*100:>5.0f}% {r['ratio']:>6.2f}x {r['thru']:>5.1f} {combined:>8.2f}x")

best = max(results, key=lambda r: r["ratio"] * (r["thru"] / max(std_thru, 1e-8)))
print(f"\n>>> BEST: threshold={best['threshold']} ratio={best['ratio']:.2f}x thru={best['thru']:.1f} "
      f"combined={best['ratio'] * best['thru']/max(std_thru,1e-8):.2f}x")

with open(os.path.join(os.path.dirname(__file__), "..", "benchmarks", "gii_tuning.json"), "w") as f:
    json.dump({"init_loss": init_loss, "std_delta": std_delta, "std_thru": std_thru,
               "results": results, "best": best}, f, indent=2)
