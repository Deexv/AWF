"""Benchmark: Standard vs Event-Driven v1 vs Event-Driven v2.

v2 improvements: scoutless, adaptive warmup, staleness counter, gradient norm signal.
"""
import os, sys, time, json, math, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from awf.core import AWFTransformer, num_params
from awf.event_training import GradientEventTrainer as EventTrainerV1, StandardTrainer
from awf.event_training_v2 import GradientEventTrainerV2 as EventTrainerV2

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
LOG_DIR = os.path.join(ROOT, "benchmarks")
DATA_FILE = os.path.join(ROOT, "data", "tinystories_train.txt")
BLOCK = 128
BATCH = 16


class ByteTokenizer:
    def __init__(self): self.vocab_size = 256
    def encode(self, t): return list(t.encode('utf-8'))
    def decode(self, ids): return bytes([i for i in ids if i < 256]).decode('utf-8', errors='ignore')


class TextDataset(Dataset):
    def __init__(self, text, tokenizer, block_size):
        self.data = tokenizer.encode(text)
        self.block_size = block_size
    def __len__(self): return max(0, len(self.data) - self.block_size - 1)
    def __getitem__(self, i):
        c = self.data[i:i + self.block_size + 1]
        return torch.tensor(c[:-1], dtype=torch.long), torch.tensor(c[1:], dtype=torch.long)


def evaluate(model, loader, max_batches=30):
    for p in model.parameters():
        if p.is_floating_point(): p.requires_grad_(True)
    model.eval()
    loss_sum, n, correct, total = 0.0, 0, 0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_batches: break
            logits = model(x)
            V = logits.size(-1)
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction="sum")
            loss_sum += loss.item(); n += y.numel()
            correct += (logits.argmax(-1) == y).sum().item(); total += y.numel()
    return loss_sum/max(n,1), correct/max(total,1)


def build_model():
    torch.manual_seed(42)
    return AWFTransformer(vocab_size=256, d_model=256, n_layers=6, n_heads=8, block_size=BLOCK,
                          residual_rank=16, sparse_k=256,
                          gen_kwargs=dict(n_fourier=16, hidden=128, n_layers=3), gen_grid=16)


def run_trainer(trainer, train_loader, val_loader, time_budget, name):
    """Run a trainer for time_budget seconds, return results."""
    t0 = time.time()
    n_steps = 0
    while time.time() - t0 < time_budget:
        for x, y in train_loader:
            if time.time() - t0 >= time_budget: break
            loss, stats = trainer.step(x, y)
            n_steps += 1
            if n_steps % 100 == 0:
                elapsed = time.time() - t0
                cum = trainer.get_cumulative_stats()
                print(f"  [{name}] step {n_steps} loss={loss:.4f} L0%={cum['L0_reuse_pct']*100:.0f} L2%={cum['L2_full_pct']*100:.0f} ({elapsed:.0f}s)")
    val_loss, val_acc = evaluate(trainer.model, val_loader)
    cum = trainer.get_cumulative_stats()
    throughput = n_steps * BATCH / time_budget
    trainer.close()
    del trainer.model
    del trainer
    return {
        "name": name, "steps": n_steps, "throughput": throughput,
        "val_loss": val_loss, "val_acc": val_acc,
        "val_ppl": math.exp(min(val_loss, 20)),
        "L0_reuse_pct": cum.get("L0_reuse_pct", 0),
        "L2_full_pct": cum.get("L2_full_pct", 1),
        "estimated_speedup": cum.get("estimated_speedup", 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_budget", type=int, default=180)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    print("=" * 70)
    print("AWF Event-Driven Training v2 Benchmark")
    print("=" * 70)

    with open(DATA_FILE) as f: text = f.read()
    n_chars = 1_000_000
    text = text[:n_chars]
    tokenizer = ByteTokenizer()
    n_train = int(0.95 * len(text))
    train_ds = TextDataset(text[:n_train], tokenizer, BLOCK)
    val_ds = TextDataset(text[n_train:], tokenizer, BLOCK)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH)
    print(f"Corpus: {n_chars:,} chars, Train: {len(train_ds):,}, Val: {len(val_ds):,}")

    # Initial eval
    init_model = build_model()
    init_val_loss, init_val_acc = evaluate(init_model, val_loader)
    print(f"Initial: val_loss={init_val_loss:.4f} val_acc={init_val_acc*100:.2f}% ppl={math.exp(min(init_val_loss,20)):.1f}")
    del init_model

    results = []

    # 1. Standard
    print(f"\n{'='*70}\nSTANDARD ({args.time_budget}s)\n{'='*70}")
    m = build_model()
    t = StandardTrainer(m, lr=args.lr)
    r = run_trainer(t, train_loader, val_loader, args.time_budget, "standard")
    r["loss_delta"] = init_val_loss - r["val_loss"]
    results.append(r)
    print(f"  → {r['steps']} steps, val_loss={r['val_loss']:.4f} delta={r['loss_delta']:.4f}")

    # 2. Event v1 (scout-based)
    print(f"\n{'='*70}\nEVENT v1 ({args.time_budget}s)\n{'='*70}")
    m = build_model()
    t = EventTrainerV1(m, lr=args.lr, reuse_threshold=0.10, approx_threshold=0.25,
                       decision_interval=4, min_full_layers=4)
    r = run_trainer(t, train_loader, val_loader, args.time_budget, "evt_v1")
    r["loss_delta"] = init_val_loss - r["val_loss"]
    results.append(r)
    print(f"  → {r['steps']} steps, val_loss={r['val_loss']:.4f} delta={r['loss_delta']:.4f}")

    # 3. Event v2 (scoutless + adaptive + staleness)
    print(f"\n{'='*70}\nEVENT v2 ({args.time_budget}s)\n{'='*70}")
    m = build_model()
    t = EventTrainerV2(m, lr=args.lr, reuse_threshold=0.10, approx_threshold=0.25,
                       min_full_layers=4, max_staleness=8, warmup_fraction=0.20,
                       max_steps=5000, gradient_norm_threshold=0.01)
    r = run_trainer(t, train_loader, val_loader, args.time_budget, "evt_v2")
    r["loss_delta"] = init_val_loss - r["val_loss"]
    results.append(r)
    print(f"  → {r['steps']} steps, val_loss={r['val_loss']:.4f} delta={r['loss_delta']:.4f}")

    # Comparison
    print(f"\n{'='*70}\nCOMPARISON\n{'='*70}")
    print(f"{'Trainer':<15} {'Steps':>7} {'Val Loss':>10} {'Delta':>8} {'L0%':>6} {'L2%':>6} {'Speedup':>8}")
    print("-" * 65)
    for r in results:
        print(f"{r['name']:<15} {r['steps']:>7} {r['val_loss']:>10.4f} {r['loss_delta']:>8.4f} "
              f"{r['L0_reuse_pct']*100:>5.1f}% {r['L2_full_pct']*100:>5.1f}% {r['estimated_speedup']:>7.2f}x")

    std_delta = results[0]["loss_delta"]
    v1_delta = results[1]["loss_delta"]
    v2_delta = results[2]["loss_delta"]
    print(f"\nLoss reduction vs init:")
    print(f"  Standard:    {std_delta:.4f} ({std_delta/init_val_loss*100:.1f}%)")
    print(f"  Event v1:    {v1_delta:.4f} ({v1_delta/init_val_loss*100:.1f}%)  ratio={v1_delta/max(std_delta,1e-8):.2f}x")
    print(f"  Event v2:    {v2_delta:.4f} ({v2_delta/init_val_loss*100:.1f}%)  ratio={v2_delta/max(std_delta,1e-8):.2f}x")

    with open(os.path.join(LOG_DIR, "event_v2_benchmark.json"), "w") as f:
        json.dump({"init": {"val_loss": init_val_loss}, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
