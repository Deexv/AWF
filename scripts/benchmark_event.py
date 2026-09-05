"""
Benchmark: Standard vs Event-Driven training on AWF LLM.

Trains two identical AWF models from the same checkpoint for the same wall-clock
time. Measures:
  - Final val loss / accuracy
  - Samples/sec (training throughput)
  - Estimated FLOPs saved by event-driven

If event-driven achieves LOWER val loss in the SAME time → it's working.
If event-driven achieves SIMILAR val loss in LESS time → speedup confirmed.

Usage:
    python scripts/benchmark_event.py --time_budget 300
    python scripts/benchmark_event.py --time_budget 60 --fast   # quick test
"""
import os, sys, time, json, math, argparse, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from awf import AWFTransformer, GradientEventTrainer, StandardTrainer, num_params
from awf.event_training import ActivationCollector

# ============================================================================
# Config
# ============================================================================
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
LOG_DIR = os.path.join(ROOT, "benchmarks")
DATA_FILE = os.path.join(ROOT, "data", "tinystories_train.txt")

BLOCK = 128
BATCH = 16


# ============================================================================
# Tokenizer & Dataset
# ============================================================================
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


# ============================================================================
# Eval
# ============================================================================
def evaluate(model, loader, max_batches=30):
    # Ensure all floating-point params are trainable for eval
    for p in model.parameters():
        if p.is_floating_point():
            p.requires_grad_(True)
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


# ============================================================================
# Main benchmark
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_budget", type=int, default=300, help="Seconds per trainer")
    parser.add_argument("--fast", action="store_true", help="Quick test (smaller subset)")
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    print("=" * 70)
    print("AWF Event-Driven Training Benchmark")
    print("=" * 70)
    print(f"Time budget per trainer: {args.time_budget}s")
    print(f"LR: {args.lr}")
    print()

    # Load data
    with open(DATA_FILE) as f:
        text = f.read()
    # Use small subset for fast benchmark
    n_chars = 500_000 if args.fast else 1_000_000
    text = text[:n_chars]
    tokenizer = ByteTokenizer()
    print(f"Corpus: {len(text):,} chars")

    n_train = int(0.95 * len(text))
    train_ds = TextDataset(text[:n_train], tokenizer, BLOCK)
    val_ds = TextDataset(text[n_train:], tokenizer, BLOCK)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH)
    print(f"Train: {len(train_ds):,} samples, Val: {len(val_ds):,}")

    # Build AWF model and load existing checkpoint
    def build_model():
        torch.manual_seed(0)
        m = AWFTransformer(vocab_size=256, d_model=256, n_layers=6, n_heads=8, block_size=BLOCK,
                          residual_rank=16, sparse_k=256,
                          gen_kwargs=dict(n_fourier=16, hidden=128, n_layers=3), gen_grid=16)
        # Load pre-trained checkpoint so both start from the same place
        ckpt = os.path.join(CKPT_DIR, "awf_10m.pt")
        if os.path.exists(ckpt):
            state = torch.load(ckpt, map_location="cpu")
            m.load_state_dict(state["model"], strict=False)
            try: m.activate_sparse_corrections()
            except: pass
        return m

    # Initial eval
    print("\nLoading checkpoint...")
    init_model = build_model()
    init_val_loss, init_val_acc = evaluate(init_model, val_loader)
    print(f"Initial val_loss={init_val_loss:.4f} val_acc={init_val_acc*100:.2f}% ppl={math.exp(min(init_val_loss,20)):.1f}")
    init_step = 0
    ckpt = os.path.join(CKPT_DIR, "awf_10m.pt")
    if os.path.exists(ckpt):
        init_step = torch.load(ckpt, map_location="cpu").get("step", 0)
    print(f"Starting from step {init_step}")
    del init_model

    # ============================================================================
    # 1. Standard training (baseline)
    # ============================================================================
    print("\n" + "=" * 70)
    print(f"STANDARD training for {args.time_budget}s")
    print("=" * 70)
    model_std = build_model()
    trainer_std = StandardTrainer(model_std, lr=args.lr)

    t0 = time.time()
    n_steps_std = 0
    train_losses_std = []
    while time.time() - t0 < args.time_budget:
        for x, y in train_loader:
            if time.time() - t0 >= args.time_budget: break
            loss, _ = trainer_std.step(x, y)
            train_losses_std.append(loss)
            n_steps_std += 1
            if n_steps_std % 50 == 0:
                elapsed = time.time() - t0
                print(f"  std step {n_steps_std} loss={loss:.4f} ({elapsed:.0f}s)")

    val_loss_std, val_acc_std = evaluate(model_std, val_loader)
    samples_per_sec_std = n_steps_std * BATCH / args.time_budget
    print(f"\nStandard: {n_steps_std} steps, val_loss={val_loss_std:.4f} val_acc={val_acc_std*100:.2f}% ppl={math.exp(min(val_loss_std,20)):.1f}")
    print(f"  Throughput: {samples_per_sec_std:.1f} samples/sec")
    del model_std, trainer_std

    # ============================================================================
    # 2. Event-driven training (same time budget)
    # ============================================================================
    print("\n" + "=" * 70)
    print(f"EVENT-DRIVEN training for {args.time_budget}s")
    print("=" * 70)
    model_evt = build_model()
    trainer_evt = GradientEventTrainer(model_evt, lr=args.lr,
                                        reuse_threshold=0.10,
                                        approx_threshold=0.25,
                                        history_size=16,
                                        decision_interval=4,
                                        min_full_layers=4,
                                        approx_scale=0.3)

    t0 = time.time()
    n_steps_evt = 0
    train_losses_evt = []
    while time.time() - t0 < args.time_budget:
        for x, y in train_loader:
            if time.time() - t0 >= args.time_budget: break
            loss, stats = trainer_evt.step(x, y)
            train_losses_evt.append(loss)
            n_steps_evt += 1
            if n_steps_evt % 50 == 0:
                elapsed = time.time() - t0
                cum = trainer_evt.get_cumulative_stats()
                print(f"  evt step {n_steps_evt} loss={loss:.4f} L0={stats['L0']} L1={stats['L1']} L2={stats['L2']} skip%={cum['L0_reuse_pct']*100:.0f} ({elapsed:.0f}s)")

    val_loss_evt, val_acc_evt = evaluate(model_evt, val_loader)
    samples_per_sec_evt = n_steps_evt * BATCH / args.time_budget
    cum_evt = trainer_evt.get_cumulative_stats()
    print(f"\nEvent-driven: {n_steps_evt} steps, val_loss={val_loss_evt:.4f} val_acc={val_acc_evt*100:.2f}% ppl={math.exp(min(val_loss_evt,20)):.1f}")
    print(f"  Throughput: {samples_per_sec_evt:.1f} samples/sec")
    print(f"  L0 reuse: {cum_evt['L0_reuse_pct']*100:.1f}%")
    print(f"  L1 approx: {cum_evt['L1_approx_pct']*100:.1f}%")
    print(f"  L2 full: {cum_evt['L2_full_pct']*100:.1f}%")
    print(f"  Generator skipped: {cum_evt['generator_skipped']} steps ({cum_evt['generator_skipped_pct']*100:.1f}%)")
    print(f"  Estimated speedup (if backward cost ∝ L2%): {cum_evt['estimated_speedup']:.2f}x")
    trainer_evt.close()
    del model_evt, trainer_evt

    # ============================================================================
    # 3. Comparison
    # ============================================================================
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<35} {'Standard':>12} {'Event-driven':>14}")
    print("-" * 65)
    print(f"{'Steps trained':<35} {n_steps_std:>12} {n_steps_evt:>14}")
    print(f"{'Samples/sec':<35} {samples_per_sec_std:>12.1f} {samples_per_sec_evt:>14.1f}")
    print(f"{'Val loss':<35} {val_loss_std:>12.4f} {val_loss_evt:>14.4f}")
    print(f"{'Val accuracy':<35} {val_acc_std*100:>11.2f}% {val_acc_evt*100:>13.2f}%")
    print(f"{'Perplexity':<35} {math.exp(min(val_loss_std,20)):>12.1f} {math.exp(min(val_loss_evt,20)):>14.1f}")
    print()
    val_loss_delta_evt = init_val_loss - val_loss_evt
    val_loss_delta_std = init_val_loss - val_loss_std
    print(f"Val loss reduction (vs init):")
    print(f"  Standard:      {val_loss_delta_std:.4f} ({val_loss_delta_std/init_val_loss*100:.1f}%)")
    print(f"  Event-driven:  {val_loss_delta_evt:.4f} ({val_loss_delta_evt/init_val_loss*100:.1f}%)")
    # Handle case where standard made loss worse
    if val_loss_delta_std > 0:
        ratio = val_loss_delta_evt / val_loss_delta_std
        print(f"  Event/Std ratio: {ratio:.2f}x (ratio > 1.0 = event-driven learned MORE)")
    else:
        # Standard diverged or stagnated — event-driven wins by being stable
        print(f"  Standard failed to reduce loss (delta={val_loss_delta_std:.4f}); event-driven reduced by {val_loss_delta_evt:.4f}")
        ratio = float('inf') if val_loss_delta_evt > 0 else 0
    print(f"  (ratio > 1.0 means event-driven learned MORE in same time = SUCCESS)")

    # Save results
    results = {
        "config": {"time_budget": args.time_budget, "lr": args.lr, "batch": BATCH,
                   "block": BLOCK, "n_train_chars": n_chars, "init_step": init_step},
        "init": {"val_loss": init_val_loss, "val_acc": init_val_acc},
        "standard": {"steps": n_steps_std, "samples_per_sec": samples_per_sec_std,
                     "val_loss": val_loss_std, "val_acc": val_acc_std,
                     "val_ppl": math.exp(min(val_loss_std, 20)),
                     "loss_delta": val_loss_delta_std},
        "event_driven": {"steps": n_steps_evt, "samples_per_sec": samples_per_sec_evt,
                         "val_loss": val_loss_evt, "val_acc": val_acc_evt,
                         "val_ppl": math.exp(min(val_loss_evt, 20)),
                         "loss_delta": val_loss_delta_evt,
                         "L0_reuse_pct": cum_evt["L0_reuse_pct"],
                         "L1_approx_pct": cum_evt["L1_approx_pct"],
                         "L2_full_pct": cum_evt["L2_full_pct"],
                         "generator_skipped": cum_evt["generator_skipped"],
                         "generator_skipped_pct": cum_evt["generator_skipped_pct"],
                         "estimated_speedup": cum_evt["estimated_speedup"]},
        "verdict": {
            "event_learned_more": val_loss_delta_evt > val_loss_delta_std,
            "event_learned_ratio": ratio if val_loss_delta_std > 0 else (float('inf') if val_loss_delta_evt > 0 else 0),
            "event_val_loss_lower": val_loss_evt < val_loss_std,
        }
    }
    out_path = os.path.join(LOG_DIR, "event_benchmark.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
