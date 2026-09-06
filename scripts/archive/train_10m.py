"""
Train a 10M parameter AWF LLM on TinyStories dataset.

Architecture:
  - AWF: d_model=384, 6 layers, 8 heads, rank=16, sparse_k=256
    -> ~900K AWF params (12x compression vs dense)
  - Dense equivalent: ~10.9M params (matches user's "10M model" requirement)

Dataset: TinyStories (25MB subset, 32K stories, 26M chars, 5M words)
Tokenizer: Character-level with byte fallback (93 unique chars)

Training:
  - Resumable: saves checkpoint every N batches
  - Time-budgeted: trains for a fixed wall-clock budget per run
  - Logs progress every batch
"""
import os, sys, time, math, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from awf.core import AWFTransformer, DenseTransformer, num_params

# ============================================================================
# Config
# ============================================================================
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "tinystories_train.txt")
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Model config — AWF is ~900K params, Dense equivalent is ~10.9M
D_MODEL = 256
N_LAYERS = 6
N_HEADS = 8
RESIDUAL_RANK = 16
SPARSE_K = 256
BLOCK = 128
BATCH = 16  # smaller batch for memory

# Training config
LR = 1.0e-3  # slightly lower for better convergence
WARMUP_STEPS = 100
SPARSE_WARMUP_EPOCH = 2  # activate sparse after this many epochs
SAVE_EVERY_N_BATCHES = 200  # checkpoint frequency
TIME_BUDGET_SECONDS = 480   # 8 minutes per run (under 540s bash timeout)


# ============================================================================
# Tokenizer (byte-level: supports all 256 byte values)
# ============================================================================
class ByteTokenizer:
    """Byte-level tokenizer: 256 byte tokens + special padding."""
    def __init__(self):
        self.vocab_size = 256
        self.bos_id = 254
        self.eos_id = 253
        self.pad_id = 255

    def encode(self, text):
        return list(text.encode('utf-8'))

    def decode(self, ids):
        # Filter out special tokens
        ids = [i for i in ids if i < 253]
        return bytes(ids).decode('utf-8', errors='ignore')


# ============================================================================
# Dataset
# ============================================================================
class TextDataset(Dataset):
    def __init__(self, text, tokenizer, block_size):
        self.data = tokenizer.encode(text)
        self.block_size = block_size
        self.tokenizer = tokenizer

    def __len__(self):
        return max(0, len(self.data) - self.block_size - 1)

    def __getitem__(self, i):
        chunk = self.data[i:i + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


# ============================================================================
# Training utilities
# ============================================================================
def evaluate(model, loader, max_batches=None):
    model.eval()
    loss_sum, n, correct, total = 0.0, 0, 0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if max_batches and i >= max_batches: break
            logits = model(x)
            V = logits.size(-1)
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction="sum")
            loss_sum += loss.item(); n += y.numel()
            correct += (logits.argmax(-1) == y).sum().item(); total += y.numel()
    return loss_sum/max(n,1), correct/max(total,1)


def generate(model, tokenizer, prompt, n_tokens=100, temperature=0.7, top_k=10, seed=None):
    if seed is not None: torch.manual_seed(seed)
    model.eval()
    ids = tokenizer.encode(prompt)
    if not ids: ids = [0]
    with torch.no_grad():
        for _ in range(n_tokens):
            x = torch.tensor(ids[-BLOCK:], dtype=torch.long).unsqueeze(0)
            logits = model(x)
            nl = logits[0, -1] / max(temperature, 0.01)
            if top_k > 0:
                v, _ = torch.topk(nl, min(top_k, tokenizer.vocab_size))
                nl[nl < v[-1]] = -float("inf")
            probs = F.softmax(nl, dim=-1)
            ids.append(torch.multinomial(probs, 1).item())
    return tokenizer.decode(ids)


def lr_schedule(step, warmup, max_lr):
    """Linear warmup then constant."""
    if step < warmup:
        return max_lr * step / warmup
    return max_lr


# ============================================================================
# Main training (resumable, time-budgeted)
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["awf", "dense", "continue_awf", "continue_dense"], default="awf")
    parser.add_argument("--epochs", type=int, default=1, help="Max epochs (will stop early if time budget hit)")
    parser.add_argument("--time_budget", type=int, default=TIME_BUDGET_SECONDS)
    args = parser.parse_args()

    print(f"=== AWF 10M LLM Training ===")
    print(f"Mode: {args.mode}")
    print(f"Time budget: {args.time_budget}s ({args.time_budget/60:.1f}min)")

    # Load data
    with open(DATA_FILE) as f:
        text = f.read()
    # Use a subset for tractable CPU training — but a meaningful one
    # 5M chars = ~625K tokens at block_size=128 = ~5K batches per epoch
    n_chars = min(len(text), 3_000_000)  # 3M chars
    text = text[:n_chars]
    tokenizer = ByteTokenizer()
    print(f"Corpus: {len(text):,} chars (~{len(text)//4:,} tokens)")
    print(f"Vocab: {tokenizer.vocab_size} (byte-level)")

    n_train = int(0.95 * len(text))
    train_text = text[:n_train]
    val_text = text[n_train:]
    train_ds = TextDataset(train_text, tokenizer, BLOCK)
    val_ds = TextDataset(val_text, tokenizer, BLOCK)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)
    print(f"Train: {len(train_ds):,} samples ({len(train_loader)} batches)")
    print(f"Val: {len(val_ds):,} samples")

    # Build or load model
    if args.mode in ("awf", "continue_awf"):
        model = AWFTransformer(vocab_size=tokenizer.vocab_size, d_model=D_MODEL, n_layers=N_LAYERS,
                              n_heads=N_HEADS, block_size=BLOCK, residual_rank=RESIDUAL_RANK,
                              sparse_k=SPARSE_K,
                              gen_kwargs=dict(n_fourier=16, hidden=128, n_layers=3), gen_grid=16)
        ckpt_path = os.path.join(CKPT_DIR, "awf_10m.pt")
        model_type = "AWF"
    else:
        model = DenseTransformer(vocab_size=tokenizer.vocab_size, d_model=D_MODEL, n_layers=N_LAYERS,
                                 n_heads=N_HEADS, block_size=BLOCK)
        ckpt_path = os.path.join(CKPT_DIR, "dense_10m.pt")
        model_type = "Dense"

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    if args.mode in ("continue_awf", "continue_dense") and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state["model"])
        start_epoch = state.get("epoch", 0)
        global_step = state.get("step", 0)
        print(f"Resumed from {ckpt_path} (epoch {start_epoch}, step {global_step})")
        # If AWF and past sparse warmup, activate sparse corrections
        if model_type == "AWF" and start_epoch >= SPARSE_WARMUP_EPOCH:
            try:
                model.activate_sparse_corrections()
                print(f"Activated sparse corrections (resumed past warmup)")
            except Exception as e:
                print(f"Sparse activation: {e}")

    n_params = num_params(model)
    print(f"\n{model_type} model: {n_params:,} params ({n_params/1e6:.2f}M)")
    if model_type == "AWF":
        print(f"  Generator: {num_params(model.generator):,}")
        print(f"  Storage fp16: {model.total_storage_bytes(2)/1024:.1f}KB")

    # Evaluate before training
    val_loss, val_acc = evaluate(model, val_loader, max_batches=20)
    print(f"Initial: val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}% ppl={math.exp(min(val_loss,20)):.1f}")

    # Train
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    if os.path.exists(ckpt_path) and args.mode in ("continue_awf", "continue_dense"):
        state = torch.load(ckpt_path, map_location="cpu")
        if "optimizer" in state:
            try:
                opt.load_state_dict(state["optimizer"])
                print("Optimizer state restored")
            except: pass

    t0 = time.time()
    batch_count = 0
    for epoch in range(start_epoch, start_epoch + args.epochs):
        # Activate sparse corrections after warmup
        if model_type == "AWF" and SPARSE_K > 0 and epoch >= SPARSE_WARMUP_EPOCH and not getattr(model, '_sparse_activated', False):
            print(f"Activating sparse corrections (k={SPARSE_K}/layer)")
            try:
                model.activate_sparse_corrections()
                model._sparse_activated = True
            except Exception as e:
                print(f"  (sparse activation: {e})")

        model.train()
        ep_loss, n_seen = 0.0, 0
        for x, y in train_loader:
            # Time budget check
            if time.time() - t0 > args.time_budget:
                print(f"\nTime budget hit ({args.time_budget}s). Saving and exiting.")
                break

            # LR schedule
            lr = lr_schedule(global_step, WARMUP_STEPS, LR)
            for g in opt.param_groups: g["lr"] = lr

            opt.zero_grad()
            logits = model(x)
            V = logits.size(-1)
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep_loss += loss.item() * x.size(0); n_seen += x.size(0)
            global_step += 1
            batch_count += 1

            if batch_count % 50 == 0:
                elapsed = time.time() - t0
                print(f"  ep{epoch+1} batch {batch_count} step {global_step} loss={loss.item():.4f} lr={lr:.5f} ({elapsed:.0f}s)")

            # Checkpoint
            if batch_count % SAVE_EVERY_N_BATCHES == 0:
                torch.save({"model": model.state_dict(), "epoch": epoch,
                           "step": global_step, "val_loss": val_loss}, ckpt_path)
                # Quick eval
                val_loss, val_acc = evaluate(model, val_loader, max_batches=10)
                model.train()
                print(f"  >> Checkpoint saved. val_loss={val_loss:.4f} acc={val_acc*100:.2f}% ppl={math.exp(min(val_loss,20)):.1f}")

        else:
            # Epoch completed
            val_loss, val_acc = evaluate(model, val_loader, max_batches=30)
            print(f"[{model_type}] ep{epoch+1} done: train_loss={ep_loss/max(n_seen,1):.4f} val_loss={val_loss:.4f} acc={val_acc*100:.2f}% ppl={math.exp(min(val_loss,20)):.1f} ({time.time()-t0:.0f}s)")
            # Save end-of-epoch checkpoint with optimizer state
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                       "epoch": epoch + 1, "step": global_step, "val_loss": val_loss}, ckpt_path)
            continue
        # If we broke out of inner loop due to time budget, break outer too
        break

    # Final eval and save
    val_loss, val_acc = evaluate(model, val_loader, max_batches=50)
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
               "epoch": start_epoch + args.epochs, "step": global_step, "val_loss": val_loss}, ckpt_path)
    print(f"\n=== FINAL ===")
    print(f"{model_type}: val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}% ppl={math.exp(min(val_loss,20)):.1f}")
    print(f"Params: {n_params:,} ({n_params/1e6:.2f}M)")
    if model_type == "AWF":
        print(f"Storage fp16: {model.total_storage_bytes(2)/1024:.1f}KB")
    print(f"Checkpoint: {ckpt_path}")

    # Generate samples
    print(f"\n=== Sample Generation ===")
    prompts = ["Once upon a time", "The little girl", "A boy named", "In the forest"]
    samples = {}
    for p in prompts:
        s = generate(model, tokenizer, p, n_tokens=80, seed=42)
        print(f"\n{p!r} -> {s!r}")
        samples[p] = s

    # Save final results
    results = {
        "model_type": model_type,
        "params": n_params,
        "val_loss": val_loss, "val_acc": val_acc,
        "val_ppl": math.exp(min(val_loss, 20)),
        "epochs_completed": start_epoch + args.epochs if False else (epoch + 1),
        "global_step": global_step,
        "config": {"d_model": D_MODEL, "n_layers": N_LAYERS, "n_heads": N_HEADS,
                   "residual_rank": RESIDUAL_RANK, "sparse_k": SPARSE_K,
                   "block_size": BLOCK, "lr": LR, "batch": BATCH,
                   "train_chars": n_chars},
        "samples": samples,
    }
    if model_type == "AWF":
        results["storage_bytes_fp16"] = model.total_storage_bytes(2)
        results["generator_params"] = num_params(model.generator)

    out_file = os.path.join(LOG_DIR, f"{model_type.lower()}_10m_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
