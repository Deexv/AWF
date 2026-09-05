"""
Unified AWF LLM Training — supports GPU, resume, multiple datasets, BPE tokenizer,
and event-driven gradient training.

Features:
  - Auto-detects CUDA GPU (falls back to CPU). Use on Google Colab with GPU runtime.
  - Resumes from checkpoint (--resume) — continues exactly where you left off
  - Trains on multiple datasets (--datasets file1.txt file2.txt ...)
  - Optional BPE tokenizer (--tokenizer bpe) for better text quality
  - Time-budgeted: saves every N batches, stops at time budget
  - **Event-driven training** (--event_driven): skips backward for low-novelty layers
    Achieves 1.5-2x more learning per unit time (proven in benchmarks/event_benchmark.json)
  - Logs to file for tracking progress across runs

Usage:
  # Standard training (CPU, resumes from checkpoint)
  python scripts/train.py --resume --time_budget 510

  # Event-driven training (1.5-2x more efficient, recommended!)
  python scripts/train.py --resume --time_budget 510 --event_driven

  # GPU training (Google Colab) — train on multiple datasets
  python scripts/train.py --datasets data/tinystories.txt data/wiki.txt --tokenizer bpe --epochs 5 --event_driven
"""
import os, sys, time, math, json, argparse, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from awf.core import AWFTransformer, DenseTransformer, num_params
from awf.event_training import GradientEventTrainer, StandardTrainer

# ============================================================================
# Config
# ============================================================================
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
LOG_DIR = os.path.join(ROOT, "benchmarks")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ============================================================================
# Tokenizers
# ============================================================================
class ByteTokenizer:
    """Byte-level tokenizer: 256 byte values. Universal — works with any text."""
    def __init__(self):
        self.vocab_size = 256

    def encode(self, text):
        return list(text.encode('utf-8'))

    def decode(self, ids):
        ids = [i for i in ids if i < 256]
        return bytes(ids).decode('utf-8', errors='ignore')


class BPETokenizer:
    """Simple BPE tokenizer trained on the corpus.

    More efficient than byte-level for English text (1 token ≈ 4 chars vs 1 char).
    Better text quality because the model learns word-level patterns.
    """
    def __init__(self, vocab_size=1024):
        self.vocab_size = vocab_size
        self.merges = []  # list of (token_a, token_b) → new_token
        self.vocab = []   # list of bytes for each token

    def _get_pair_counts(self, ids):
        counts = {}
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i+1])
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def train(self, text, target_vocab_size=None):
        """Train BPE merges on text."""
        target = target_vocab_size or self.vocab_size
        # Start with 256 byte tokens
        ids = list(text.encode('utf-8'))
        self.vocab = [bytes([i]) for i in range(256)]
        print(f"Training BPE: starting with {len(self.vocab)} tokens, target {target}")
        while len(self.vocab) < target:
            counts = self._get_pair_counts(ids)
            if not counts: break
            best_pair = max(counts, key=counts.get)
            new_id = len(self.vocab)
            # Merge in ids
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i+1]) == best_pair:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids
            self.merges.append(best_pair)
            self.vocab.append(self.vocab[best_pair[0]] + self.vocab[best_pair[1]])
            if len(self.vocab) % 100 == 0:
                print(f"  {len(self.vocab)} tokens")
        self.vocab_size = len(self.vocab)
        print(f"BPE trained: {self.vocab_size} tokens")

    def encode(self, text):
        ids = list(text.encode('utf-8'))
        # Apply merges in order
        for i, (a, b) in enumerate(self.merges):
            new_id = 256 + i
            new_ids = []
            j = 0
            while j < len(ids):
                if j < len(ids) - 1 and (ids[j], ids[j+1]) == (a, b):
                    new_ids.append(new_id)
                    j += 2
                else:
                    new_ids.append(ids[j])
                    j += 1
            ids = new_ids
        return ids

    def decode(self, ids):
        out = b""
        for i in ids:
            if i < len(self.vocab):
                out += self.vocab[i]
        return out.decode('utf-8', errors='ignore')

    def save(self, path):
        d = {"vocab_size": self.vocab_size,
             "merges": [[a, b] for a, b in self.merges],
             "vocab": [list(v) for v in self.vocab]}
        with open(path, "w") as f:
            json.dump(d, f)

    def load(self, path):
        with open(path) as f:
            d = json.load(f)
        self.vocab_size = d["vocab_size"]
        self.merges = [tuple(m) for m in d["merges"]]
        self.vocab = [bytes(v) for v in d["vocab"]]


# ============================================================================
# Dataset
# ============================================================================
class TextDataset(Dataset):
    def __init__(self, text, tokenizer, block_size):
        self.data = tokenizer.encode(text)
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.data) - self.block_size - 1)

    def __getitem__(self, i):
        chunk = self.data[i:i + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


# ============================================================================
# Utilities
# ============================================================================
def load_datasets(paths, max_chars_per_dataset=None, seed=42):
    """Load and concatenate multiple text datasets."""
    texts = []
    for p in paths:
        if not os.path.exists(p):
            print(f"WARNING: dataset {p} not found, skipping")
            continue
        with open(p) as f:
            t = f.read()
        if max_chars_per_dataset and len(t) > max_chars_per_dataset:
            random.seed(seed)
            start = random.randint(0, len(t) - max_chars_per_dataset)
            t = t[start:start + max_chars_per_dataset]
        texts.append(t)
        print(f"  {p}: {len(t):,} chars")
    return "\n\n".join(texts)


def evaluate(model, loader, device, max_batches=50):
    model.eval()
    loss_sum, n, correct, total = 0.0, 0, 0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_batches: break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            V = logits.size(-1)
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction="sum")
            loss_sum += loss.item(); n += y.numel()
            correct += (logits.argmax(-1) == y).sum().item(); total += y.numel()
    return loss_sum/max(n,1), correct/max(total,1)


def generate(model, tokenizer, device, prompt, n_tokens=150, temperature=0.8, top_k=20, top_p=0.9, seed=None):
    if seed is not None: torch.manual_seed(seed)
    model.eval()
    ids = tokenizer.encode(prompt)
    if not ids: ids = [0]
    with torch.no_grad():
        for _ in range(n_tokens):
            x = torch.tensor([ids[-model.block_size:] if hasattr(model, 'block_size') else ids[-128:]],
                              dtype=torch.long, device=device)
            logits = model(x)
            nl = logits[0, -1] / max(temperature, 0.01)
            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(nl, min(top_k, nl.size(-1)))
                nl[nl < v[-1]] = -float("inf")
            # Nucleus (top-p) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(nl, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cum_probs > top_p
                sorted_mask[1:] = sorted_mask[:-1].clone()
                sorted_mask[0] = False
                indices_to_remove = sorted_mask.scatter(-1, sorted_indices, sorted_mask)
                nl[indices_to_remove] = -float("inf")
            probs = F.softmax(nl, dim=-1)
            ids.append(torch.multinomial(probs, 1).item())
    return tokenizer.decode(ids)


def lr_schedule(step, warmup, max_lr, decay_steps=None):
    """Linear warmup then cosine decay (or constant if no decay_steps)."""
    if step < warmup:
        return max_lr * step / max(warmup, 1)
    if decay_steps is None:
        return max_lr
    if step > decay_steps:
        return max_lr * 0.1
    # Cosine decay
    progress = (step - warmup) / max(decay_steps - warmup, 1)
    return max_lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1)))


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="AWF LLM Training (GPU-resumable)")
    # Model config
    parser.add_argument("--model", choices=["awf", "dense"], default="awf")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--residual_rank", type=int, default=16)
    parser.add_argument("--sparse_k", type=int, default=256)
    parser.add_argument("--block_size", type=int, default=128)
    # Training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--time_budget", type=int, default=510, help="Max seconds per run")
    parser.add_argument("--save_every", type=int, default=200, help="Save checkpoint every N batches")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--sparse_warmup_epoch", type=int, default=2)
    # Data
    parser.add_argument("--datasets", nargs="+", default=["data/tinystories_train.txt"],
                        help="One or more text files to train on")
    parser.add_argument("--max_chars", type=int, default=3000000,
                        help="Max chars per dataset (for memory control)")
    parser.add_argument("--tokenizer", choices=["byte", "bpe"], default="byte")
    parser.add_argument("--bpe_vocab", type=int, default=1024)
    # Resume
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Specific checkpoint path (default: auto)")
    # Logging
    parser.add_argument("--log_file", type=str, default=None)
    # Event-driven training
    parser.add_argument("--event_driven", action="store_true",
                        help="Use event-driven gradient training (1.5-2x more efficient)")
    parser.add_argument("--reuse_threshold", type=float, default=0.10,
                        help="Novelty threshold for L0 reuse (skip)")
    parser.add_argument("--approx_threshold", type=float, default=0.25,
                        help="Novelty threshold for L1 approx")
    parser.add_argument("--decision_interval", type=int, default=4,
                        help="Run scout every N steps to re-decide levels")
    args = parser.parse_args()

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== AWF LLM Training ===")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    print(f"Model: {args.model}, d={args.d_model}, L={args.n_layers}, h={args.n_heads}")

    # Resolve dataset paths relative to repo root
    dataset_paths = [os.path.join(ROOT, p) if not os.path.isabs(p) else p for p in args.datasets]
    print(f"Datasets: {dataset_paths}")
    text = load_datasets(dataset_paths, max_chars_per_dataset=args.max_chars)
    print(f"Total corpus: {len(text):,} chars")

    # Tokenizer
    if args.tokenizer == "bpe":
        tok_path = os.path.join(CKPT_DIR, "bpe_tokenizer.json")
        tokenizer = BPETokenizer(vocab_size=args.bpe_vocab)
        if os.path.exists(tok_path) and args.resume:
            tokenizer.load(tok_path)
            print(f"Loaded BPE tokenizer: {tokenizer.vocab_size} tokens")
        else:
            tokenizer.train(text[:500000])  # train on first 500K chars for speed
            tokenizer.save(tok_path)
            print(f"Saved BPE tokenizer to {tok_path}")
    else:
        tokenizer = ByteTokenizer()
    print(f"Tokenizer: {args.tokenizer}, vocab: {tokenizer.vocab_size}")

    n_train = int(0.95 * len(text))
    train_text = text[:n_train]
    val_text = text[n_train:]
    train_ds = TextDataset(train_text, tokenizer, args.block_size)
    val_ds = TextDataset(val_text, tokenizer, args.block_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Train: {len(train_ds):,} samples ({len(train_loader)} batches)")
    print(f"Val: {len(val_ds):,} samples")

    # Build model
    if args.model == "awf":
        model = AWFTransformer(vocab_size=tokenizer.vocab_size, d_model=args.d_model,
                              n_layers=args.n_layers, n_heads=args.n_heads,
                              block_size=args.block_size, residual_rank=args.residual_rank,
                              sparse_k=args.sparse_k,
                              gen_kwargs=dict(n_fourier=16, hidden=128, n_layers=3), gen_grid=16)
    else:
        model = DenseTransformer(vocab_size=tokenizer.vocab_size, d_model=args.d_model,
                                 n_layers=args.n_layers, n_heads=args.n_heads,
                                 block_size=args.block_size)
    model = model.to(device)

    # Checkpoint path
    ckpt_path = args.checkpoint or os.path.join(CKPT_DIR, f"{args.model}_10m.pt")

    # Resume
    start_epoch = 0
    global_step = 0
    if args.resume and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        # Handle vocab mismatch — if checkpoint vocab differs, can't resume cleanly
        try:
            model.load_state_dict(state["model"], strict=False)
            start_epoch = state.get("epoch", 0)
            global_step = state.get("step", 0)
            print(f"\nResumed from {ckpt_path}")
            print(f"  Epoch: {start_epoch}, Step: {global_step}")
            print(f"  Previous val_loss: {state.get('val_loss', '?')}")
            if args.model == "awf" and start_epoch >= args.sparse_warmup_epoch and args.sparse_k > 0:
                try:
                    model.activate_sparse_corrections()
                    print(f"  Sparse corrections activated")
                except Exception as e:
                    print(f"  (sparse: {e})")
        except Exception as e:
            print(f"  Resume failed (architecture mismatch?): {e}")
            print(f"  Starting fresh")

    n_params = num_params(model)
    print(f"\n{args.model.upper()} model: {n_params:,} params ({n_params/1e6:.2f}M)")
    if args.model == "awf":
        print(f"  Generator: {num_params(model.generator):,}")
        print(f"  Storage fp16: {model.total_storage_bytes(2)/1024:.1f}KB")

    # Initial eval
    val_loss, val_acc = evaluate(model, val_loader, device, max_batches=20)
    print(f"Initial: val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}% ppl={math.exp(min(val_loss,20)):.1f}")

    # Train
    if args.event_driven and args.model == "awf":
        print("\n=== EVENT-DRIVEN TRAINING MODE ===")
        print(f"  L0 reuse threshold: {args.reuse_threshold}")
        print(f"  L1 approx threshold: {args.approx_threshold}")
        print(f"  Decision interval: every {args.decision_interval} steps")
        trainer = GradientEventTrainer(
            model, lr=args.lr,
            reuse_threshold=args.reuse_threshold,
            approx_threshold=args.approx_threshold,
            decision_interval=args.decision_interval,
        )
        # Load optimizer state if resuming
        if args.resume and os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location=device)
            if "optimizer" in state:
                try:
                    trainer.optimizer.load_state_dict(state["optimizer"])
                    print("Optimizer state restored (event trainer)")
                except: pass
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        if args.resume and os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location=device)
            if "optimizer" in state:
                try:
                    opt.load_state_dict(state["optimizer"])
                    print("Optimizer state restored")
                except: pass
        trainer = None

    t0 = time.time()
    batch_count = 0
    log_lines = []

    for epoch in range(start_epoch, start_epoch + args.epochs):
        # Activate sparse corrections after warmup
        if args.model == "awf" and args.sparse_k > 0 and epoch >= args.sparse_warmup_epoch and not getattr(model, '_sparse_activated', False):
            print(f"\nActivating sparse corrections (k={args.sparse_k}/layer)")
            try:
                model.activate_sparse_corrections()
                model._sparse_activated = True
            except Exception as e:
                print(f"  (sparse: {e})")

        model.train()
        ep_loss, n_seen = 0.0, 0
        for x, y in train_loader:
            if time.time() - t0 > args.time_budget:
                print(f"\nTime budget hit ({args.time_budget}s). Saving and exiting.")
                break

            x, y = x.to(device), y.to(device)

            if trainer is not None:
                # Event-driven path
                loss_val, stats = trainer.step(x, y)
                loss = torch.tensor(loss_val)  # for logging
            else:
                # Standard path
                lr = lr_schedule(global_step, args.warmup_steps, args.lr)
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
                lr_now = trainer.optimizer.param_groups[0]['lr'] if trainer else lr
                line = f"  ep{epoch+1} batch {batch_count} step {global_step} loss={loss.item():.4f} lr={lr_now:.5f} ({elapsed:.0f}s)"
                print(line)
                log_lines.append(line)

            if batch_count % args.save_every == 0:
                # Save with whichever optimizer is active
                opt_state = trainer.optimizer.state_dict() if trainer else opt.state_dict()
                torch.save({"model": model.state_dict(), "optimizer": opt_state,
                           "epoch": epoch, "step": global_step, "val_loss": val_loss,
                           "config": vars(args)}, ckpt_path)
                val_loss, val_acc = evaluate(model, val_loader, device, max_batches=20)
                model.train()
                line = f"  >> Checkpoint. val_loss={val_loss:.4f} acc={val_acc*100:.2f}% ppl={math.exp(min(val_loss,20)):.1f}"
                print(line)
                log_lines.append(line)
        else:
            # Epoch completed
            val_loss, val_acc = evaluate(model, val_loader, device, max_batches=50)
            line = f"[{args.model}] ep{epoch+1} done: train={ep_loss/max(n_seen,1):.4f} val={val_loss:.4f} acc={val_acc*100:.2f}% ppl={math.exp(min(val_loss,20)):.1f} ({time.time()-t0:.0f}s)"
            print(line)
            log_lines.append(line)
            opt_state = trainer.optimizer.state_dict() if trainer else opt.state_dict()
            torch.save({"model": model.state_dict(), "optimizer": opt_state,
                       "epoch": epoch + 1, "step": global_step, "val_loss": val_loss,
                       "config": vars(args)}, ckpt_path)
            continue
        break  # time budget hit

    # Final save
    val_loss, val_acc = evaluate(model, val_loader, device, max_batches=50)
    opt_state = trainer.optimizer.state_dict() if trainer else opt.state_dict()
    torch.save({"model": model.state_dict(), "optimizer": opt_state,
               "epoch": start_epoch + args.epochs, "step": global_step, "val_loss": val_loss,
               "config": vars(args)}, ckpt_path)

    print(f"\n=== FINAL ===")
    print(f"{args.model}: val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}% ppl={math.exp(min(val_loss,20)):.1f}")
    print(f"Params: {n_params:,} ({n_params/1e6:.2f}M)")
    print(f"Steps: {global_step}")
    print(f"Checkpoint: {ckpt_path}")

    # Generate samples
    print(f"\n=== Sample Generation ===")
    prompts = ["Once upon a time", "The little girl", "A boy named", "In the forest", "Today I learned"]
    samples = {}
    for p in prompts:
        s = generate(model, tokenizer, device, p, n_tokens=100, temperature=0.8, top_k=20, top_p=0.9, seed=42)
        print(f"\n{p!r} -> {s!r}")
        samples[p] = s

    # Save log
    if args.log_file:
        with open(args.log_file, "w") as f:
            f.write("\n".join(log_lines))
    elif device == "cpu":
        log_path = os.path.join(LOG_DIR, f"{args.model}_train_log.json")
        with open(log_path, "w") as f:
            json.dump({"step": global_step, "val_loss": val_loss, "val_acc": val_acc,
                       "samples": samples, "log_lines": log_lines}, f, indent=2)


if __name__ == "__main__":
    main()
