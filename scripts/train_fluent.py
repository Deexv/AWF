"""
Train a fluent AWF LLM on Google Colab GPU.

This is the PRODUCTION training script that produces fluent English output.
Key differences from the prototype:
  - BPE tokenizer (1024 vocab, 4x more efficient than byte-level)
  - Bigger model (d=384, 8 layers → 1.4M params)
  - Output caching for 10x training speedup
  - Full checkpoint with BPE tokenizer embedded
  - Resume from checkpoint (same or different dataset)
  - Fluency evaluation

Usage on Colab:
  !python scripts/train_fluent.py --epochs 5 --batch_size 32
  !python scripts/train_fluent.py --resume --epochs 5 --batch_size 32
  !python scripts/train_fluent.py --resume --datasets data/tinystories.txt my_book.txt --epochs 3

Usage on CPU (slower but works):
  !python scripts/train_fluent.py --time_budget 510 --batch_size 16
"""
import os, sys, time, math, json, argparse, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from awf.core import AWFTransformer, DenseTransformer, num_params
from awf.output_caching_trainer import OutputCachingTrainer

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
LOG_DIR = os.path.join(ROOT, "benchmarks")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ============================================================================
# BPE Tokenizer (saved with checkpoint for resume)
# ============================================================================
class BPETokenizer:
    """BPE tokenizer that saves/loads with the checkpoint."""
    def __init__(self, vocab_size=1024):
        self.vocab_size = vocab_size
        self.merges = []
        self.vocab = [bytes([i]) for i in range(256)]

    def _get_pair_counts(self, ids):
        counts = {}
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i+1])
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def train(self, text, target_vocab_size=None):
        target = target_vocab_size or self.vocab_size
        ids = list(text.encode('utf-8'))
        self.vocab = [bytes([i]) for i in range(256)]
        print(f"Training BPE: {len(self.vocab)} → {target} tokens...")
        while len(self.vocab) < target:
            counts = self._get_pair_counts(ids)
            if not counts: break
            best_pair = max(counts, key=counts.get)
            new_id = len(self.vocab)
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
            if len(self.vocab) % 200 == 0:
                print(f"  {len(self.vocab)} tokens")
        self.vocab_size = len(self.vocab)
        print(f"BPE trained: {self.vocab_size} tokens")

    def encode(self, text):
        ids = list(text.encode('utf-8'))
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

    def to_dict(self):
        return {"vocab_size": self.vocab_size,
                "merges": [[a, b] for a, b in self.merges],
                "vocab": [list(v) for v in self.vocab]}

    def from_dict(self, d):
        self.vocab_size = d["vocab_size"]
        self.merges = [tuple(m) for m in d["merges"]]
        self.vocab = [bytes(v) for v in d["vocab"]]


class ByteTokenizer:
    def __init__(self): self.vocab_size = 256
    def encode(self, t): return list(t.encode('utf-8'))
    def decode(self, ids): return bytes([i for i in ids if i < 256]).decode('utf-8', errors='ignore')
    def to_dict(self): return {"type": "byte"}
    def from_dict(self, d): pass


# ============================================================================
# Dataset
# ============================================================================
class TextDataset(Dataset):
    def __init__(self, text, tokenizer, block_size):
        self.data = tokenizer.encode(text)
        self.block_size = block_size
    def __len__(self): return max(0, len(self.data) - self.block_size - 1)
    def __getitem__(self, i):
        c = self.data[i:i + self.block_size + 1]
        return torch.tensor(c[:-1], dtype=torch.long), torch.tensor(c[1:], dtype=torch.long)


# ============================================================================
# Config — tuned for fluency
# ============================================================================
# Model: d=384, 8 layers → ~1.4M params (2.3x bigger than prototype)
# BPE: 1024 vocab (4x more efficient than byte-level)
# Output caching: 10x training speedup
D_MODEL = 384
N_LAYERS = 8
N_HEADS = 8
RESIDUAL_RANK = 16
SPARSE_K = 256
BLOCK = 256        # longer context (was 128) — better for BPE
BPE_VOCAB = 1024   # BPE tokenizer vocab


def load_datasets(paths, max_chars_per_dataset=None, seed=42):
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


def evaluate(model, loader, device, max_batches=30):
    for p in model.parameters():
        if p.is_floating_point(): p.requires_grad_(True)
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


def generate(model, tokenizer, device, prompt, n_tokens=200, temperature=0.7, top_k=30, top_p=0.9, repetition_penalty=1.2, seed=None):
    if seed is not None: torch.manual_seed(seed)
    model.eval()
    ids = tokenizer.encode(prompt)
    if not ids: ids = [0]
    with torch.no_grad():
        for _ in range(n_tokens):
            x = torch.tensor([ids[-BLOCK:]], dtype=torch.long, device=device)
            logits = model(x)
            nl = logits[0, -1] / max(temperature, 0.01)
            if repetition_penalty != 1.0:
                recent = ids[-50:]
                for t in set(recent):
                    if t < nl.size(-1):
                        nl[t] = nl[t] / repetition_penalty
            if top_k > 0:
                v, _ = torch.topk(nl, min(top_k, nl.size(-1)))
                nl[nl < v[-1]] = -float("inf")
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(nl, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cum_probs > top_p
                sorted_mask[1:] = sorted_mask[:-1].clone()
                sorted_mask[0] = False
                indices_to_remove = sorted_mask.scatter(-1, sorted_indices, sorted_mask)
                nl = nl.masked_fill(indices_to_remove, -float("inf"))
            probs = F.softmax(nl, dim=-1)
            ids.append(torch.multinomial(probs, 1).item())
    return tokenizer.decode(ids)


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train a fluent AWF LLM")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--datasets", nargs="+", default=["data/tinystories_train.txt"],
                        help="One or more text files to train on")
    parser.add_argument("--max_chars", type=int, default=5000000, help="Max chars per dataset")
    parser.add_argument("--epochs", type=int, default=1, help="Max epochs")
    parser.add_argument("--time_budget", type=int, default=999999, help="Max seconds (default: unlimited)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (32 for GPU, 16 for CPU)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_every", type=int, default=500, help="Save checkpoint every N batches")
    parser.add_argument("--use_bpe", action="store_true", default=True, help="Use BPE tokenizer (recommended)")
    parser.add_argument("--bpe_vocab", type=int, default=BPE_VOCAB)
    parser.add_argument("--use_output_cache", action="store_true", default=True, help="Use output caching (10x speedup)")
    parser.add_argument("--cache_max_staleness", type=int, default=5, help="Output cache staleness (5=conservative, 9999=extreme)")
    parser.add_argument("--d_model", type=int, default=D_MODEL)
    parser.add_argument("--n_layers", type=int, default=N_LAYERS)
    parser.add_argument("--n_heads", type=int, default=N_HEADS)
    parser.add_argument("--checkpoint_name", type=str, default="awf_fluent.pt",
                        help="Checkpoint filename (use different names for different models)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== AWF Fluent LLM Training ===")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    print(f"Model: d={args.d_model}, L={args.n_layers}, h={args.n_heads}")
    print(f"Tokenizer: {'BPE ' + str(args.bpe_vocab) if args.use_bpe else 'byte-level'}")
    print(f"Output caching: {args.use_output_cache} (staleness={args.cache_max_staleness})")

    # Load data
    dataset_paths = [os.path.join(ROOT, p) if not os.path.isabs(p) else p for p in args.datasets]
    print(f"\nDatasets: {dataset_paths}")
    text = load_datasets(dataset_paths, max_chars_per_dataset=args.max_chars)
    print(f"Total corpus: {len(text):,} chars")

    # Tokenizer
    ckpt_path = os.path.join(CKPT_DIR, args.checkpoint_name)
    bpe_path = os.path.join(CKPT_DIR, "bpe_fluent.json")

    if args.use_bpe:
        tokenizer = BPETokenizer(vocab_size=args.bpe_vocab)
        if args.resume and os.path.exists(bpe_path):
            with open(bpe_path) as f:
                tokenizer.from_dict(json.load(f))
            print(f"Loaded BPE tokenizer: {tokenizer.vocab_size} tokens")
        elif not args.resume or not os.path.exists(ckpt_path):
            # Train BPE on a subset for speed (first 100K chars)
            tokenizer.train(text[:100000])
            with open(bpe_path, "w") as f:
                json.dump(tokenizer.to_dict(), f)
            print(f"Saved BPE tokenizer to {bpe_path}")
        else:
            # Resuming but BPE not found — try to load from checkpoint
            if os.path.exists(ckpt_path):
                state = torch.load(ckpt_path, map_location="cpu")
                if "tokenizer" in state:
                    tokenizer.from_dict(state["tokenizer"])
                    print(f"Loaded BPE from checkpoint: {tokenizer.vocab_size} tokens")
                else:
                    tokenizer.train(text[:500000])
                    with open(bpe_path, "w") as f:
                        json.dump(tokenizer.to_dict(), f)
    else:
        tokenizer = ByteTokenizer()
    print(f"Tokenizer vocab: {tokenizer.vocab_size}")

    # Dataset
    n_train = int(0.95 * len(text))
    train_ds = TextDataset(text[:n_train], tokenizer, BLOCK)
    val_ds = TextDataset(text[n_train:], tokenizer, BLOCK)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Train: {len(train_ds):,} samples ({len(train_loader)} batches)")
    print(f"Val: {len(val_ds):,} samples")

    # Build model
    model = AWFTransformer(vocab_size=tokenizer.vocab_size, d_model=args.d_model,
                          n_layers=args.n_layers, n_heads=args.n_heads, block_size=BLOCK,
                          residual_rank=RESIDUAL_RANK, sparse_k=SPARSE_K,
                          gen_kwargs=dict(n_fourier=16, hidden=128, n_layers=3), gen_grid=16)
    model = model.to(device)

    # Resume
    start_step = 0
    if args.resume and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        try:
            model.load_state_dict(state["model"], strict=False)
            start_step = state.get("step", 0)
            print(f"\nResumed from {ckpt_path} (step {start_step})")
            # Activate sparse corrections if past warmup
            if SPARSE_K > 0:
                try: model.activate_sparse_corrections()
                except: pass
        except Exception as e:
            print(f"Resume failed: {e}. Starting fresh.")

    n_params = num_params(model)
    print(f"\nAWF model: {n_params:,} params ({n_params/1e6:.2f}M)")
    if device == "cpu":
        print(f"Storage fp16: {model.total_storage_bytes(2)/1024:.1f}KB")

    # Evaluate before training
    val_loss, val_acc = evaluate(model, val_loader, device, max_batches=30)
    print(f"Initial: val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}% ppl={math.exp(min(val_loss,20)):.1f}")

    # Trainer
    if args.use_output_cache:
        print(f"\nUsing OutputCachingTrainer (staleness={args.cache_max_staleness})")
        from awf.event_training import StandardTrainer as _ST  # fallback
        trainer = OutputCachingTrainer(model, lr=args.lr,
            reuse_threshold=0.80, max_staleness=args.cache_max_staleness,
            warmup_steps=10, min_full_blocks=1)
        optimizer = trainer.optimizer
    else:
        from awf.event_training import StandardTrainer
        trainer = StandardTrainer(model, lr=args.lr)
        optimizer = trainer.optimizer

    # Train
    t0 = time.time()
    step = start_step
    log_lines = []

    for epoch in range(args.epochs):
        if args.use_output_cache:
            trainer.model.train()
        else:
            model.train()

        for batch_idx, (x, y) in enumerate(train_loader):
            if time.time() - t0 > args.time_budget:
                print(f"\nTime budget hit ({args.time_budget}s). Saving and exiting.")
                break

            x, y = x.to(device), y.to(device)

            if args.use_output_cache:
                loss_val, stats = trainer.step(x, y)
                loss = torch.tensor(loss_val)
            else:
                optimizer.zero_grad()
                logits = model(x)
                V = logits.size(-1)
                loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            step += 1

            if step % 100 == 0:
                elapsed = time.time() - t0
                lr_now = optimizer.param_groups[0]['lr']
                skip_info = f" skip%={stats.get('skip_pct', 0)*100:.0f}" if args.use_output_cache else ""
                line = f"  ep{epoch+1} step {step} loss={loss.item():.4f} lr={lr_now:.5f}{skip_info} ({elapsed:.0f}s)"
                print(line)
                log_lines.append(line)

            if step % args.save_every == 0:
                # Save checkpoint with tokenizer
                opt_state = trainer.optimizer.state_dict() if args.use_output_cache else optimizer.state_dict()
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": opt_state,
                    "step": step,
                    "val_loss": val_loss,
                    "tokenizer": tokenizer.to_dict() if args.use_bpe else None,
                    "config": {"d_model": args.d_model, "n_layers": args.n_layers,
                               "n_heads": args.n_heads, "block_size": BLOCK,
                               "vocab_size": tokenizer.vocab_size,
                               "residual_rank": RESIDUAL_RANK, "sparse_k": SPARSE_K},
                }, ckpt_path)
                # Quick eval
                val_loss, val_acc = evaluate(model, val_loader, device, max_batches=20)
                if args.use_output_cache: trainer.model.train()
                else: model.train()
                line = f"  >> Checkpoint. val_loss={val_loss:.4f} acc={val_acc*100:.2f}% ppl={math.exp(min(val_loss,20)):.1f}"
                print(line)
                log_lines.append(line)

                # Generate sample
                sample = generate(model, tokenizer, device, "Once upon a time", n_tokens=100, seed=42)
                print(f"     Sample: {sample[:120]}")

        else:
            val_loss, val_acc = evaluate(model, val_loader, device, max_batches=50)
            line = f"[ep{epoch+1}] val_loss={val_loss:.4f} acc={val_acc*100:.2f}% ppl={math.exp(min(val_loss,20)):.1f} ({time.time()-t0:.0f}s)"
            print(line)
            log_lines.append(line)
            opt_state = trainer.optimizer.state_dict() if args.use_output_cache else optimizer.state_dict()
            torch.save({
                "model": model.state_dict(), "optimizer": opt_state,
                "step": step, "val_loss": val_loss,
                "tokenizer": tokenizer.to_dict() if args.use_bpe else None,
                "config": {"d_model": args.d_model, "n_layers": args.n_layers,
                           "n_heads": args.n_heads, "block_size": BLOCK,
                           "vocab_size": tokenizer.vocab_size,
                           "residual_rank": RESIDUAL_RANK, "sparse_k": SPARSE_K},
            }, ckpt_path)
            continue
        break

    # Final save
    val_loss, val_acc = evaluate(model, val_loader, device, max_batches=50)
    opt_state = trainer.optimizer.state_dict() if args.use_output_cache else optimizer.state_dict()
    torch.save({
        "model": model.state_dict(), "optimizer": opt_state,
        "step": step, "val_loss": val_loss,
        "tokenizer": tokenizer.to_dict() if args.use_bpe else None,
        "config": {"d_model": args.d_model, "n_layers": args.n_layers,
                   "n_heads": args.n_heads, "block_size": BLOCK,
                   "vocab_size": tokenizer.vocab_size,
                   "residual_rank": RESIDUAL_RANK, "sparse_k": SPARSE_K},
    }, ckpt_path)

    if args.use_output_cache:
        trainer.close()

    print(f"\n=== FINAL ===")
    print(f"Step: {step}")
    print(f"Val loss: {val_loss:.4f}, Val acc: {val_acc*100:.2f}%")
    print(f"Perplexity: {math.exp(min(val_loss,20)):.1f}")
    print(f"Checkpoint: {ckpt_path}")

    # Generate samples
    print(f"\n=== Text Generation (fluency test) ===")
    prompts = [
        "Once upon a time",
        "The little girl",
        "A boy named Tom",
        "In the forest",
        "Today I learned",
    ]
    samples = {}
    for p in prompts:
        s = generate(model, tokenizer, device, p, n_tokens=150, temperature=0.7, top_k=30, seed=42)
        print(f"\n--- {p!r} ---")
        print(s)
        samples[p] = s

    # Save results
    with open(os.path.join(LOG_DIR, "fluent_training_log.json"), "w") as f:
        json.dump({"step": step, "val_loss": val_loss, "val_acc": val_acc,
                   "samples": samples, "log_lines": log_lines,
                   "config": {"d_model": args.d_model, "n_layers": args.n_layers,
                              "n_heads": args.n_heads, "vocab_size": tokenizer.vocab_size,
                              "use_bpe": args.use_bpe, "use_output_cache": args.use_output_cache}}, f, indent=2)
    print(f"\nLog saved to benchmarks/fluent_training_log.json")


if __name__ == "__main__":
    main()
