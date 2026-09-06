"""
Chat with the fluent AWF LLM.

Automatically loads the correct tokenizer (BPE or byte-level) from the checkpoint.
Supports: single prompt, interactive mode, comparison mode.

Usage:
    python scripts/chat_fluent.py --prompt "Once upon a time"
    python scripts/chat_fluent.py --interactive
    python scripts/chat_fluent.py --prompt "Once upon a time" --tokens 300 --temperature 0.6
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn.functional as F
from awf.core import AWFTransformer, DenseTransformer, num_params
from train_fluent import BPETokenizer, ByteTokenizer, generate, BLOCK

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")


def load_model(checkpoint_name="awf_fluent.pt", device="cpu"):
    """Load model + tokenizer from checkpoint."""
    ckpt_path = os.path.join(CKPT_DIR, checkpoint_name)
    if not os.path.exists(ckpt_path):
        print(f"ERROR: No checkpoint at {ckpt_path}")
        print(f"Train first: python scripts/train_fluent.py --epochs 1 --time_budget 1800")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]

    model = AWFTransformer(
        vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
        n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
        block_size=cfg["block_size"], residual_rank=cfg["residual_rank"],
        sparse_k=cfg["sparse_k"],
        gen_kwargs=dict(n_fourier=16, hidden=128, n_layers=3), gen_grid=16
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load tokenizer from checkpoint
    if ckpt.get("tokenizer"):
        tokenizer = BPETokenizer()
        tokenizer.from_dict(ckpt["tokenizer"])
    else:
        tokenizer = ByteTokenizer()

    print(f"Loaded: {checkpoint_name}")
    print(f"  Params: {num_params(model):,} ({num_params(model)/1e6:.2f}M)")
    print(f"  Step: {ckpt.get('step', '?')}")
    print(f"  Val loss: {ckpt.get('val_loss', '?')}")
    print(f"  Tokenizer: {type(tokenizer).__name__} (vocab={tokenizer.vocab_size})")
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Chat with the fluent AWF LLM")
    parser.add_argument("--prompt", type=str, help="Single prompt to generate from")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive chat mode")
    parser.add_argument("--checkpoint", type=str, default="awf_fluent.pt")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", "-t", type=float, default=0.7)
    parser.add_argument("--top_k", "-k", type=int, default=30)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(args.checkpoint, device)

    if args.prompt:
        print(f"\nPrompt: {args.prompt!r}")
        print(f"Generated ({args.tokens} tokens, temp={args.temperature}):")
        result = generate(model, tokenizer, device, args.prompt, n_tokens=args.tokens,
                         temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                         repetition_penalty=args.repetition_penalty, seed=args.seed)
        print(result)
    elif args.interactive:
        print(f"\n{'='*60}")
        print("AWF Fluent LLM Chat (type 'quit' to exit)")
        print(f"Settings: temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
        print(f"{'='*60}\n")
        while True:
            try:
                prompt = input("You> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            if not prompt: continue
            if prompt.lower() in ("quit", "exit", "q"):
                break
            result = generate(model, tokenizer, device, prompt, n_tokens=args.tokens,
                             temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                             repetition_penalty=args.repetition_penalty, seed=None)
            print(f"AI> {result}\n")
    else:
        # Demo mode
        print(f"\n=== Demo Generation ===")
        prompts = [
            "Once upon a time",
            "The little girl",
            "A boy named Tom",
            "In the forest",
            "Today I learned",
            "The old man smiled",
        ]
        for p in prompts:
            result = generate(model, tokenizer, device, p, n_tokens=120,
                             temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                             repetition_penalty=args.repetition_penalty, seed=42)
            print(f"\n--- {p!r} ---")
            print(result)


if __name__ == "__main__":
    main()
