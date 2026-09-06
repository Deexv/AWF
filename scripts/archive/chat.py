"""
Interactive chat with the AWF 10M LLM.

Usage:
    python scripts/chat.py
    python scripts/chat.py --prompt "Once upon a time"
    python scripts/chat.py --interactive

The model generates text using the pre-trained AWF checkpoint.
Supports:
  - Single prompt generation
  - Interactive REPL mode
  - Streaming output (token by token)
  - Temperature and top-k control
"""
import os, sys, argparse, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn.functional as F
from awf.core import AWFTransformer, DenseTransformer, num_params

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")
BLOCK = 128
D_MODEL = 256
N_LAYERS = 6
N_HEADS = 8
RESIDUAL_RANK = 16
SPARSE_K = 256


class ByteTokenizer:
    def __init__(self):
        self.vocab_size = 256
    def encode(self, text):
        return list(text.encode('utf-8'))
    def decode(self, ids):
        ids = [i for i in ids if i < 256]
        return bytes(ids).decode('utf-8', errors='ignore')


def load_model(model_type="awf"):
    """Load the trained AWF or Dense model."""
    tokenizer = ByteTokenizer()
    if model_type == "awf":
        model = AWFTransformer(vocab_size=256, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
                              block_size=BLOCK, residual_rank=RESIDUAL_RANK, sparse_k=SPARSE_K,
                              gen_kwargs=dict(n_fourier=16, hidden=128, n_layers=3), gen_grid=16)
        ckpt_path = os.path.join(CKPT_DIR, "awf_10m.pt")
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state["model"])
            print(f"Loaded AWF checkpoint (epoch {state.get('epoch','?')}, step {state.get('step','?')})")
            # Activate sparse corrections if needed
            if SPARSE_K > 0:
                try:
                    model.activate_sparse_corrections()
                except: pass
        else:
            print(f"WARNING: No AWF checkpoint at {ckpt_path}")
    else:
        model = DenseTransformer(vocab_size=256, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS, block_size=BLOCK)
        ckpt_path = os.path.join(CKPT_DIR, "dense_10m.pt")
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state["model"])
            print(f"Loaded Dense checkpoint (epoch {state.get('epoch','?')}, step {state.get('step','?')})")
        else:
            print(f"WARNING: No Dense checkpoint at {ckpt_path}")
    model.eval()
    n_params = num_params(model)
    print(f"{model_type.upper()} model: {n_params:,} params ({n_params/1e6:.2f}M)")
    return model, tokenizer


def generate_stream(model, tokenizer, prompt, n_tokens=150, temperature=0.7, top_k=10, seed=None):
    """Generate text token by token, yielding each decoded chunk for streaming."""
    if seed is not None: torch.manual_seed(seed)
    model.eval()
    ids = tokenizer.encode(prompt)
    if not ids: ids = [0]
    # Yield the prompt first
    yield tokenizer.decode(ids)

    with torch.no_grad():
        for _ in range(n_tokens):
            x = torch.tensor(ids[-BLOCK:], dtype=torch.long).unsqueeze(0)
            logits = model(x)
            nl = logits[0, -1] / max(temperature, 0.01)
            if top_k > 0:
                v, _ = torch.topk(nl, min(top_k, tokenizer.vocab_size))
                nl[nl < v[-1]] = -float("inf")
            probs = F.softmax(nl, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            ids.append(next_id)
            yield tokenizer.decode([next_id])


def generate(model, tokenizer, prompt, n_tokens=150, temperature=0.7, top_k=10, seed=None):
    """Generate text (non-streaming)."""
    return "".join(generate_stream(model, tokenizer, prompt, n_tokens, temperature, top_k, seed))


def interactive_chat(model, tokenizer, temperature=0.7, top_k=10):
    """Run an interactive chat session."""
    print("\n" + "="*60)
    print("AWF LLM Chat (type 'quit' or 'exit' to stop)")
    print("="*60)
    print(f"Model: {type(model).__name__}, {num_params(model):,} params")
    print(f"Settings: temperature={temperature}, top_k={top_k}")
    print()

    while True:
        try:
            prompt = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print("AI> ", end="", flush=True)
        try:
            for chunk in generate_stream(model, tokenizer, prompt, n_tokens=200,
                                         temperature=temperature, top_k=top_k, seed=None):
                print(chunk, end="", flush=True)
        except Exception as e:
            print(f"\n[Error: {e}]")
        print("\n")


def main():
    parser = argparse.ArgumentParser(description="Chat with the AWF 10M LLM")
    parser.add_argument("--prompt", type=str, help="Single prompt to generate from")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive chat mode")
    parser.add_argument("--model", choices=["awf", "dense"], default="awf", help="Which model to use")
    parser.add_argument("--tokens", type=int, default=150, help="Max tokens to generate")
    parser.add_argument("--temperature", "-t", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", "-k", type=int, default=10, help="Top-k sampling")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)

    if args.prompt:
        print(f"\nPrompt: {args.prompt!r}")
        print(f"Generated:")
        for chunk in generate_stream(model, tokenizer, args.prompt, args.tokens,
                                      args.temperature, args.top_k, args.seed):
            print(chunk, end="", flush=True)
        print()
    elif args.interactive:
        interactive_chat(model, tokenizer, args.temperature, args.top_k)
    else:
        # Default: show a few sample generations
        print("\nGenerating samples (use --interactive for chat mode, --prompt '...' for single prompt):")
        prompts = [
            "Once upon a time",
            "The little girl",
            "A boy named Tom",
            "In the forest",
            "Today I learned",
        ]
        for p in prompts:
            print(f"\n--- {p!r} ---")
            for chunk in generate_stream(model, tokenizer, p, n_tokens=100,
                                         temperature=args.temperature, top_k=args.top_k, seed=42):
                print(chunk, end="", flush=True)
            print()


if __name__ == "__main__":
    main()
