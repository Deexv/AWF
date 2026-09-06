"""
Enhanced chat interface for the AWF LLM.

Features:
  - Better sampling (temperature, top-k, top-p, repetition penalty)
  - Streaming output (token by token)
  - Multi-prompt batch generation
  - Quality comparison vs Dense
  - Interactive REPL mode

Usage:
    python scripts/chat_v2.py                          # demo mode
    python scripts/chat_v2.py --interactive             # REPL
    python scripts/chat_v2.py --prompt "Once upon..."   # single
    python scripts/chat_v2.py --compare                  # AWF vs Dense
"""
import os, sys, argparse, json, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn.functional as F
from awf.core import AWFTransformer, DenseTransformer, num_params

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")
BLOCK = 128


class ByteTokenizer:
    def __init__(self):
        self.vocab_size = 256
    def encode(self, text):
        return list(text.encode('utf-8'))
    def decode(self, ids):
        ids = [i for i in ids if i < 256]
        return bytes(ids).decode('utf-8', errors='ignore')


def load_model(model_type="awf", device="cpu"):
    """Load AWF or Dense model with current architecture."""
    tokenizer = ByteTokenizer()
    if model_type == "awf":
        model = AWFTransformer(vocab_size=256, d_model=256, n_layers=6, n_heads=8,
                              block_size=BLOCK, residual_rank=16, sparse_k=256,
                              gen_kwargs=dict(n_fourier=16, hidden=128, n_layers=3), gen_grid=16)
        ckpt = os.path.join(CKPT_DIR, "awf_10m.pt")
    else:
        model = DenseTransformer(vocab_size=256, d_model=256, n_layers=6, n_heads=8, block_size=BLOCK)
        ckpt = os.path.join(CKPT_DIR, "dense_10m.pt")

    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"], strict=False)
        print(f"Loaded {model_type} checkpoint (epoch {state.get('epoch','?')}, step {state.get('step','?')})")
        if model_type == "awf" and hasattr(model, 'activate_sparse_corrections'):
            try: model.activate_sparse_corrections()
            except: pass
    else:
        print(f"WARNING: No checkpoint at {ckpt}")
    model = model.to(device)
    model.eval()
    n_params = num_params(model)
    print(f"{model_type.upper()} model: {n_params:,} params ({n_params/1e6:.2f}M)")
    return model, tokenizer


def generate_stream(model, tokenizer, prompt, n_tokens=200, temperature=0.8, top_k=30,
                    top_p=0.9, repetition_penalty=1.2, seed=None, device="cpu"):
    """Generate text with better sampling: temperature + top-k + top-p + repetition penalty."""
    if seed is not None: torch.manual_seed(seed)
    model.eval()
    ids = tokenizer.encode(prompt)
    if not ids: ids = [0]

    # Yield the prompt
    yield tokenizer.decode(ids)

    with torch.no_grad():
        for _ in range(n_tokens):
            x = torch.tensor([ids[-BLOCK:]], dtype=torch.long, device=device)
            logits = model(x)
            nl = logits[0, -1] / max(temperature, 0.01)

            # Repetition penalty: reduce probability of recently-used tokens
            if repetition_penalty != 1.0:
                recent = ids[-50:]  # last 50 tokens
                for t in set(recent):
                    if t < nl.size(-1):
                        nl[t] = nl[t] / repetition_penalty

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
                nl[nl != nl] = -float("inf")  # nan to -inf
                nl = nl.masked_fill(indices_to_remove, -float("inf"))

            probs = F.softmax(nl, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            ids.append(next_id)
            yield tokenizer.decode([next_id])


def generate(model, tokenizer, prompt, **kwargs):
    """Non-streaming generate."""
    return "".join(generate_stream(model, tokenizer, prompt, **kwargs))


def interactive_chat(model, tokenizer, temperature=0.8, top_k=30, top_p=0.9,
                     repetition_penalty=1.2, device="cpu"):
    """Interactive REPL chat."""
    print("\n" + "="*60)
    print("AWF LLM Chat (type 'quit' to exit, 'help' for commands)")
    print("="*60)
    print(f"Model: {type(model).__name__}, {num_params(model):,} params")
    print(f"Settings: temp={temperature}, top_k={top_k}, top_p={top_p}, rep_penalty={repetition_penalty}")
    print()

    while True:
        try:
            prompt = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not prompt: continue
        if prompt.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if prompt.lower() == "help":
            print("Commands:")
            print("  quit/exit     - Exit chat")
            print("  help          - Show this help")
            print("  temp 0.5      - Set temperature")
            print("  topk 50       - Set top_k")
            print("  topp 0.95     - Set top_p")
            print("  reppen 1.3    - Set repetition penalty")
            print("  status        - Show current settings")
            continue
        if prompt.lower().startswith("temp "):
            try: temperature = float(prompt.split()[1])
            except: print("Usage: temp 0.5")
            continue
        if prompt.lower().startswith("topk "):
            try: top_k = int(prompt.split()[1])
            except: print("Usage: topk 50")
            continue
        if prompt.lower().startswith("topp "):
            try: top_p = float(prompt.split()[1])
            except: print("Usage: topp 0.95")
            continue
        if prompt.lower().startswith("reppen "):
            try: repetition_penalty = float(prompt.split()[1])
            except: print("Usage: reppen 1.3")
            continue
        if prompt.lower() == "status":
            print(f"temp={temperature}, top_k={top_k}, top_p={top_p}, rep_penalty={repetition_penalty}")
            continue

        print("AI> ", end="", flush=True)
        try:
            for chunk in generate_stream(model, tokenizer, prompt, n_tokens=200,
                                         temperature=temperature, top_k=top_k, top_p=top_p,
                                         repetition_penalty=repetition_penalty, device=device):
                print(chunk, end="", flush=True)
        except Exception as e:
            print(f"\n[Error: {e}]")
        print("\n")


def main():
    parser = argparse.ArgumentParser(description="AWF LLM Chat (enhanced)")
    parser.add_argument("--prompt", type=str, help="Single prompt")
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--compare", action="store_true", help="Compare AWF vs Dense")
    parser.add_argument("--model", choices=["awf", "dense"], default="awf")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", "-t", type=float, default=0.8)
    parser.add_argument("--top_k", "-k", type=int, default=30)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.compare:
        print("=== AWF vs Dense Comparison ===\n")
        prompts = ["Once upon a time", "The little girl", "A boy named Tom", "In the forest"]
        awf_model, tok = load_model("awf", device)
        dense_model, _ = load_model("dense", device)
        for p in prompts:
            print(f"\n--- {p!r} ---")
            a = generate(awf_model, tok, p, n_tokens=args.tokens, temperature=args.temperature,
                         top_k=args.top_k, top_p=args.top_p, repetition_penalty=args.repetition_penalty,
                         seed=args.seed, device=device)
            d = generate(dense_model, tok, p, n_tokens=args.tokens, temperature=args.temperature,
                         top_k=args.top_k, top_p=args.top_p, repetition_penalty=args.repetition_penalty,
                         seed=args.seed, device=device)
            print(f"AWF:   {a}")
            print(f"DENSE: {d}")
        return

    model, tokenizer = load_model(args.model, device)

    if args.prompt:
        print(f"\nPrompt: {args.prompt!r}")
        print("Generated:")
        for chunk in generate_stream(model, tokenizer, args.prompt, n_tokens=args.tokens,
                                      temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                                      repetition_penalty=args.repetition_penalty, seed=args.seed, device=device):
            print(chunk, end="", flush=True)
        print()
    elif args.interactive:
        interactive_chat(model, tokenizer, args.temperature, args.top_k, args.top_p,
                        args.repetition_penalty, device)
    else:
        # Demo mode
        print("\n=== Demo Generation (use --interactive for chat, --prompt '...' for single) ===")
        prompts = ["Once upon a time", "The little girl", "A boy named", "In the forest", "Today I learned"]
        for p in prompts:
            print(f"\n--- {p!r} ---")
            for chunk in generate_stream(model, tokenizer, p, n_tokens=120,
                                         temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                                         repetition_penalty=args.repetition_penalty, seed=42, device=device):
                print(chunk, end="", flush=True)
            print()


if __name__ == "__main__":
    main()
