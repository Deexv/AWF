"""
demo.py — Minimal demo: chat with the COMPRESSED model.

This is the corrected version of the demo. It loads ONLY the compressed
GGUF file (Q4_K_M). The original (uncompressed) HuggingFace model is
NEVER loaded, NEVER referenced, NEVER needed at runtime.

Requirements:
    pip install llama-cpp-python

Usage:
    # Interactive chat
    python demo.py --model compressed/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf

    # One-shot prompt
    python demo.py --model compressed/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \\
        --prompt "Explain recursion in one sentence."

    # With a different system prompt
    python demo.py --model compressed/X.gguf --system "You are a pirate."

If you don't pass --model, it auto-detects the first .gguf file in
./compressed/ (so the demo works out of the box after running the
compress_model_to_gguf_q4_k_m.ipynb notebook or the included setup).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

try:
    from llama_cpp import Llama
except ImportError:
    print("ERROR: llama-cpp-python not installed.")
    print("  Install with:  pip install llama-cpp-python")
    sys.exit(1)


def find_default_model() -> str | None:
    """Auto-detect the first .gguf in ./compressed/."""
    candidates = sorted(glob.glob("compressed/*.gguf"))
    return candidates[0] if candidates else None


def main():
    ap = argparse.ArgumentParser(description="Chat with the COMPRESSED model only.")
    ap.add_argument("--model", default=None,
                    help="Path to the .gguf file. Auto-detects ./compressed/*.gguf if omitted.")
    ap.add_argument("--prompt", default=None,
                    help="One-shot prompt. If omitted, runs in interactive chat mode.")
    ap.add_argument("--system", default="You are a helpful, concise assistant.",
                    help="System prompt.")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--n-ctx", type=int, default=2048)
    ap.add_argument("--n-threads", type=int, default=4)
    ap.add_argument("--n-gpu-layers", type=int, default=0,
                    help="Set to 99 if you have a GPU available.")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--verbose", action="store_true",
                    help="Show llama.cpp internal logs (defaults to off).")
    args = ap.parse_args()

    model_path = args.model or find_default_model()
    if not model_path:
        print("ERROR: No model specified and no .gguf files found in ./compressed/.")
        print("  Either:")
        print("    1. Run the compress notebook first: notebooks/compress_model_to_gguf_q4_k_m.ipynb")
        print("    2. Or download a pre-quantized .gguf from https://huggingface.co/models?other=gguf")
        print("    3. Then pass --model path/to/your-model.gguf")
        sys.exit(2)
    if not os.path.exists(model_path):
        print(f"ERROR: model file not found: {model_path}")
        sys.exit(2)

    size_mib = os.path.getsize(model_path) / 1024 / 1024
    print(f"[demo] Loading COMPRESSED model: {model_path}")
    print(f"[demo] Model size: {size_mib:.1f} MiB")
    print(f"[demo] (No base model, no transformers, no PyTorch needed at runtime.)")
    sys.stdout.flush()

    llm = Llama(
        model_path=model_path,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_gpu_layers=args.n_gpu_layers,
        verbose=args.verbose,
    )

    print(f"[demo] Ready. max_tokens={args.max_tokens} temp={args.temperature}")
    print()

    # ---- One-shot mode ----
    if args.prompt:
        out = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": args.system},
                {"role": "user", "content": args.prompt},
            ],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        text = out["choices"][0]["message"]["content"].strip()
        usage = out.get("usage", {})
        print(f"> {args.prompt}\n")
        print(text)
        print(f"\n[usage] {usage}")
        return

    # ---- Interactive chat mode ----
    history = [{"role": "system", "content": args.system}]
    print(f"system: {args.system}")
    print("type 'quit' or Ctrl-D to exit, 'reset' to clear history.\n")

    while True:
        try:
            user = input("user> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            break
        if not user:
            continue
        if user.lower() in {"quit", "exit"}:
            break
        if user.lower() == "reset":
            history = [{"role": "system", "content": args.system}]
            print("[history cleared]\n")
            continue

        history.append({"role": "user", "content": user})
        out = llm.create_chat_completion(
            messages=history,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            stop=["<|im_end|>", "</s>"],
        )
        reply = out["choices"][0]["message"]["content"].strip()
        history.append({"role": "assistant", "content": reply})
        print(f"assistant> {reply}\n")


if __name__ == "__main__":
    main()
