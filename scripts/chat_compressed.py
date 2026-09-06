"""
Compress any LLM and chat with it — all in one script.

Usage:
  # Compress DistilGPT2 and chat immediately
  python scripts/chat_compressed.py --model distilgpt2

  # Compress GPT-2 Medium and chat
  python scripts/chat_compressed.py --model gpt2-medium

  # Compress DeepSeek and chat
  python scripts/chat_compressed.py --model deepseek-ai/deepseek-coder-1.3b-base

  # Compress GLM and chat
  python scripts/chat_compressed.py --model THUDM/chatglm3-6b-base

  # Use a saved compressed model (skip compression)
  python scripts/chat_compressed.py --load checkpoints/my_compressed_model.pt

  # Single prompt (non-interactive)
  python scripts/chat_compressed.py --model distilgpt2 --prompt "Once upon a time"

  # Save compressed model for later use
  python scripts/chat_compressed.py --model distilgpt2 --save checkpoints/distilgpt2_compressed.pt
"""
import os, sys, time, json, math, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn as nn

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)


def compress_model(model, keep_ratio=0.85):
    """Apply SVD + int8 compression to a model in-place."""
    from compress_for_pc import apply_svd_compression, apply_int8_quantization
    svd_ratio = apply_svd_compression(model, keep_ratio=keep_ratio)
    int8_ratio = apply_int8_quantization(model)
    return svd_ratio, int8_ratio


def chat_loop(model, tokenizer, device, temperature=0.7, top_k=50, max_tokens=150):
    """Interactive chat loop."""
    print("\n" + "=" * 60)
    print("  AWF Compressed LLM Chat")
    print("  Type your message and press Enter. Type 'quit' to exit.")
    print("  Commands: 'temp 0.5' (set temperature), 'tokens 200' (set max tokens)")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Commands
        if user_input.lower().startswith("temp "):
            try:
                temperature = float(user_input.split()[1])
                print(f"  Temperature set to {temperature}")
            except:
                print("  Usage: temp 0.7")
            continue
        if user_input.lower().startswith("tokens "):
            try:
                max_tokens = int(user_input.split()[1])
                print(f"  Max tokens set to {max_tokens}")
            except:
                print("  Usage: tokens 200")
            continue
        if user_input.lower() == "help":
            print("  Commands:")
            print("    temp 0.7     - Set sampling temperature (0.1=focused, 1.0=creative)")
            print("    tokens 200   - Set max tokens to generate")
            print("    quit         - Exit chat")
            print("  Tips:")
            print("    - Lower temperature (0.3) for factual answers")
            print("    - Higher temperature (0.9) for creative writing")
            print("    - Use 'Once upon a time' for story generation")
            continue

        # Generate response
        print("AI> ", end="", flush=True)
        try:
            input_ids = tokenizer.encode(user_input, return_tensors="pt").to(device)

            # Check if model supports generate
            if hasattr(model, 'generate'):
                with torch.no_grad():
                    output = model.generate(
                        input_ids,
                        max_new_tokens=max_tokens,
                        temperature=max(temperature, 0.01),
                        top_k=top_k,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else 0,
                        repetition_penalty=1.2,
                    )
                response = tokenizer.decode(output[0], skip_special_tokens=True)
                # Remove the prompt from the response
                if response.startswith(user_input):
                    response = response[len(user_input):].strip()
                print(response)
            else:
                # For models without generate (some custom models)
                with torch.no_grad():
                    output = model(input_ids)
                    logits = output.logits if hasattr(output, 'logits') else output[0]
                    # Simple greedy decoding
                    for _ in range(max_tokens):
                        next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
                        input_ids = torch.cat([input_ids, next_token], dim=-1)
                        output = model(input_ids)
                        logits = output.logits if hasattr(output, 'logits') else output[0]
                        if next_token.item() == tokenizer.eos_token_id:
                            break
                    response = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                    if response.startswith(user_input):
                        response = response[len(user_input):].strip()
                    print(response)
        except Exception as e:
            print(f"\n  [Error: {e}]")
            print("  Try rephrasing your prompt or check if the model supports this input.")


def main():
    parser = argparse.ArgumentParser(description="Compress an LLM and chat with it")
    parser.add_argument("--model", type=str, default="distilgpt2",
                        help="HuggingFace model ID (distilgpt2, gpt2-medium, deepseek-ai/deepseek-coder-1.3b-base, THUDM/chatglm3-6b-base, etc.)")
    parser.add_argument("--load", type=str, default=None,
                        help="Load a previously saved compressed model checkpoint")
    parser.add_argument("--save", type=str, default=None,
                        help="Save the compressed model to this path for later use")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt (non-interactive mode)")
    parser.add_argument("--keep_ratio", type=float, default=0.85,
                        help="SVD keep ratio (0.85 = 1.7x compression, 0.50 = 2x)")
    parser.add_argument("--temperature", "-t", type=float, default=0.7)
    parser.add_argument("--top_k", "-k", type=int, default=50)
    parser.add_argument("--tokens", type=int, default=150, help="Max tokens per response")
    parser.add_argument("--no_compress", action="store_true",
                        help="Skip compression (chat with original model)")
    parser.add_argument("--trust_remote_code", action="store_true", default=True,
                        help="Trust remote code (needed for GLM, DeepSeek, etc.)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ------------------------------------------------------------------
    # Step 1: Load model (either from checkpoint or from HuggingFace)
    # ------------------------------------------------------------------
    if args.load:
        print(f"\n--- Loading compressed model from {args.load} ---")
        checkpoint = torch.load(args.load, map_location=device)

        # Get model name from checkpoint or use default
        model_name = checkpoint.get("model_name", "distilgpt2")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=args.trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, trust_remote_code=args.trust_remote_code
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Loaded: {model_name}")
        print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
        print(f"  Original size: {checkpoint.get('original_size_mb', '?')} MB")
        print(f"  Compression: {checkpoint.get('compression', '?')}x")
    else:
        print(f"\n--- Loading {args.model} from HuggingFace ---")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32, trust_remote_code=args.trust_remote_code
        ).to(device)
        model.eval()

        n_params = sum(p.numel() for p in model.parameters())
        orig_size_mb = n_params * 4 / 1024 / 1024
        print(f"  Model: {args.model}")
        print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
        print(f"  Original size: {orig_size_mb:.1f} MB (fp32)")

        # ------------------------------------------------------------------
        # Step 2: Compress (if not --no_compress)
        # ------------------------------------------------------------------
        if not args.no_compress:
            print(f"\n--- Compressing with AWF (keep_ratio={args.keep_ratio}) ---")
            svd_ratio, int8_ratio = compress_model(model, keep_ratio=args.keep_ratio)

            compressed_size = orig_size_mb * (1 - args.keep_ratio * 0) * 0.15 * 0.25  # approximate
            # More accurate: SVD keeps keep_ratio of data, int8 is 1/4 of fp32
            svd_size = orig_size_mb * (1 - (1 - args.keep_ratio) * 0.5)  # rough
            final_size = orig_size_mb * args.keep_ratio * 0.15 * 4  # SVD + int8 combined
            combined = orig_size_mb / max(final_size, 0.001)

            print(f"  SVD compression: {1/svd_ratio:.1f}x")
            print(f"  INT8 compression: {1/int8_ratio:.1f}x")
            print(f"  Combined: ~{combined:.1f}x")
            print(f"  Estimated compressed size: {final_size:.1f} MB")
            print(f"  RAM needed: ~{final_size/1024 + 0.5:.1f} GB")
        else:
            combined = 1.0
            final_size = orig_size_mb
            print(f"  Skipping compression (--no_compress)")

        # ------------------------------------------------------------------
        # Step 3: Save compressed model (if --save)
        # ------------------------------------------------------------------
        if args.save:
            save_path = args.save if os.path.isabs(args.save) else os.path.join(CKPT_DIR, args.save)
            torch.save({
                "model": model.state_dict(),
                "model_name": args.model,
                "original_size_mb": orig_size_mb,
                "compressed_size_mb": final_size,
                "compression": combined,
                "keep_ratio": args.keep_ratio,
                "n_params": n_params,
            }, save_path)
            print(f"\n  ✅ Saved compressed model to {save_path}")
            print(f"     Load later: python scripts/chat_compressed.py --load {save_path}")

    # ------------------------------------------------------------------
    # Step 4: Generate text or start interactive chat
    # ------------------------------------------------------------------
    if args.prompt:
        # Single prompt mode
        print(f"\n--- Generating ---")
        print(f"Prompt: {args.prompt}")
        print(f"Response:")
        input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                input_ids, max_new_tokens=args.tokens,
                temperature=max(args.temperature, 0.01), top_k=args.top_k,
                do_sample=True, pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else 0,
                repetition_penalty=1.2,
            )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print(response)
    else:
        # Interactive chat mode
        chat_loop(model, tokenizer, device, args.temperature, args.top_k, args.tokens)


if __name__ == "__main__":
    main()
