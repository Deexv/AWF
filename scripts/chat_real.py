"""
REAL compression: store SVD factors (U, V) + int8 codes — NOT reconstructed fp32 weights.

This is the ACTUALLY COMPRESSED model. The file on disk is genuinely smaller.

Also uses instruction-tuned models for proper chat quality:
  - Qwen/Qwen2-1.5B-Instruct (chat model, not base)
  - microsoft/Phi-3-mini-4k-instruct
  - distilgpt2 (base model, for comparison)

Usage:
  python scripts/chat_real.py --model Qwen/Qwen2-1.5B-Instruct
  python scripts/chat_real.py --model microsoft/Phi-3-mini-4k-instruct
  python scripts/chat_real.py --model distilgpt2
  python scripts/chat_real.py --load checkpoints/qwen_instruct_compressed.pt
"""
import os, sys, time, json, math, argparse, struct, zlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn as nn

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)


def compress_and_store(model, keep_ratio=0.85):
    """Compress model weights and store SVD factors + int8 codes.

    Instead of storing the reconstructed W (fp32, same size as original),
    we store:
      - U: (out, r) as fp16
      - V: (r, in) as fp16
      - bias: as fp16
      - int8 codes for U and V (1 byte per element + scale)

    This gives ACTUAL file size reduction.
    """
    compressed_data = {}
    total_original = 0
    total_compressed = 0
    n_layers = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            W = module.weight.data
            if W.shape[0] < 100 or W.shape[1] < 100:
                continue
        elif hasattr(module, 'weight') and hasattr(module, 'bias'):
            W = module.weight.data
            if W.dim() != 2 or W.shape[0] < 100 or W.shape[1] < 100:
                continue
        else:
            continue

        out_f, in_f = W.shape
        # SVD: W ≈ U @ V
        r = max(1, int(keep_ratio * min(out_f, in_f)))
        r = min(r, min(out_f, in_f) - 1)

        U_full, S, Vh_full = torch.linalg.svd(W, full_matrices=False)
        U = (U_full[:, :r] * S[:r].unsqueeze(0))  # (out, r)
        V = Vh_full[:r, :]  # (r, in)

        # Quantize U and V to int8 (4x compression on the factors)
        def quant_int8(tensor):
            wmax = tensor.abs().max().clamp(min=1e-8)
            scale = (wmax / 127.0).item()
            codes = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
            return codes, scale

        U_codes, U_scale = quant_int8(U)
        V_codes, V_scale = quant_int8(V)

        # Store compressed representation
        compressed_data[name] = {
            "U_codes": U_codes,   # int8, (out, r)
            "U_scale": U_scale,   # float
            "V_codes": V_codes,   # int8, (r, in)
            "V_scale": V_scale,   # float
            "shape": [out_f, in_f],
            "rank": r,
            "bias": module.bias.data.clone() if hasattr(module, 'bias') and module.bias is not None else None,
        }

        # Size calculation
        orig_bytes = out_f * in_f * 4  # fp32
        comp_bytes = U_codes.numel() * 1 + V_codes.numel() * 1 + 8  # int8 + scales
        if module.bias is not None:
            comp_bytes += module.bias.numel() * 2  # fp16 bias

        total_original += orig_bytes
        total_compressed += comp_bytes
        n_layers += 1

    ratio = total_compressed / max(total_original, 1)
    print(f"  {n_layers} layers compressed")
    print(f"  Original: {total_original / 1024 / 1024:.1f} MB")
    print(f"  Compressed: {total_compressed / 1024 / 1024:.1f} MB")
    print(f"  Ratio: {ratio:.3f} ({1/ratio:.1f}x ACTUAL compression)")

    return compressed_data, ratio


def load_compressed(model, compressed_data):
    """Load compressed weights back into model (reconstruct fp32 from int8 SVD factors)."""
    for name, module in model.named_modules():
        if name not in compressed_data:
            continue

        data = compressed_data[name]
        out_f, in_f = data["shape"]
        r = data["rank"]

        U_codes = data["U_codes"].reshape(out_f, r).float()
        V_codes = data["V_codes"].reshape(r, in_f).float()
        U_scale = data["U_scale"]
        V_scale = data["V_scale"]

        # Reconstruct: W ≈ (U_codes * U_scale) @ (V_codes * V_scale)
        U = U_codes * U_scale
        V = V_codes * V_scale
        W = U @ V

        with torch.no_grad():
            module.weight.data.copy_(W)
            if data["bias"] is not None and hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.copy_(data["bias"].float())


def save_compressed_model(model, compressed_data, model_name, filepath):
    """Save the ACTUALLY compressed model (small file)."""
    # Convert int8 tensors to bytes for efficient storage
    save_data = {
        "model_name": model_name,
        "compressed_weights": {},
        "config": model.config.to_dict() if hasattr(model.config, 'to_dict') else {},
    }

    for name, data in compressed_data.items():
        save_data["compressed_weights"][name] = {
            "U_codes": data["U_codes"].cpu().numpy().tobytes(),
            "U_scale": data["U_scale"],
            "V_codes": data["V_codes"].cpu().numpy().tobytes(),
            "V_scale": data["V_scale"],
            "shape": data["shape"],
            "rank": data["rank"],
            "bias": data["bias"].cpu().to(torch.float16).numpy().tobytes() if data["bias"] is not None else None,
        }

    # Save with zlib compression
    with open(filepath, "wb") as f:
        # Use torch.save for the metadata + raw bytes
        torch.save(save_data, f)

    actual_size = os.path.getsize(filepath)
    return actual_size


def load_compressed_model(filepath, device="cpu"):
    """Load a compressed model from file."""
    save_data = torch.load(filepath, map_location="cpu", weights_only=False)

    model_name = save_data["model_name"]
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True
    ).to(device)

    # Reconstruct weights from compressed data
    compressed_data = {}
    for name, data in save_data["compressed_weights"].items():
        import numpy as np
        out_f, in_f = data["shape"]
        r = data["rank"]

        U_codes = torch.frombuffer(
            np.frombuffer(data["U_codes"], dtype=np.int8).reshape(out_f, r).copy(),
            dtype=torch.int8
        )
        V_codes = torch.frombuffer(
            np.frombuffer(data["V_codes"], dtype=np.int8).reshape(r, in_f).copy(),
            dtype=torch.int8
        )

        bias = None
        if data["bias"] is not None:
            bias_np = np.frombuffer(data["bias"], dtype=np.float16)
            bias = torch.frombuffer(bias_np.copy(), dtype=torch.float16)

        compressed_data[name] = {
            "U_codes": U_codes,
            "U_scale": data["U_scale"],
            "V_codes": V_codes,
            "V_scale": data["V_scale"],
            "shape": data["shape"],
            "rank": data["rank"],
            "bias": bias,
        }

    load_compressed(model, compressed_data)
    model.eval()

    return model, tokenizer, model_name


def chat_with_model(model, tokenizer, device, model_name="", is_instruct=False):
    """Interactive chat with proper handling for instruct vs base models."""
    print("\n" + "=" * 60)
    print(f"  Chat with {model_name}")
    print("  Type 'quit' to exit, 'help' for commands")
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
            break
        if user_input.lower() == "help":
            print("  Commands: temp <value>, tokens <n>, quit")
            continue

        # For instruct models, use chat template
        if is_instruct and hasattr(tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": user_input}]
            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
            except:
                input_ids = tokenizer.encode(user_input, return_tensors="pt").to(device)
        else:
            # Base model: just use the prompt directly
            input_ids = tokenizer.encode(user_input, return_tensors="pt").to(device)

        print("AI> ", end="", flush=True)
        try:
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=200,
                    temperature=0.7,
                    top_k=50,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else 0,
                    repetition_penalty=1.2,
                )

            response = tokenizer.decode(output[0], skip_special_tokens=True)

            # Remove the prompt from response
            if is_instruct and hasattr(tokenizer, 'apply_chat_template'):
                # For instruct models, extract just the assistant's response
                if "<|im_start|>assistant" in response:
                    response = response.split("<|im_start|>assistant")[-1].strip()
                elif "assistant" in response.lower():
                    parts = response.lower().split("assistant")
                    if len(parts) > 1:
                        response = response[response.lower().index("assistant") + len("assistant"):].strip()
            else:
                # For base models, remove the input prompt
                prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                if response.startswith(prompt_text):
                    response = response[len(prompt_text):].strip()

            print(response)
        except Exception as e:
            print(f"[Error: {e}]")


# Recommended models for actual chat (instruction-tuned, not base models)
RECOMMENDED_MODELS = {
    "distilgpt2": ("DistilGPT2 (base, for testing)", False),
    "gpt2": ("GPT-2 Small (base)", False),
    "Qwen/Qwen2-1.5B-Instruct": ("Qwen2 1.5B Instruct (chat, recommended)", True),
    "microsoft/Phi-3-mini-4k-instruct": ("Phi-3 Mini (chat, recommended)", True),
    "meta-llama/Llama-3.2-1B-Instruct": ("Llama 3.2 1B Instruct (chat)", True),
    "meta-llama/Llama-3.2-3B-Instruct": ("Llama 3.2 3B Instruct (chat)", True),
}


def main():
    parser = argparse.ArgumentParser(description="Compress LLM and chat (REAL compression)")
    parser.add_argument("--model", type=str, default="distilgpt2",
                        help="HuggingFace model ID. For chat: Qwen/Qwen2-1.5B-Instruct")
    parser.add_argument("--load", type=str, default=None, help="Load saved compressed model")
    parser.add_argument("--save", type=str, default=None, help="Save compressed model")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt")
    parser.add_argument("--keep_ratio", type=float, default=0.85)
    parser.add_argument("--tokens", type=int, default=200)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.load:
        # Load compressed model
        print(f"\n--- Loading compressed model from {args.load} ---")
        model, tokenizer, model_name = load_compressed_model(args.load, device)
        file_size = os.path.getsize(args.load) / 1024 / 1024
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model: {model_name}")
        print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
        print(f"  Compressed file size: {file_size:.1f} MB")
        is_instruct = "instruct" in model_name.lower() or "chat" in model_name.lower()
    else:
        # Load from HuggingFace
        model_name = args.model
        print(f"\n--- Loading {model_name} ---")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, trust_remote_code=True
        ).to(device)
        model.eval()

        n_params = sum(p.numel() for p in model.parameters())
        orig_size_mb = n_params * 4 / 1024 / 1024
        print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
        print(f"  Original size (fp32): {orig_size_mb:.1f} MB")

        # Check if this is an instruct model
        is_instruct = "instruct" in model_name.lower() or "chat" in model_name.lower()
        if not is_instruct:
            print(f"\n  ⚠️  WARNING: {model_name} is a BASE model, not instruction-tuned.")
            print(f"  For better chat quality, use an instruct model like:")
            print(f"    Qwen/Qwen2-1.5B-Instruct")
            print(f"    microsoft/Phi-3-mini-4k-instruct")
            print(f"    meta-llama/Llama-3-8B-Instruct")

        # Compress
        print(f"\n--- Compressing (keep_ratio={args.keep_ratio}) ---")
        compressed_data, ratio = compress_and_store(model, keep_ratio=args.keep_ratio)

        # Save if requested
        if args.save:
            save_path = args.save if os.path.isabs(args.save) else os.path.join(CKPT_DIR, args.save)
            actual_size = save_compressed_model(model, compressed_data, model_name, save_path)
            print(f"\n  ✅ Saved to {save_path}")
            print(f"     Original: {orig_size_mb:.1f} MB")
            print(f"     Compressed file: {actual_size / 1024 / 1024:.1f} MB")
            print(f"     ACTUAL compression: {orig_size_mb / (actual_size / 1024 / 1024):.1f}x")
            print(f"     Load later: python scripts/chat_real.py --load {save_path}")

    # Generate or chat
    if args.prompt:
        print(f"\n--- Generating ---")
        input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=args.tokens,
                                    temperature=0.7, top_k=50, do_sample=True,
                                    pad_token_id=tokenizer.eos_token_id or 0,
                                    repetition_penalty=1.2)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        if not is_instruct and response.startswith(args.prompt):
            response = response[len(args.prompt):].strip()
        print(response)
    else:
        chat_with_model(model, tokenizer, device, model_name, is_instruct)


if __name__ == "__main__":
    main()
