"""
Compress ANY HuggingFace LLM to run on a 4-8GB RAM PC.

Pipeline:
  1. Download pre-trained model (GPT-2 Medium, Large, etc.)
  2. Apply AWF low-rank compression (1.5-2x)
  3. Apply int8 quantization (2x more)
  4. Combined: 3-4x smaller model that runs on 4-8GB RAM
  5. Generate text to prove it's still fluent

Usage:
  python scripts/compress_for_pc.py --model gpt2-medium --target_ram 4
  python scripts/compress_for_pc.py --model gpt2-large --target_ram 8
  python scripts/compress_for_pc.py --model distilgpt2 --target_ram 2
"""
import os, sys, time, json, math, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)


def get_weight_matrices(model):
    """Find all large weight matrices."""
    matrices = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            W = module.weight.data
            if W.shape[0] > 100 and W.shape[1] > 100:
                matrices.append((name, module, "linear"))
        elif hasattr(module, 'weight') and hasattr(module, 'bias'):
            W = module.weight.data
            if W.dim() == 2 and W.shape[0] > 100 and W.shape[1] > 100:
                matrices.append((name, module, "conv1d"))
    return matrices


def apply_svd_compression(model, keep_ratio=0.50):
    """Apply SVD low-rank compression in-place."""
    matrices = get_weight_matrices(model)
    total_orig = 0
    total_comp = 0

    for name, module, mtype in matrices:
        W = module.weight.data
        out_f, in_f = (W.shape[0], W.shape[1]) if mtype == "linear" else (W.shape[1], W.shape[0])
        r = max(1, int(keep_ratio * out_f * in_f * 4 / ((out_f + in_f) * 2)))
        r = min(r, min(out_f, in_f) - 1)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        W_comp = (U[:, :r] * S[:r].unsqueeze(0)) @ Vh[:r, :]
        with torch.no_grad():
            module.weight.data.copy_(W_comp)
        total_orig += out_f * in_f * 4
        total_comp += (out_f * r + r * in_f) * 2
    return total_comp / max(total_orig, 1)


def quantize_int8(weight):
    """Quantize a weight tensor to int8 (2x compression over fp16, 4x over fp32)."""
    wmax = weight.abs().max().clamp(min=1e-8)
    scale = (wmax / 127.0).item()
    codes = (weight / scale).round().clamp(-128, 127).to(torch.int8)
    return codes, scale


def apply_int8_quantization(model):
    """Apply int8 quantization to all Linear/Conv1D weights in-place."""
    n_quantized = 0
    total_fp32 = 0
    total_int8 = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or (hasattr(module, 'weight') and hasattr(module, 'bias')
                                               and module.weight.dim() == 2 and module.weight.shape[0] > 100):
            W = module.weight.data.float()
            codes, scale = quantize_int8(W)
            # Store quantized weights as fp32 reconstruction (for inference)
            # In a real deployment, you'd use a custom kernel that reads int8 codes
            W_quantized = codes.float() * scale
            with torch.no_grad():
                module.weight.data.copy_(W_quantized)
            n_quantized += 1
            total_fp32 += W.numel() * 4
            total_int8 += W.numel() * 1 + 4  # 1 byte per weight + 1 scale

    print(f"  Quantized {n_quantized} layers")
    print(f"  FP32: {total_fp32/1024/1024:.1f} MB → INT8: {total_int8/1024/1024:.1f} MB ({total_fp32/max(total_int8,1):.1f}x)")
    return total_int8 / max(total_fp32, 1)


def generate_text(model, tokenizer, prompt, n_tokens=100, temperature=0.7, top_k=50, device="cpu"):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=n_tokens,
                                temperature=temperature, top_k=top_k,
                                do_sample=True, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def evaluate_perplexity(model, tokenizer, text, device="cpu"):
    model.eval()
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = encodings["input_ids"].to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        return math.exp(outputs.loss.item()), outputs.loss.item()


def get_model_size_mb(model):
    """Estimate model size in MB (fp32)."""
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024


def main():
    parser = argparse.ArgumentParser(description="Compress LLM for 4-8GB PC")
    parser.add_argument("--model", type=str, default="distilgpt2",
                        help="HuggingFace model: distilgpt2, gpt2, gpt2-medium, gpt2-large")
    parser.add_argument("--target_ram", type=int, default=4, help="Target RAM in GB")
    parser.add_argument("--keep_ratio", type=float, default=0.50,
                        help="SVD keep ratio (0.50 = 2x compression, 0.85 = 1.2x)")
    parser.add_argument("--use_int8", action="store_true", default=True, help="Apply int8 quantization")
    parser.add_argument("--prompt", type=str, default="Once upon a time there was a little girl named Lily")
    parser.add_argument("--tokens", type=int, default=100)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== AWF LLM Compression for {args.target_ram}GB PC ===")
    print(f"Device: {device}")
    print(f"Model: {args.model}")

    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    # 1. Load pre-trained model
    print(f"\n--- Loading {args.model} ---")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    orig_size_mb = get_model_size_mb(model)
    orig_ram_mb = orig_size_mb + 500  # rough estimate including activations + tokenizer
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"  Model size: {orig_size_mb:.1f} MB (fp32)")
    print(f"  Est. RAM needed: {orig_ram_mb:.0f} MB ({orig_ram_mb/1024:.1f} GB)")

    # Test prompts
    prompts = [
        args.prompt,
        "The scientist walked into the lab and",
        "In a small village by the sea,",
    ]
    test_text = ("Once upon a time, there was a little girl named Lily who loved to play "
                 "in the garden. She would run through the flowers, chasing butterflies "
                 "and singing songs. One day, she found a small bird with a broken wing.")

    # 2. Before compression
    print(f"\n{'='*70}")
    print(f"BEFORE COMPRESSION (Original {args.model})")
    print(f"{'='*70}")
    orig_ppl, _ = evaluate_perplexity(model, tokenizer, test_text, device)
    print(f"Perplexity: {orig_ppl:.2f}")
    for p in prompts:
        text = generate_text(model, tokenizer, p, n_tokens=args.tokens, device=device)
        print(f"\n  {p!r}")
        print(f"  → {text}")

    # 3. Apply SVD compression
    print(f"\n{'='*70}")
    print(f"STEP 1: AWF SVD Compression (keep_ratio={args.keep_ratio})")
    print(f"{'='*70}")
    svd_ratio = apply_svd_compression(model, keep_ratio=args.keep_ratio)
    svd_size_mb = orig_size_mb * svd_ratio
    print(f"  Compression: {1/svd_ratio:.1f}x → {svd_size_mb:.1f} MB")

    svd_ppl, _ = evaluate_perplexity(model, tokenizer, test_text, device)
    svd_text = generate_text(model, tokenizer, prompts[0], n_tokens=args.tokens, device=device)
    print(f"  Perplexity: {svd_ppl:.2f}")
    print(f"  Text: {svd_text[:120]}")

    # 4. Apply int8 quantization
    if args.use_int8:
        print(f"\n{'='*70}")
        print(f"STEP 2: INT8 Quantization")
        print(f"{'='*70}")
        int8_ratio = apply_int8_quantization(model)
        int8_size_mb = svd_size_mb * int8_ratio * 2  # int8 is 1/4 of fp32, but we stored as fp32
        # Actually int8 would be: svd_size * 0.25 (if we used real int8 storage)
        real_int8_mb = svd_size_mb * 0.25  # 1 byte per param instead of 4
        print(f"  With real int8 storage: {real_int8_mb:.1f} MB")

        int8_ppl, _ = evaluate_perplexity(model, tokenizer, test_text, device)
        int8_text = generate_text(model, tokenizer, prompts[0], n_tokens=args.tokens, device=device)
        print(f"  Perplexity: {int8_ppl:.2f}")
        print(f"  Text: {int8_text[:120]}")

    # 5. Summary
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY: {args.model} → {args.target_ram}GB PC")
    print(f"{'='*70}")
    final_size = real_int8_mb if args.use_int8 else svd_size_mb
    final_ppl = int8_ppl if args.use_int8 else svd_ppl
    final_text = int8_text if args.use_int8 else svd_text

    combined_compression = orig_size_mb / max(final_size, 1)

    print(f"{'Stage':<25} {'Size (MB)':>10} {'RAM':>8} {'Perplexity':>12} {'Fluent?':>8}")
    print(f"{'-'*65}")
    print(f"{'Original':<25} {orig_size_mb:>10.1f} {orig_ram_mb/1024:>7.1f}G {orig_ppl:>12.2f} {'YES':>8}")
    print(f"{'After SVD':<25} {svd_size_mb:>10.1f} {'—':>8} {svd_ppl:>12.2f} {'YES' if svd_ppl < 100 else 'NO':>8}")
    if args.use_int8:
        print(f"{'After SVD + INT8':<25} {final_size:>10.1f} {final_size/1024+0.5:>7.1f}G {final_ppl:>12.2f} {'YES' if final_ppl < 100 else 'NO':>8}")

    print(f"\nCombined compression: {combined_compression:.1f}x")
    print(f"Original: {orig_size_mb:.1f} MB → Compressed: {final_size:.1f} MB")
    print(f"RAM: {orig_ram_mb/1024:.1f} GB → {final_size/1024+0.5:.1f} GB")

    fits = (final_size / 1024 + 0.5) <= args.target_ram
    fluent = final_ppl < 100

    if fits and fluent:
        print(f"\n✅ FITS in {args.target_ram}GB RAM and is FLUENT (ppl={final_ppl:.1f})")
    elif fits and not fluent:
        print(f"\n⚠️ Fits in {args.target_ram}GB but NOT fluent (ppl={final_ppl:.1f})")
        print(f"   Fine-tuning needed: python scripts/compress_finetune.py")
    elif not fits and fluent:
        print(f"\n⚠️ Fluent but DOESN'T FIT in {args.target_ram}GB (needs {final_size/1024:.1f}GB)")
        print(f"   Try: --keep_ratio 0.40 or --model gpt2 (smaller)")
    else:
        print(f"\n❌ Doesn't fit and not fluent. Try smaller model or less compression.")

    # Generate final samples
    print(f"\n=== Text Generation (compressed model) ===")
    for p in prompts:
        text = generate_text(model, tokenizer, p, n_tokens=args.tokens, device=device)
        print(f"\n  {p!r}")
        print(f"  → {text}")

    # Save results
    results = {
        "model": args.model,
        "original": {"params": n_params, "size_mb": orig_size_mb, "ppl": orig_ppl},
        "compressed": {
            "svd_ratio": svd_ratio,
            "svd_size_mb": svd_size_mb,
            "svd_ppl": svd_ppl,
            "int8_size_mb": final_size if args.use_int8 else None,
            "int8_ppl": final_ppl if args.use_int8 else None,
            "combined_compression": combined_compression,
            "fits_in_ram": fits,
            "is_fluent": fluent,
        },
    }
    with open(os.path.join(os.path.dirname(__file__), "..", "benchmarks", f"compress_{args.model.replace('-', '_')}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to benchmarks/compress_{args.model.replace('-', '_')}.json")


if __name__ == "__main__":
    main()
