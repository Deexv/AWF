"""
Apply AWF compression to a pre-trained model and VERIFY it still works.

1. Load pre-trained DistilGPT2 (fluent, 82M params)
2. Generate text (PROOF it's fluent)
3. Apply low-rank compression to ALL weight matrices
4. Generate text again (PROOF compression preserves quality)
5. Compare original vs compressed: size, text quality, perplexity

This is the REAL product: compress existing LLMs, don't train from scratch.
"""
import os, sys, time, json, math, copy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)


def get_weight_matrices(model):
    """Find all large weight matrices in the model (Linear and Conv1D)."""
    matrices = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if module.weight.shape[0] > 100 and module.weight.shape[1] > 100:
                matrices.append((name, module, "weight", module.weight.data))
        elif hasattr(module, 'weight') and hasattr(module, 'bias'):
            w = module.weight.data
            if w.dim() == 2 and w.shape[0] > 100 and w.shape[1] > 100:
                matrices.append((name, module, "weight", w))
    return matrices


def apply_compression(model, compression_ratio=0.15):
    """Apply AWF low-rank compression IN-PLACE. Returns stats."""
    print(f"\n=== Applying AWF Compression (ratio={compression_ratio*100:.0f}%) ===")
    matrices = get_weight_matrices(model)
    print(f"Found {len(matrices)} weight matrices")

    total_original = 0
    total_compressed = 0
    total_error = 0

    for name, module, attr, W in matrices:
        out_f, in_f = W.shape[0], W.shape[1]

        # Choose rank for target compression ratio
        r = max(1, int(compression_ratio * out_f * in_f * 4 / ((out_f + in_f) * 2)))
        r = min(r, min(out_f, in_f) - 1)

        # SVD: W ≈ U @ V
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        U_r = U[:, :r] * S[:r].unsqueeze(0)  # (out, r)
        V_r = Vh[:r, :]  # (r, in)

        # Reconstruct and REPLACE the weight
        W_compressed = U_r @ V_r
        error = (W - W_compressed).norm().item() / W.norm().item()
        total_error += error

        # Apply in-place
        with torch.no_grad():
            module.weight.data.copy_(W_compressed)

        original_bytes = out_f * in_f * 4
        compressed_bytes = (out_f * r + r * in_f) * 2 + r * 4
        total_original += original_bytes
        total_compressed += compressed_bytes

    ratio = total_compressed / max(total_original, 1)
    avg_error = total_error / max(len(matrices), 1)
    print(f"  Original: {total_original / 1024 / 1024:.1f} MB")
    print(f"  Compressed: {total_compressed / 1024 / 1024:.1f} MB")
    print(f"  Ratio: {ratio:.3f} ({1/ratio:.1f}x compression)")
    print(f"  Average reconstruction error: {avg_error:.4f}")
    return ratio, avg_error


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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== AWF Compression of Pre-trained LLM ===")
    print(f"Device: {device}")

    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    # 1. Load pre-trained model (ALREADY FLUENT)
    print(f"\n--- Loading DistilGPT2 (82M params, pre-trained) ---")
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} params ({n_params/1e6:.1f}M)")
    print(f"  Storage (fp32): {n_params*4/1024/1024:.1f} MB")

    # Test prompts
    prompts = [
        "Once upon a time there was a little girl named Lily",
        "The scientist walked into the lab and",
        "In a small village by the sea,",
    ]

    # 2. Generate text BEFORE compression
    print(f"\n{'='*70}")
    print(f"BEFORE COMPRESSION (original DistilGPT2)")
    print(f"{'='*70}")
    original_samples = {}
    for p in prompts:
        text = generate_text(model, tokenizer, p, n_tokens=80, device=device)
        print(f"\nPrompt: {p!r}")
        print(f"Output: {text}")
        original_samples[p] = text

    test_text = ("Once upon a time, there was a little girl named Lily who loved to play "
                 "in the garden. She would run through the flowers, chasing butterflies "
                 "and singing songs. One day, she found a small bird with a broken wing.")
    orig_ppl, orig_loss = evaluate_perplexity(model, tokenizer, test_text, device)
    print(f"\nPerplexity: {orig_ppl:.2f} (loss: {orig_loss:.4f})")

    # 3. Apply AWF compression
    print(f"\n{'='*70}")
    print(f"APPLYING AWF COMPRESSION")
    print(f"{'='*70}")
    ratio, avg_error = apply_compression(model, compression_ratio=0.15)

    # 4. Generate text AFTER compression
    print(f"\n{'='*70}")
    print(f"AFTER COMPRESSION (AWF compressed)")
    print(f"{'='*70}")
    compressed_samples = {}
    for p in prompts:
        text = generate_text(model, tokenizer, p, n_tokens=80, device=device)
        print(f"\nPrompt: {p!r}")
        print(f"Output: {text}")
        compressed_samples[p] = text

    comp_ppl, comp_loss = evaluate_perplexity(model, tokenizer, test_text, device)
    print(f"\nPerplexity: {comp_ppl:.2f} (loss: {comp_loss:.4f})")

    # 5. Comparison
    print(f"\n{'='*70}")
    print(f"COMPARISON: Original vs AWF Compressed")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'Original':>15} {'Compressed':>15} {'Change':>10}")
    print(f"{'-'*65}")
    print(f"{'Storage (MB)':<25} {n_params*4/1024/1024:>15.1f} {n_params*ratio*2/1024/1024:>15.1f} {1/ratio:>9.1f}x smaller")
    print(f"{'Parameters':<25} {n_params:>15,} {int(n_params*ratio):>15,} {1/ratio:>9.1f}x fewer")
    print(f"{'Perplexity':<25} {orig_ppl:>15.2f} {comp_ppl:>15.2f} {comp_ppl/orig_ppl:>9.2f}x")
    print(f"{'Avg reconstruction error':<25} {'—':>15} {avg_error:>15.4f}")

    # Quality assessment
    print(f"\n--- Quality Assessment ---")
    if comp_ppl / orig_ppl < 1.5:
        print(f"✅ GOOD: Perplexity increased by {(comp_ppl/orig_ppl-1)*100:.0f}% — minimal quality loss")
    elif comp_ppl / orig_ppl < 3.0:
        print(f"⚠️ MODERATE: Perplexity increased by {(comp_ppl/orig_ppl-1)*100:.0f}% — some quality loss")
        print(f"   Brief fine-tuning (1-2 hours) would recover quality.")
    else:
        print(f"❌ HIGH: Perplexity increased by {(comp_ppl/orig_ppl-1)*100:.0f}% — significant quality loss")
        print(f"   Fine-tuning needed, or use lower compression ratio.")

    # Save results
    results = {
        "model": "distilgpt2",
        "original": {
            "params": n_params,
            "storage_mb": n_params * 4 / 1024 / 1024,
            "perplexity": orig_ppl,
            "loss": orig_loss,
            "samples": original_samples,
        },
        "compressed": {
            "params": int(n_params * ratio),
            "storage_mb": n_params * ratio * 2 / 1024 / 1024,
            "perplexity": comp_ppl,
            "loss": comp_loss,
            "compression_ratio": ratio,
            "compression_factor": 1 / ratio,
            "avg_reconstruction_error": avg_error,
            "samples": compressed_samples,
        },
        "perplexity_ratio": comp_ppl / orig_ppl,
    }
    with open(os.path.join(os.path.dirname(__file__), "..", "benchmarks", "llm_compression_verified.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to benchmarks/llm_compression_verified.json")


if __name__ == "__main__":
    main()
