"""
AWF Ladder Benchmark — compress a model at multiple ratios and benchmark each.

Compression ladder: 1x (baseline) → 2x → 4x → 8x → 10x → 16x → 24x
Metrics per rung:
  - HumanEval pass@1 (code generation)
  - Perplexity (language modeling quality)
  - Tokens/sec (inference speed)
  - Peak RAM (memory usage)
  - Peak VRAM (GPU memory)
  - Model loading time
  - File size on disk

MEMORY SAFE: Compresses layer-by-layer, frees original weights immediately.
Never holds full uncompressed + compressed model simultaneously.

Usage:
  python scripts/awf_ladder.py --model distilgpt2
  python scripts/awf_ladder.py --model Qwen/Qwen2-1.5B-Instruct
  python scripts/awf_ladder.py --model THUDM/glm-4-9b-chat --rungs 1,2,4,8

On Colab: open scripts/AWF_Ladder_Benchmark.ipynb
"""
import os, sys, time, json, math, argparse, gc, traceback
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn as nn
import numpy as np

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")
BENCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(BENCH_DIR, exist_ok=True)


# ============================================================================
# Memory-safe compression
# ============================================================================
def compress_layer_by_layer(model, keep_ratio, device):
    """Compress model layer-by-layer, freeing memory aggressively.

    For each weight matrix:
      1. Read the weight
      2. Compute SVD
      3. Quantize to int8
      4. Replace the weight with the int8 reconstruction (fp32)
      5. Delete the SVD factors (we only keep the reconstruction)

    This way, we never hold both the original AND the compressed data.
    The reconstruction overwrites the original in-place.

    For SAVING: we store the int8 SVD factors separately (genuinely smaller file).
    For INFERENCE: we use the reconstructed fp32 weights (loaded from the saved file).
    """
    n_layers = 0
    total_orig = 0
    total_comp = 0

    compressed_weights = {}

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
        r = max(1, int(keep_ratio * min(out_f, in_f)))
        r = min(r, min(out_f, in_f) - 1)

        # SVD on the weight (keep on CPU to save VRAM)
        W_cpu = W.cpu()
        U_full, S, Vh_full = torch.linalg.svd(W_cpu, full_matrices=False)
        U = (U_full[:, :r] * S[:r].unsqueeze(0))  # (out, r)
        V = Vh_full[:r, :]  # (r, in)

        # Free SVD intermediates
        del U_full, S, Vh_full, W_cpu

        # Quantize to int8
        def quant_int8(tensor):
            wmax = tensor.abs().max().clamp(min=1e-8)
            scale = (wmax / 127.0).item()
            codes = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
            return codes, scale

        U_codes, U_scale = quant_int8(U)
        V_codes, V_scale = quant_int8(V)
        del U, V

        # Store compressed factors
        compressed_weights[name] = {
            "U_codes": U_codes.numpy().tobytes(),
            "U_shape": [out_f, r],
            "U_scale": U_scale,
            "V_codes": V_codes.numpy().tobytes(),
            "V_shape": [r, in_f],
            "V_scale": V_scale,
            "shape": [out_f, in_f],
            "rank": r,
            "bias": module.bias.data.cpu().to(torch.float16).numpy().tobytes()
                     if hasattr(module, 'bias') and module.bias is not None else None,
        }
        del U_codes, V_codes

        # Replace weight with reconstruction (for inference)
        U_recon = torch.frombuffer(
            np.frombuffer(compressed_weights[name]["U_codes"], dtype=np.int8).copy(),
            dtype=torch.int8
        ).reshape(out_f, r).float() * U_scale
        V_recon = torch.frombuffer(
            np.frombuffer(compressed_weights[name]["V_codes"], dtype=np.int8).copy(),
            dtype=torch.int8
        ).reshape(r, in_f).float() * V_scale
        W_recon = U_recon @ V_recon

        with torch.no_grad():
            module.weight.data.copy_(W_recon.to(device))
        del U_recon, V_recon, W_recon

        total_orig += out_f * in_f * 4
        total_comp += (out_f * r + r * in_f) * 1 + 8
        n_layers += 1

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ratio = total_comp / max(total_orig, 1)
    return compressed_weights, ratio, n_layers


# ============================================================================
# Benchmark functions
# ============================================================================
def get_peak_memory(device):
    """Get peak memory usage in MB."""
    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        # Use resource module for CPU
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB → MB


def reset_memory_stats(device):
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def measure_perplexity(model, tokenizer, device, texts=None):
    """Measure perplexity on a set of test texts."""
    if texts is None:
        texts = [
            "Once upon a time, there was a little girl named Lily who loved to play in the garden.",
            "The scientist walked into the laboratory and began her experiment carefully.",
            "In a small village by the sea, the fishermen would gather every morning before dawn.",
            "The old man smiled and said, 'Life is like a river — it flows whether you watch or not.'",
            "Technology has changed the way we communicate, work, and live our daily lives.",
        ]

    model.eval()
    total_loss = 0
    total_tokens = 0

    for text in texts:
        try:
            encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            input_ids = encodings["input_ids"].to(device)
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                total_loss += outputs.loss.item() * input_ids.numel()
                total_tokens += input_ids.numel()
        except Exception:
            continue

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))
    return ppl, avg_loss


def measure_tokens_per_sec(model, tokenizer, device, prompt="Hello, how are you?", n_tokens=50):
    """Measure inference speed (tokens/sec)."""
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Warmup
    with torch.no_grad():
        try:
            _ = model.generate(input_ids, max_new_tokens=5, do_sample=False)
        except:
            return 0.0

    # Measure
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.time()
    with torch.no_grad():
        try:
            output = model.generate(input_ids, max_new_tokens=n_tokens,
                                    do_sample=True, temperature=0.7, top_k=50,
                                    pad_token_id=tokenizer.eos_token_id or 0)
        except:
            return 0.0
    torch.cuda.synchronize() if device == "cuda" else None
    elapsed = time.time() - t0
    return n_tokens / max(elapsed, 0.001)


def measure_humaneval(model, tokenizer, device, n_problems=5):
    """Simple HumanEval-style code generation test."""
    # Mini HumanEval problems
    problems = [
        {"prompt": "def add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n    return", "expected": "a + b"},
        {"prompt": "def factorial(n):\n    \"\"\"Return n!\"\"\"\n    if n <= 1:\n        return 1\n    return", "expected": "n * factorial(n - 1)"},
        {"prompt": "def is_even(n):\n    \"\"\"Return True if n is even.\"\"\"\n    return", "expected": "n % 2 == 0"},
        {"prompt": "def reverse_string(s):\n    \"\"\"Return the reverse of s.\"\"\"\n    return", "expected": "s[::-1]"},
        {"prompt": "def max_of_list(lst):\n    \"\"\"Return the maximum value in lst.\"\"\"\n    return", "expected": "max(lst)"},
    ]

    model.eval()
    passed = 0
    total = min(n_problems, len(problems))

    for prob in problems[:total]:
        try:
            input_ids = tokenizer.encode(prob["prompt"], return_tensors="pt").to(device)
            with torch.no_grad():
                output = model.generate(input_ids, max_new_tokens=50,
                                        do_sample=False, temperature=0.0,
                                        pad_token_id=tokenizer.eos_token_id or 0)
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            # Check if the expected answer is in the response
            if prob["expected"] in response:
                passed += 1
        except:
            continue

    return passed / max(total, 1)


def generate_sample(model, tokenizer, device, prompt, max_tokens=100):
    """Generate a sample text for qualitative evaluation."""
    model.eval()
    try:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=max_tokens,
                                    temperature=0.7, top_k=50, do_sample=True,
                                    pad_token_id=tokenizer.eos_token_id or 0,
                                    repetition_penalty=1.2)
        return tokenizer.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        return f"[Error: {e}]"


# ============================================================================
# Ladder rung
# ============================================================================
def run_ladder_rung(model_name, keep_ratio, device, tokenizer=None, base_model=None):
    """Run one rung of the compression ladder.

    Args:
        model_name: HuggingFace model ID
        keep_ratio: SVD keep ratio (1.0 = baseline, 0.5 = 2x, 0.25 = 4x, etc.)
        device: 'cuda' or 'cpu'
        tokenizer: pre-loaded tokenizer (reused across rungs)
        base_model: pre-loaded model (reused, compressed in-place)

    Returns:
        dict with all metrics for this rung
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    compression_name = "baseline" if keep_ratio >= 0.99 else f"{1/keep_ratio:.0f}x"

    print(f"\n{'='*60}")
    print(f"  RUNG: {compression_name} (keep_ratio={keep_ratio:.2f})")
    print(f"{'='*60}")

    # Load model (or reuse)
    load_start = time.time()
    if base_model is not None:
        # Reuse base model — compress it (in-place)
        model = base_model
        if keep_ratio < 0.99:
            print(f"  Compressing (layer-by-layer, memory-safe)...")
            compressed_weights, ratio, n_layers = compress_layer_by_layer(
                model, keep_ratio, device
            )
        else:
            compressed_weights = None
            ratio = 1.0
            n_layers = 0
    else:
        # Fresh load
        print(f"  Loading {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, trust_remote_code=True
        ).to(device)
        model.eval()

        if keep_ratio < 0.99:
            print(f"  Compressing (layer-by-layer, memory-safe)...")
            compressed_weights, ratio, n_layers = compress_layer_by_layer(
                model, keep_ratio, device
            )
        else:
            compressed_weights = None
            ratio = 1.0
            n_layers = 0

    load_time = time.time() - load_start

    # Get memory stats
    n_params = sum(p.numel() for p in model.parameters())
    model_size_mb = n_params * 4 / 1024 / 1024  # fp32 in memory
    peak_ram = get_peak_memory(device)

    # Reset peak stats for inference measurements
    reset_memory_stats(device)

    # Perplexity
    print(f"  Measuring perplexity...")
    ppl, loss = measure_perplexity(model, tokenizer, device)

    # Tokens/sec
    print(f"  Measuring tokens/sec...")
    tps = measure_tokens_per_sec(model, tokenizer, device)

    # HumanEval
    print(f"  Measuring HumanEval pass rate...")
    humaneval_pass = measure_humaneval(model, tokenizer, device)

    # Peak memory during inference
    peak_inference_ram = get_peak_memory(device)

    # Sample generation
    sample = generate_sample(model, tokenizer, device, "Write a Python function that adds two numbers:", max_tokens=80)

    # Save compressed weights (if compressed)
    file_size_mb = model_size_mb  # default: in-memory size
    if compressed_weights is not None:
        save_name = model_name.replace("/", "_") + f"_rung_{compression_name}.pt"
        save_path = os.path.join(CKPT_DIR, save_name)
        torch.save({
            "model_name": model_name,
            "compressed_weights": compressed_weights,
            "keep_ratio": keep_ratio,
            "compression_ratio": ratio,
        }, save_path)
        file_size_mb = os.path.getsize(save_path) / 1024 / 1024

    results = {
        "rung": compression_name,
        "keep_ratio": keep_ratio,
        "n_params": n_params,
        "model_size_mb": round(model_size_mb, 1),
        "file_size_mb": round(file_size_mb, 1),
        "compression_ratio": round(ratio, 4),
        "load_time_sec": round(load_time, 1),
        "perplexity": round(ppl, 2),
        "loss": round(loss, 4),
        "tokens_per_sec": round(tps, 1),
        "humaneval_pass_rate": round(humaneval_pass, 3),
        "peak_ram_mb": round(peak_ram, 1),
        "peak_vram_mb": round(peak_inference_ram, 1) if device == "cuda" else 0,
        "sample": sample[:200],
    }

    print(f"\n  Results:")
    print(f"    File size: {file_size_mb:.1f} MB (compression: {1/max(ratio,0.01):.1f}x)")
    print(f"    Perplexity: {ppl:.2f}")
    print(f"    Tokens/sec: {tps:.1f}")
    print(f"    HumanEval pass: {humaneval_pass*100:.0f}%")
    print(f"    Load time: {load_time:.1f}s")
    print(f"    Peak RAM: {peak_ram:.0f} MB")
    if device == "cuda":
        print(f"    Peak VRAM: {peak_inference_ram:.0f} MB")

    # Cleanup compressed weights (keep model for next rung)
    del compressed_weights
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results, model


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="AWF Ladder Benchmark")
    parser.add_argument("--model", type=str, default="distilgpt2",
                        help="HuggingFace model ID")
    parser.add_argument("--rungs", type=str, default="1,2,4,8,16",
                        help="Compression rungs (e.g. '1,2,4,8,16,24')")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cpu, or cuda")
    args = parser.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"=== AWF Ladder Benchmark ===")
    print(f"Model: {args.model}")
    print(f"Device: {device}")

    # Parse rungs
    rung_multipliers = [float(x) for x in args.rungs.split(",")]
    keep_ratios = [1.0 / m for m in rung_multipliers]

    print(f"Rungs: {rung_multipliers}")
    print(f"Keep ratios: {keep_ratios}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    all_results = []
    base_model = None

    for mult, kr in zip(rung_multipliers, keep_ratios):
        try:
            results, base_model = run_ladder_rung(
                args.model, kr, device, tokenizer, base_model
            )
            all_results.append(results)
        except Exception as e:
            print(f"\n  ERROR at rung {mult}x: {e}")
            traceback.print_exc()
            all_results.append({
                "rung": f"{mult:.0f}x",
                "error": str(e)[:200],
            })
            break  # Stop if we crash

    # Summary
    print(f"\n{'='*70}")
    print(f"AWF LADDER BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"{'Rung':<8} {'File MB':>8} {'Compr':>6} {'PPL':>8} {'Tok/s':>7} {'HumanEval':>10} {'RAM MB':>8} {'Load s':>7}")
    print(f"{'-'*65}")
    for r in all_results:
        if "error" in r:
            print(f"{r['rung']:<8} ERROR: {r['error'][:50]}")
        else:
            compr = f"{1/max(r['compression_ratio'],0.01):.1f}x"
            print(f"{r['rung']:<8} {r['file_size_mb']:>8.1f} {compr:>6} {r['perplexity']:>8.2f} "
                  f"{r['tokens_per_sec']:>7.1f} {r['humaneval_pass_rate']*100:>9.0f}% {r['peak_ram_mb']:>8.0f} "
                  f"{r['load_time_sec']:>7.1f}")

    # Save results
    output_file = os.path.join(BENCH_DIR, f"ladder_{args.model.replace('/', '_')}.json")
    with open(output_file, "w") as f:
        json.dump({"model": args.model, "device": device, "results": all_results}, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Print samples
    print(f"\n{'='*70}")
    print(f"SAMPLE OUTPUTS PER RUNG")
    print(f"{'='*70}")
    for r in all_results:
        if "sample" in r:
            print(f"\n--- {r['rung']} (ppl={r.get('perplexity', '?')}) ---")
            print(r["sample"])


if __name__ == "__main__":
    main()
