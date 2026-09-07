"""
Compression Level Comparison Test
=================================

Runs a controlled comparison of multiple GGUF quantization levels
against the SAME set of prompts, collecting:

  - File size
  - Load time
  - Perplexity (on a fixed evaluation corpus)
  - Latency per response
  - Throughput (tokens/sec)
  - Sample responses (for human inspection)

The output is a single JSON file with all results, suitable for
including in the GitHub repo as proof.

Usage:
    python compare_quant_levels.py
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

from llama_cpp import Llama

# ---------- Configuration ----------

# Fixed evaluation prompts — same for every model, so we can compare
# apples-to-apples across quantization levels.
EVAL_PROMPTS = [
    # 1. Factual Q&A
    "What is the capital of France?",
    # 2. Math
    "If a shirt costs $25 and there's a 20% discount, what's the final price? Show your work.",
    # 3. Code generation
    "Write a Python function that returns the nth Fibonacci number. Just the code.",
    # 4. Summarization
    "Summarize the plot of Romeo and Juliet in two sentences.",
    # 5. Creative writing
    "Write a four-line poem about the ocean.",
    # 6. Reasoning
    "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain.",
    # 7. Translation
    "Translate 'Good morning, how are you?' into French and Spanish.",
    # 8. List generation
    "List three benefits of regular exercise.",
    # 9. Definition
    "Define photosynthesis in one sentence.",
    # 10. Comparison
    "What's the difference between a virus and a bacterium?",
]

# Fixed corpus for perplexity estimation (a few paragraphs of varied text).
# We compute pseudo-perplexity: average negative log-likelihood per token.
PERPLEXITY_CORPUS = [
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.",
    "In machine learning, gradient descent is an iterative optimization algorithm used to minimize a loss function. The model's parameters are updated in the direction of steepest descent of the loss surface.",
    "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll. The process converts carbon dioxide and water into glucose and oxygen.",
    "The capital of France is Paris, which sits on the Seine river. The Eiffel Tower, built in 1889, is one of the city's most famous landmarks and attracts millions of visitors every year.",
    "To make a simple omelette, crack two eggs into a bowl and beat them with a fork until the yolks and whites are fully combined. Heat a non-stick pan over medium heat and add a small amount of butter.",
    "Python is a high-level programming language known for its readable syntax and large standard library. It supports multiple paradigms including procedural, object-oriented, and functional programming.",
    "The Mediterranean Sea is bordered by Europe to the north, Asia to the east, and Africa to the south. It has played a central role in the trade and cultural exchange between these continents for thousands of years.",
    "Albert Einstein's theory of relativity revolutionized our understanding of space, time, and gravity. The famous equation E=mc² shows the equivalence of mass and energy.",
]


# ---------- Helpers ----------

def file_size_mib(path: str) -> float:
    return os.path.getsize(path) / 1024 / 1024


def safe_mean(xs): return statistics.mean(xs) if xs else 0.0
def safe_median(xs): return statistics.median(xs) if xs else 0.0
def safe_p(xs, p):
    if not xs: return 0.0
    s = sorted(xs)
    return s[min(int(len(s) * p), len(s) - 1)]


# ---------- Perplexity ----------

def make_llm(model_path: str) -> Llama:
    """Load model with logits_all=True so we can compute perplexity."""
    return Llama(
        model_path=model_path,
        n_ctx=512,
        n_threads=2,
        n_gpu_layers=0,
        verbose=False,
        use_mmap=True,
        logits_all=True,  # needed for echo=True + logprobs to work
    )


def compute_pseudo_perplexity(llm: Llama, corpus: list[str]) -> dict:
    """Compute average NLL per token across the corpus.

    Uses create_completion with logprobs=1 and echo=True to get the
    log-probabilities of each prompt token under the model.
    """
    import math

    total_nll = 0.0
    total_tokens = 0
    per_text = []

    for text in corpus:
        try:
            out = llm.create_completion(
                prompt=text,
                max_tokens=1,           # must be >0 for completion to run
                temperature=0.0,
                logprobs=1,             # ask for top-1 logprobs
                echo=True,              # return logprobs for the prompt tokens
            )
            choices = out.get("choices", [])
            if not choices:
                continue
            cp = choices[0]
            logprobs_data = cp.get("logprobs") or {}
            token_logprobs = logprobs_data.get("token_logprobs") or []
            # First token has no logprob (no context); skip None entries
            valid = [float(lp) for lp in token_logprobs if lp is not None]
            if not valid:
                continue
            text_nll = -sum(valid)
            text_tokens = len(valid)
            per_text.append({"text": text[:80], "nll": text_nll, "tokens": text_tokens})
            total_nll += text_nll
            total_tokens += text_tokens
        except Exception as e:
            per_text.append({"text": text[:80], "error": f"{type(e).__name__}: {e}",
                            "nll": 0.0, "tokens": 0})
            continue

    avg_nll = total_nll / max(1, total_tokens)
    return {
        "total_nll": float(total_nll),
        "total_tokens": int(total_tokens),
        "avg_nll_per_token": float(avg_nll),
        "pseudo_perplexity": float(math.exp(avg_nll)) if total_tokens > 0 else 0.0,
        "per_text": per_text,
    }


# ---------- Main ----------

def test_one_model(model_path: str, model_label: str) -> dict:
    print(f"\n{'='*60}")
    print(f"Testing: {model_label} ({model_path})")
    print(f"{'='*60}")

    size_mib = file_size_mib(model_path)
    print(f"Size: {size_mib:.1f} MiB")

    # Load
    t0 = time.perf_counter()
    llm = make_llm(model_path)
    load_time = time.perf_counter() - t0
    print(f"Load time: {load_time:.2f}s")

    # Perplexity
    print("Computing pseudo-perplexity...")
    t0 = time.perf_counter()
    perplexity = compute_pseudo_perplexity(llm, PERPLEXITY_CORPUS)
    ppl_time = time.perf_counter() - t0
    print(f"  avg NLL/token: {perplexity['avg_nll_per_token']:.4f}")
    print(f"  pseudo-perplexity: {perplexity['pseudo_perplexity']:.2f}  ({ppl_time:.1f}s)")

    # Generation tests
    print("Running generation tests...")
    generations = []
    latencies = []
    throughputs = []
    response_lengths = []

    for i, prompt in enumerate(EVAL_PROMPTS):
        t0 = time.perf_counter()
        try:
            out = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are a helpful, concise assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=128,
                temperature=0.0,  # greedy for reproducibility
                top_p=1.0,
            )
            wall = time.perf_counter() - t0
            text = (out.get("choices", [{}])[0]
                       .get("message", {})
                       .get("content", "") or "").strip()
            usage = out.get("usage", {}) or {}
            n_resp = int(usage.get("completion_tokens", 0))
            tps = (n_resp / wall) if wall > 0 else 0.0
            latencies.append(wall)
            throughputs.append(tps)
            response_lengths.append(n_resp)
            generations.append({
                "idx": i,
                "prompt": prompt,
                "response": text,
                "wall_s": wall,
                "tokens_per_s": tps,
                "n_response_tokens": n_resp,
                "ok": True,
            })
            print(f"  [{i+1}/{len(EVAL_PROMPTS)}] {wall:.2f}s, {n_resp} tok, {tps:.1f} tps")
        except Exception as e:
            generations.append({
                "idx": i,
                "prompt": prompt,
                "response": "",
                "wall_s": time.perf_counter() - t0,
                "tokens_per_s": 0.0,
                "n_response_tokens": 0,
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
            })
            print(f"  [{i+1}/{len(EVAL_PROMPTS)}] FAILED: {e}")

    # Free the model
    del llm

    return {
        "label": model_label,
        "model_path": model_path,
        "size_mib": size_mib,
        "load_time_s": load_time,
        "perplexity": perplexity,
        "perplexity_time_s": ppl_time,
        "n_generation_tests": len(EVAL_PROMPTS),
        "n_ok": sum(1 for g in generations if g["ok"]),
        "n_fail": sum(1 for g in generations if not g["ok"]),
        "mean_latency_s": safe_mean(latencies),
        "median_latency_s": safe_median(latencies),
        "p95_latency_s": safe_p(latencies, 0.95),
        "mean_throughput_tokens_per_s": safe_mean(throughputs),
        "median_throughput_tokens_per_s": safe_median(throughputs),
        "mean_response_tokens": safe_mean(response_lengths),
        "median_response_tokens": safe_median(response_lengths),
        "generations": generations,
    }


def main():
    comparison_dir = Path("/home/z/my-project/comparison")
    out_path = comparison_dir / "quant_level_comparison.json"

    # The 5 quantization levels we want to compare.
    # Sorted by expected compression level (least aggressive first).
    models = [
        ("Q8_0",   str(comparison_dir / "Qwen-Q8_0.gguf")),
        ("Q5_K_M", str(comparison_dir / "Qwen-Q5_K_M.gguf")),
        ("Q4_K_M", str(comparison_dir / "Qwen-Q4_K_M.gguf")),
        ("Q2_K",   str(comparison_dir / "Qwen-Q2_K.gguf")),
    ]

    # Reference: also test F16 if available
    f16_path = "/home/z/my-project/comparison/Qwen-F16.gguf"
    if os.path.exists(f16_path):
        models.insert(0, ("F16_reference", f16_path))

    results = []
    for label, path in models:
        if not os.path.exists(path):
            print(f"SKIP (missing): {label} at {path}")
            continue
        r = test_one_model(path, label)
        results.append(r)
        # Flush partial results after each model in case of crash
        with open(out_path, "w") as f:
            json.dump({"models": results, "config": {
                "eval_prompts": EVAL_PROMPTS,
                "perplexity_corpus_lines": len(PERPLEXITY_CORPUS),
                "max_tokens": 128,
                "temperature": 0.0,
                "n_ctx": 512,
                "n_threads": 2,
            }}, f, indent=2)

    print(f"\n=== ALL DONE — wrote {out_path} ===")
    print(f"\n=== SUMMARY TABLE ===")
    print(f"{'Level':<10} {'Size MiB':<10} {'Load s':<8} {'PPL':<10} "
          f"{'Mean lat s':<12} {'Mean tps':<10} {'OK':<5}")
    for r in results:
        print(f"{r['label']:<10} {r['size_mib']:<10.1f} {r['load_time_s']:<8.2f} "
              f"{r['perplexity']['pseudo_perplexity']:<10.2f} "
              f"{r['mean_latency_s']:<12.3f} "
              f"{r['mean_throughput_tokens_per_s']:<10.2f} "
              f"{r['n_ok']}/{r['n_generation_tests']}")


if __name__ == "__main__":
    main()
