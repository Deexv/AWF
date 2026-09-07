"""
10K Synthetic Conversation Test - COMPRESSED MODEL ONLY
=======================================================

This script loads ONLY the compressed Q4_K_M GGUF file and runs 10,000
synthetic conversations against it. The original (uncompressed) HF model
is NEVER loaded, NEVER referenced, NEVER needed at runtime.

This is the proof that the compressed file is truly standalone.

Usage:
    python run_10k_test.py --model compressed/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
    python run_10k_test.py --model compressed/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf --n 10000 --workers 2
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from pathlib import Path
from threading import Lock
from typing import Any

from llama_cpp import Llama


# ---------- Synthetic prompt corpus ----------
# A diverse set of instruction-style prompts that exercise the model's
# instruct-tuning. Each is a self-contained user turn.

PROMPT_TEMPLATES = [
    "List {n} {adj} {noun} and explain each in one sentence.",
    "Write a {length} story about a {adj} {noun} that learns to {verb}.",
    "What are the main differences between {noun} and {noun2}?",
    "Translate 'Hello, how are you?' into a polite request involving {noun}.",
    "Give {n} tips for someone trying to {verb} {adverb}.",
    "Summarize the concept of {noun} for a beginner in {length} sentences.",
    "Write a function (pseudo-code is fine) that takes a list of {noun} and returns the {adj} one.",
    "If a {noun} costs ${n} and you buy {n2}, how much do you pay? Show your reasoning.",
    "Name {n} famous people associated with {noun} and what they did.",
    "Explain why {noun} is {adj}, using an analogy involving {noun2}.",
]

ADJECTIVES = ["small", "ancient", "modern", "mysterious", "tiny", "vast", "forgotten",
              "futuristic", "rustic", "elegant", "broken", "shiny", "weathered",
              "electric", "wooden", "underwater", "mountainous", "lunar", "urban"]

NOUNS = ["robot", "wizard", "library", "dragon", "cat", "knight", "garden",
         "spaceship", "merchant", "professor", "island", "castle", "village",
         "river", "machine", "book", "festival", "forest", "kingdom", "comet"]

VERBS = ["fly", "sing", "negotiate", "build", "escape", "cook", "compose",
         "translate", "investigate", "celebrate", "explore", "defend", "repair"]

ADVERBS = ["quickly", "patiently", "creatively", "safely", "on a budget",
           "with friends", "at night", "in winter", "without tools"]

LENGTHS = ["short", "medium", "long", "two-paragraph", "rhyming"]


def make_prompt(rng: random.Random) -> str:
    """Generate one synthetic user prompt."""
    tpl = rng.choice(PROMPT_TEMPLATES)
    return tpl.format(
        n=rng.randint(3, 7),
        n2=rng.randint(2, 12),
        adj=rng.choice(ADJECTIVES),
        noun=rng.choice(NOUNS),
        noun2=rng.choice(NOUNS),
        verb=rng.choice(VERBS),
        adverb=rng.choice(ADVERBS),
        length=rng.choice(LENGTHS),
    )


# ---------- Result records ----------

@dataclass
class ConversationResult:
    idx: int
    prompt: str
    response: str
    n_prompt_tokens: int
    n_response_tokens: int
    wall_s: float
    tokens_per_s: float
    ok: bool
    error: str = ""


@dataclass
class TestStats:
    model_path: str
    n_conversations: int
    n_ok: int = 0
    n_fail: int = 0
    total_prompt_tokens: int = 0
    total_response_tokens: int = 0
    total_wall_s: float = 0.0
    latencies: list[float] = field(default_factory=list)
    throughputs: list[float] = field(default_factory=list)
    response_lengths: list[int] = field(default_factory=list)
    started_at: float = 0.0
    ended_at: float = 0.0


# ---------- Worker ----------

def make_llm(model_path: str, n_ctx: int, n_threads: int) -> Llama:
    """Load ONLY the compressed .gguf file. No base model, no adapters."""
    return Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=0,           # CPU-only by design (portability)
        verbose=False,
        use_mlock=False,
        use_mmap=True,
    )

def run_one(
    llm: Llama,
    idx: int,
    prompt: str,
    max_tokens: int,
    rng: random.Random,
) -> ConversationResult:
    """Run a single chat completion against the compressed model."""
    t0 = time.perf_counter()
    try:
        # Use create_chat_completion so the model's chat template is applied.
        # For Qwen2.5 the template is shipped inside the GGUF file metadata,
        # so we don't need the HF tokenizer at runtime.
        out = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful, concise assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        wall = time.perf_counter() - t0
        text = (out.get("choices", [{}])[0]
                  .get("message", {})
                  .get("content", "") or "").strip()
        usage = out.get("usage", {}) or {}
        n_prompt = int(usage.get("prompt_tokens", 0))
        n_resp = int(usage.get("completion_tokens", 0))
        tps = (n_resp / wall) if wall > 0 else 0.0
        return ConversationResult(
            idx=idx, prompt=prompt, response=text,
            n_prompt_tokens=n_prompt, n_response_tokens=n_resp,
            wall_s=wall, tokens_per_s=tps,
            ok=True,
        )
    except Exception as e:
        wall = time.perf_counter() - t0
        return ConversationResult(
            idx=idx, prompt=prompt, response="",
            n_prompt_tokens=0, n_response_tokens=0,
            wall_s=wall, tokens_per_s=0.0,
            ok=False, error=f"{type(e).__name__}: {e}",
        )


# ---------- Driver ----------

def main():
    ap = argparse.ArgumentParser(description="10K compressed-model conversation test")
    ap.add_argument("--model", required=True,
                    help="Path to the COMPRESSED .gguf file (e.g. compressed/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf)")
    ap.add_argument("--n", type=int, default=10000,
                    help="Number of conversations to run (default 10000)")
    ap.add_argument("--max-tokens", type=int, default=64,
                    help="Max tokens to generate per conversation (default 64)")
    ap.add_argument("--n-ctx", type=int, default=1024,
                    help="Per-conversation context window (default 1024)")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel llama instances. Each loads its own copy of the model.")
    ap.add_argument("--threads", type=int, default=2,
                    help="Threads per llama instance. With workers=N on a C-core box, use --threads floor(C/N).")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out-dir", default="results",
                    help="Where to write JSON + CSV results")
    ap.add_argument("--sample-every", type=int, default=500,
                    help="Print progress every N conversations")
    ap.add_argument("--save-every", type=int, default=1000,
                    help="Flush partial results to CSV every N conversations")
    args = ap.parse_args()

    model_path = os.path.abspath(args.model)
    if not os.path.exists(model_path):
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        sys.exit(2)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = TestStats(
        model_path=model_path,
        n_conversations=args.n,
        started_at=time.time(),
    )

    rng = random.Random(args.seed)
    prompts = [make_prompt(rng) for _ in range(args.n)]

    csv_path = out_dir / f"conversations_{int(stats.started_at)}.csv"
    json_path = out_dir / f"summary_{int(stats.started_at)}.json"

    print(f"[10K TEST] model      : {model_path}")
    print(f"[10K TEST] size on disk: {os.path.getsize(model_path)/1024/1024:.1f} MiB")
    print(f"[10K TEST] target     : {args.n} conversations")
    print(f"[10K TEST] workers    : {args.workers}")
    print(f"[10K TEST] max_tokens : {args.max_tokens}")
    print(f"[10K TEST] output     : {csv_path}")
    sys.stdout.flush()

    csv_lock = Lock()
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_fp:
        writer = csv.DictWriter(csv_fp, fieldnames=list(ConversationResult.__annotations__.keys()))
        writer.writeheader()
        csv_fp.flush()

        def flush(result: ConversationResult):
            with csv_lock:
                writer.writerow(asdict(result))
                # Periodically flush so partial progress is durable.
                if result.idx % args.save_every == 0:
                    csv_fp.flush()

        n_workers = max(1, args.workers)
        # Each worker gets its own Llama instance. They all load the same
        # compressed .gguf file; no base model is referenced anywhere.
        llms = [make_llm(model_path, args.n_ctx, args.threads) for _ in range(n_workers)]

        def _do(item):
            idx, prompt = item
            worker_id = idx % n_workers
            return run_one(llms[worker_id], idx, prompt, args.max_tokens, rng)

        t_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_do, (i, prompts[i])): i for i in range(args.n)}
            done = 0
            for fut in as_completed(futures):
                res: ConversationResult = fut.result()
                if res.ok:
                    stats.n_ok += 1
                    stats.total_prompt_tokens += res.n_prompt_tokens
                    stats.total_response_tokens += res.n_response_tokens
                    stats.latencies.append(res.wall_s)
                    stats.throughputs.append(res.tokens_per_s)
                    stats.response_lengths.append(res.n_response_tokens)
                else:
                    stats.n_fail += 1
                stats.total_wall_s += res.wall_s
                flush(res)
                done += 1
                if done % args.sample_every == 0 or done == args.n:
                    elapsed = time.perf_counter() - t_start
                    avg_tps = (sum(stats.throughputs) / len(stats.throughputs)
                               if stats.throughputs else 0.0)
                    print(f"  [{done:>6}/{args.n}] elapsed={elapsed:6.1f}s "
                          f"ok={stats.n_ok} fail={stats.n_fail} "
                          f"avg_tps={avg_tps:.2f} "
                          f"last_tokens={res.n_response_tokens}")
                    sys.stdout.flush()

    stats.ended_at = time.time()

    # ----- Summary -----
    def safe_mean(xs):
        return statistics.mean(xs) if xs else 0.0

    def safe_median(xs):
        return statistics.median(xs) if xs else 0.0

    def safe_p95(xs):
        if not xs:
            return 0.0
        xs2 = sorted(xs)
        return xs2[int(len(xs2) * 0.95)]

    summary = {
        "model_path": stats.model_path,
        "model_size_mib": os.path.getsize(model_path) / 1024 / 1024,
        "n_conversations": stats.n_conversations,
        "n_ok": stats.n_ok,
        "n_fail": stats.n_fail,
        "ok_rate": stats.n_ok / max(1, stats.n_conversations),
        "total_prompt_tokens": stats.total_prompt_tokens,
        "total_response_tokens": stats.total_response_tokens,
        "wall_clock_s": stats.ended_at - stats.started_at,
        "sum_of_per_call_wall_s": stats.total_wall_s,
        "latency_s": {
            "mean": safe_mean(stats.latencies),
            "median": safe_median(stats.latencies),
            "p95": safe_p95(stats.latencies),
            "min": min(stats.latencies) if stats.latencies else 0.0,
            "max": max(stats.latencies) if stats.latencies else 0.0,
        },
        "throughput_tokens_per_s": {
            "mean": safe_mean(stats.throughputs),
            "median": safe_median(stats.throughputs),
            "p95": safe_p95(stats.throughputs),
        },
        "response_length_tokens": {
            "mean": safe_mean(stats.response_lengths),
            "median": safe_median(stats.response_lengths),
            "min": min(stats.response_lengths) if stats.response_lengths else 0,
            "max": max(stats.response_lengths) if stats.response_lengths else 0,
        },
        "aggregate_output_tokens_per_s": (
            stats.total_response_tokens / max(1e-9, stats.ended_at - stats.started_at)
        ),
        "workers": n_workers,
        "threads_per_worker": args.threads,
        "max_tokens": args.max_tokens,
        "n_ctx": args.n_ctx,
        "seed": args.seed,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {json_path}")


if __name__ == "__main__":
    main()
