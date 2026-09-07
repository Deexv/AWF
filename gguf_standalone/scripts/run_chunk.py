"""
Chunked 10K runner.

Runs a fixed chunk of conversations against the compressed model and APPENDS
the results to a master CSV. Re-running advances the offset.

This is the safe way to do long tests inside an interactive shell that kills
background processes — each chunk fits inside a single bash timeout window.

Usage:
    python run_chunk.py --model compressed/X.gguf --total 10000 --chunk 250 --offset 0
    python run_chunk.py --model compressed/X.gguf --total 10000 --chunk 250 --offset 250
    ...
    python run_chunk.py --model compressed/X.gguf --total 10000 --chunk 250 --offset 9750

After all chunks are done, run finalize_results.py to merge.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

# Make sure we can import the test harness
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_10k_test import make_llm, make_prompt, run_one  # noqa: E402


MASTER_CSV = Path("results/conversations_master.csv")
PROGRESS_FILE = Path("results/progress.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--total", type=int, default=10000)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--offset", type=int, default=0,
                    help="Start at this conversation index (auto-advanced if omitted via progress.json).")
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--n-ctx", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    model_path = os.path.abspath(args.model)
    if not os.path.exists(model_path):
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        sys.exit(2)

    Path("results").mkdir(exist_ok=True)

    # Determine starting offset
    offset = args.offset
    if offset == 0 and PROGRESS_FILE.exists():
        import json
        with open(PROGRESS_FILE) as f:
            offset = json.load(f).get("next_offset", 0)
        print(f"[CHUNK] resumed from progress.json: offset={offset}")

    if offset >= args.total:
        print(f"[CHUNK] already done (offset={offset} >= total={args.total}).")
        return

    end = min(offset + args.chunk, args.total)

    # Deterministic prompt generation: regenerate the same prompt for the same idx.
    # We seed once per idx so resuming doesn't shift the corpus.
    rng = random.Random(args.seed)
    # Pre-generate prompts up to `end` (memory-bounded — prompts are tiny).
    prompts = []
    for i in range(end):
        prompts.append(make_prompt(rng))

    # Append mode: write header only on first chunk
    write_header = not MASTER_CSV.exists()
    with open(MASTER_CSV, "a", newline="", encoding="utf-8") as csv_fp:
        fields = ["idx", "prompt", "response", "n_prompt_tokens",
                 "n_response_tokens", "wall_s", "tokens_per_s", "ok", "error"]
        writer = csv.DictWriter(csv_fp, fieldnames=fields)
        if write_header:
            writer.writeheader()
            csv_fp.flush()

        llm = make_llm(model_path, args.n_ctx, args.threads)

        t0 = time.perf_counter()
        ok = 0
        fail = 0
        total_resp_tokens = 0
        for idx in range(offset, end):
            res = run_one(llm, idx, prompts[idx], args.max_tokens, rng)
            writer.writerow({
                "idx": res.idx, "prompt": res.prompt, "response": res.response,
                "n_prompt_tokens": res.n_prompt_tokens,
                "n_response_tokens": res.n_response_tokens,
                "wall_s": res.wall_s, "tokens_per_s": res.tokens_per_s,
                "ok": res.ok, "error": res.error,
            })
            if (idx - offset + 1) % 25 == 0:
                csv_fp.flush()
            if res.ok:
                ok += 1
                total_resp_tokens += res.n_response_tokens
            else:
                fail += 1
            if (idx - offset + 1) % 50 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  [{idx+1:>5}/{end}] chunk_elapsed={elapsed:5.1f}s "
                      f"ok={ok} fail={fail} last_tps={res.tokens_per_s:.1f}",
                      flush=True)

        csv_fp.flush()

    # Persist progress
    import json
    next_offset = end
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"next_offset": next_offset, "total": args.total,
                   "model": model_path}, f, indent=2)

    elapsed = time.perf_counter() - t0
    print(f"[CHUNK] done offset={offset}..{end-1} "
          f"ok={ok} fail={fail} elapsed={elapsed:.1f}s "
          f"avg_tps={total_resp_tokens/max(1,elapsed):.2f} "
          f"next_offset={next_offset}")


if __name__ == "__main__":
    main()
