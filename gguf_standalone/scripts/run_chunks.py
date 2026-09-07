"""
Run multiple chunks of the 10K test in one process.

Avoids per-chunk model-load overhead by reusing the same Llama instance
across chunks, while still flushing to disk every chunk.

Usage:
    python run_chunks.py --model compressed/X.gguf --chunks 2 --chunk-size 250
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_10k_test import make_llm, make_prompt, run_one  # noqa: E402


MASTER_CSV = Path("results/conversations_master.csv")
PROGRESS_FILE = Path("results/progress.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--total", type=int, default=10000)
    ap.add_argument("--chunk-size", type=int, default=250)
    ap.add_argument("--chunks", type=int, default=2,
                    help="How many chunks to run this invocation.")
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

    import json

    # Resume
    offset = 0
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            offset = json.load(f).get("next_offset", 0)
        print(f"[CHUNKS] resumed from progress.json: offset={offset}", flush=True)

    if offset >= args.total:
        print(f"[CHUNKS] already done (offset={offset} >= total={args.total}).")
        return

    rng = random.Random(args.seed)

    write_header = not MASTER_CSV.exists()
    csv_fp = open(MASTER_CSV, "a", newline="", encoding="utf-8")
    fields = ["idx", "prompt", "response", "n_prompt_tokens",
              "n_response_tokens", "wall_s", "tokens_per_s", "ok", "error"]
    writer = csv.DictWriter(csv_fp, fieldnames=fields)
    if write_header:
        writer.writeheader()
        csv_fp.flush()

    # Reuse one Llama instance across all chunks
    print(f"[CHUNKS] loading model: {model_path}", flush=True)
    llm = make_llm(model_path, args.n_ctx, args.threads)
    print(f"[CHUNKS] model loaded.", flush=True)

    chunks_run = 0
    overall_t0 = time.perf_counter()
    overall_ok = 0
    overall_fail = 0
    overall_resp_tokens = 0
    overall_gen_time = 0.0  # sum of wall_s for ok calls

    while chunks_run < args.chunks and offset < args.total:
        end = min(offset + args.chunk_size, args.total)
        chunk_t0 = time.perf_counter()
        chunk_ok = 0
        chunk_fail = 0
        chunk_resp_tokens = 0
        chunk_gen_time = 0.0

        for idx in range(offset, end):
            # Generate prompt deterministically by index
            # (we rebuild rng per idx so it's stable across resume)
            local_rng = random.Random(args.seed + idx)
            prompt = make_prompt(local_rng)
            res = run_one(llm, idx, prompt, args.max_tokens, local_rng)
            writer.writerow({
                "idx": res.idx, "prompt": res.prompt, "response": res.response,
                "n_prompt_tokens": res.n_prompt_tokens,
                "n_response_tokens": res.n_response_tokens,
                "wall_s": res.wall_s, "tokens_per_s": res.tokens_per_s,
                "ok": res.ok, "error": res.error,
            })
            if res.ok:
                chunk_ok += 1
                chunk_resp_tokens += res.n_response_tokens
                chunk_gen_time += res.wall_s
                overall_resp_tokens += res.n_response_tokens
                overall_gen_time += res.wall_s
            else:
                chunk_fail += 1
            if (idx - offset + 1) % 50 == 0:
                csv_fp.flush()
                elapsed = time.perf_counter() - chunk_t0
                print(f"  [{idx+1:>5}/{end}] chunk_elapsed={elapsed:5.1f}s "
                      f"ok={chunk_ok} fail={chunk_fail} last_tps={res.tokens_per_s:.1f}",
                      flush=True)

        csv_fp.flush()
        chunk_elapsed = time.perf_counter() - chunk_t0
        overall_ok += chunk_ok
        overall_fail += chunk_fail
        chunks_run += 1

        # Persist progress
        next_offset = end
        with open(PROGRESS_FILE, "w") as f:
            json.dump({"next_offset": next_offset, "total": args.total,
                       "model": model_path}, f, indent=2)

        print(f"[CHUNK {chunks_run}] offset={offset}..{end-1} "
              f"ok={chunk_ok} fail={chunk_fail} elapsed={chunk_elapsed:.1f}s "
              f"avg_tps={chunk_resp_tokens/max(1,chunk_gen_time):.2f} "
              f"next_offset={next_offset}",
              flush=True)

        offset = end

    csv_fp.close()

    overall_elapsed = time.perf_counter() - overall_t0
    print(f"[CHUNKS] total this run: chunks={chunks_run} "
          f"ok={overall_ok} fail={overall_fail} "
          f"elapsed={overall_elapsed:.1f}s "
          f"avg_tps={overall_resp_tokens/max(1,overall_gen_time):.2f}")


if __name__ == "__main__":
    main()
