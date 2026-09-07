"""
Aggregate the chunked test results into a final summary.

After running run_chunks.py multiple times to reach 10K conversations,
this script reads the master CSV and produces:

  - results/final_summary.json    (aggregate metrics)
  - results/sample_responses.json (first 20 + last 20 responses, for inspection)

Usage:
    python finalize_results.py
"""
from __future__ import annotations

import csv
import json
import os
import statistics
import sys
from pathlib import Path

MASTER_CSV = Path("results/conversations_master.csv")
SUMMARY_OUT = Path("results/final_summary.json")
SAMPLES_OUT = Path("results/sample_responses.json")


def safe_mean(xs): return statistics.mean(xs) if xs else 0.0
def safe_median(xs): return statistics.median(xs) if xs else 0.0
def safe_p(xs, p):
    if not xs: return 0.0
    s = sorted(xs)
    return s[min(int(len(s) * p), len(s) - 1)]


def main():
    if not MASTER_CSV.exists():
        print(f"ERROR: {MASTER_CSV} not found. Run run_chunks.py first.")
        sys.exit(2)

    rows = []
    with open(MASTER_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["idx"] = int(r["idx"])
            r["n_prompt_tokens"] = int(r["n_prompt_tokens"])
            r["n_response_tokens"] = int(r["n_response_tokens"])
            r["wall_s"] = float(r["wall_s"])
            r["tokens_per_s"] = float(r["tokens_per_s"])
            r["ok"] = (r["ok"] == "True")
            rows.append(r)

    n_total = len(rows)
    n_ok = sum(1 for r in rows if r["ok"])
    n_fail = n_total - n_ok
    latencies = [r["wall_s"] for r in rows if r["ok"]]
    tps = [r["tokens_per_s"] for r in rows if r["ok"]]
    resp_lens = [r["n_response_tokens"] for r in rows if r["ok"]]
    total_prompt_tokens = sum(r["n_prompt_tokens"] for r in rows if r["ok"])
    total_resp_tokens = sum(r["n_response_tokens"] for r in rows if r["ok"])
    sum_of_wall = sum(r["wall_s"] for r in rows if r["ok"])

    # Errors bucketized
    errors = {}
    for r in rows:
        if not r["ok"] and r["error"]:
            errors[r["error"]] = errors.get(r["error"], 0) + 1

    # Model size
    model_path = None
    progress_file = Path("results/progress.json")
    if progress_file.exists():
        with open(progress_file) as f:
            model_path = json.load(f).get("model")
    model_size_mib = (os.path.getsize(model_path) / 1024 / 1024) if model_path and os.path.exists(model_path) else None

    summary = {
        "model_path": model_path,
        "model_size_mib": model_size_mib,
        "n_conversations": n_total,
        "n_ok": n_ok,
        "n_fail": n_fail,
        "ok_rate": n_ok / max(1, n_total),
        "total_prompt_tokens": total_prompt_tokens,
        "total_response_tokens": total_resp_tokens,
        "sum_of_per_call_wall_s": sum_of_wall,
        "mean_throughput_tokens_per_s": safe_mean(tps),
        "median_throughput_tokens_per_s": safe_median(tps),
        "p95_throughput_tokens_per_s": safe_p(tps, 0.95),
        "mean_latency_s": safe_mean(latencies),
        "median_latency_s": safe_median(latencies),
        "p95_latency_s": safe_p(latencies, 0.95),
        "mean_response_tokens": safe_mean(resp_lens),
        "median_response_tokens": safe_median(resp_lens),
        "errors": errors,
    }

    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote: {SUMMARY_OUT}")

    # Sample first 20 + last 20 responses for inspection
    samples = {
        "first_20": [
            {"idx": r["idx"], "prompt": r["prompt"], "response": r["response"]}
            for r in rows[:20]
        ],
        "last_20": [
            {"idx": r["idx"], "prompt": r["prompt"], "response": r["response"]}
            for r in rows[-20:]
        ],
    }
    with open(SAMPLES_OUT, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
    print(f"Wrote: {SAMPLES_OUT}")

    print("\n=== FINAL SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
