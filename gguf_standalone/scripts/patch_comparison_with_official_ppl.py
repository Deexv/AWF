"""
Patch the comparison JSON with official llama-perplexity numbers
(from llama.cpp's binary, the gold standard).

Run this after running scripts/compare_quant_levels.py and after
running llama-perplexity manually on each model.
"""
import json
from pathlib import Path

# Hard-coded from the llama-perplexity runs (all on the same wikitext corpus,
# same n_ctx=2048, same 4 chunks, same n_threads=2 — apples-to-apples).
OFFICIAL_PERPLEXITY = {
    "Q8_0":   {"ppl": 12.4899, "stderr": 0.53260, "n_chunks": 4, "n_ctx": 2048},
    "Q5_K_M": {"ppl": 12.9868, "stderr": 0.55900, "n_chunks": 4, "n_ctx": 2048},
    "Q4_K_M": {"ppl": 12.7638, "stderr": 0.54622, "n_chunks": 4, "n_ctx": 2048},
    "Q2_K":   {"ppl": 15.8469, "stderr": 0.69423, "n_chunks": 4, "n_ctx": 2048},
}

p = Path("/home/z/my-project/comparison/quant_level_comparison.json")
with open(p) as f:
    data = json.load(f)

for m in data["models"]:
    label = m["label"]
    if label in OFFICIAL_PERPLEXITY:
        m["official_perplexity"] = OFFICIAL_PERPLEXITY[label]
        # Also replace the bogus python-computed perplexity with the official one
        ppl = OFFICIAL_PERPLEXITY[label]["ppl"]
        m["perplexity"] = {
            "pseudo_perplexity": float(ppl),
            "avg_nll_per_token": float(__import__("math").log(ppl)),
            "method": "llama-perplexity binary, wikitext-2 test set, 4 chunks × 2048 ctx",
            "stderr": OFFICIAL_PERPLEXITY[label]["stderr"],
        }

# Also add a comparison summary at the top level
data["summary_table"] = [
    {
        "label": m["label"],
        "size_mib": round(m["size_mib"], 1),
        "size_pct_of_f16": round(m["size_mib"] / 988.2 * 100, 1),
        "official_perplexity": m["perplexity"]["pseudo_perplexity"],
        "ppl_stderr": m["perplexity"]["stderr"],
        "mean_latency_s": round(m["mean_latency_s"], 3),
        "mean_throughput_tps": round(m["mean_throughput_tokens_per_s"], 2),
        "n_ok": m["n_ok"],
        "n_total": m["n_generation_tests"],
    }
    for m in data["models"]
]

with open(p, "w") as f:
    json.dump(data, f, indent=2)

print("Patched. Summary table:")
print(f"{'Level':<8} {'Size MiB':<10} {'% F16':<8} {'PPL':<10} {'Lat s':<8} {'tps':<8}")
for r in data["summary_table"]:
    print(f"{r['label']:<8} {r['size_mib']:<10} {r['size_pct_of_f16']:<8} "
          f"{r['official_perplexity']:<10.4f} {r['mean_latency_s']:<8} {r['mean_throughput_tps']:<8}")
