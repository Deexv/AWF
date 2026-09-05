# AWF — Algorithmic Weight Fabric

> **Two verified breakthroughs in one repo:**
> 1. **Weight compression**: 8× fewer params, 9× less storage, generates more diverse text than dense
> 2. **Training speedup**: 10-50× faster training via output caching (verified on 5M chars)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## Quick Start

```bash
git clone https://github.com/Deexv/AWF.git
cd AWF
pip install -r requirements.txt
python scripts/download_tinystories.py

# Chat with the pre-trained AWF LLM
python scripts/chat_v2.py --interactive

# Benchmark training speedup (10x verified)
python scripts/benchmark_output_cache.py --time_budget 100

# Benchmark AWF vs Dense compression
python scripts/benchmark_10m.py
```

## Breakthrough 1: Algorithmic Weight Fabric (AWF)

Weights are generated on demand by a small shared program, not stored as a tensor.

### Results (TinyStories, 622K params AWF vs 4.9M Dense)

| Metric | Dense (4.9M) | AWF (622K) | Improvement |
|---|---|---|---|
| Parameters | 4,903,168 | 622,048 | **7.88× fewer** |
| Storage | 19,153 KB | 2,116 KB | **9.05× smaller** |
| Val accuracy | 33.8% | 38.3% | **AWF wins** |
| Text repetition | 64.4% | 0.6% | **107× less** |
| Longest char run | 69.5 | 1.7 | **42× shorter** |

Dense generates "pppppppp..." (repetition collapse). AWF generates real words and sentence structure.

### How AWF Works

```
                    ┌──────────────────────────┐
                    │  CoordGenerator (shared) │  ~43K params, AMORTIZED across all layers
                    │  - 16 Fourier features   │
                    │  - Layer embedding        │
                    │  - 3-layer MLP (hidden=128)│
                    └──────────────────────────┘
                                ↓
                    generates low-res weight field
                                ↓
                    bilinear upsample to (M, N)
                                ↓
                ┌───────────────┴───────────────┐
                ↓                               ↓
          Per-layer low-rank                Per-layer sparse
          W += U @ V (rank 16)             W += scale * tern {-1,0,+1} (k=256)
                ↓
          Full weight matrix (reconstructed on demand, never stored)
```

## Breakthrough 2: Output Caching Training Speedup

### The Idea

Most transformer blocks produce nearly identical outputs across consecutive batches. By caching block OUTPUTS (not just weights), we can skip the ENTIRE block — no generator, no matmul, no backward. Zero compute.

### How It Works

For each transformer block, at each step:
1. Pre-hook captures block INPUT
2. Compute input novelty (cosine similarity to recent inputs)
3. If novelty < threshold AND not stale: **SKIP block entirely**
   - Patch forward to return cached output (zero compute)
   - Freeze all parameters (no backward)
4. Else: compute block normally, cache output

### Verified Results (5M chars — 5× larger than initial test)

| Config | Skip% | Learning Ratio | Throughput | **Combined** |
|---|---|---|---|---|
| Standard | 0% | 1.00× | 22.2 | 1.00× |
| Conservative (ms=5) | 83% | 0.96× | 229.5 | **9.92×** |
| Moderate (ms=20) | 83% | 0.96× | 231.8 | **10.01×** |
| **Extreme (ms=9999)** | **100%** | **0.97×** | **1,213.0** | **53.09×** |

### What Each Config Means

- **Conservative (ms=5)**: Blocks recompute every 5 steps. Safe, robust. ~10× speedup.
- **Extreme (ms=9999)**: Blocks freeze after 5-step warmup. Only embeddings + head train. ~50× speedup.

The extreme config works because transformer blocks converge quickly — after brief warmup, they act as a fixed feature extractor, and subsequent learning happens in the embeddings and output head.

### Config

```python
from awf.output_caching_trainer import OutputCachingTrainer

# 10x speedup (conservative, blocks update every 5 steps)
trainer = OutputCachingTrainer(model, lr=1e-3,
    reuse_threshold=0.80, max_staleness=5, warmup_steps=10, min_full_blocks=1)

# 50x speedup (extreme, blocks freeze after warmup)
trainer = OutputCachingTrainer(model, lr=1e-3,
    reuse_threshold=0.99, max_staleness=9999, warmup_steps=5, min_full_blocks=0)

for x, y in loader:
    loss, stats = trainer.step(x, y)
```

## Resumable Training

AWF training resumes from where you left off — even across sessions, machines, or datasets:

```bash
# Train for 8 minutes today
python scripts/train.py --resume --event_driven --time_budget 510

# Continue tomorrow (auto-loads checkpoint + optimizer state)
python scripts/train.py --resume --event_driven --time_budget 510

# Train on multiple datasets across runs
python scripts/train.py --resume --datasets data/tinystories.txt data/my_text.txt
```

## GPU Support (Google Colab)

The training script auto-detects CUDA. On Colab T4 GPU: ~10× faster than CPU.

Open `scripts/AWF_Training_GPU.ipynb` in Google Colab with GPU runtime.

## Repository Structure

```
AWF/
├── awf/
│   ├── core.py                          # AWF library (CoordGenerator, AWFLinear, AWFTransformer)
│   ├── event_training.py                # Event-driven trainer v1 (scout-based)
│   ├── event_training_v2.py             # v2 (scoutless + adaptive warmup + staleness)
│   ├── event_training_v3.py             # v3 (weight caching at layer level)
│   ├── block_v3_trainer.py             # Block-level weight caching (1.81x)
│   ├── output_caching_trainer.py       # ⭐ OUTPUT CACHING (10-50x speedup)
│   ├── block_event_trainer.py           # Block-level event trainer with learned gate
│   ├── gii_trainer.py                   # Gradient Information Index (batch filtering)
│   ├── combined_trainer.py              # GII + block-level combined
│   └── grad_only_trainer.py            # Gradient-norm-only baseline
├── scripts/
│   ├── train.py                         # ⭐ Unified training (GPU, resume, multi-dataset, --event_driven)
│   ├── chat_v2.py                       # Enhanced chat (temp, top-k, top-p, rep penalty)
│   ├── benchmark_output_cache.py       # ⭐ Output caching benchmark (the speedup proof)
│   ├── benchmark_10m.py               # AWF vs Dense benchmark
│   ├── benchmark_max_final.py          # Block-v3 vs standard
│   ├── AWF_Training_GPU.ipynb          # Google Colab notebook
│   ├── download_tinystories.py         # Dataset downloader
│   └── ... (tuning scripts)
├── checkpoints/
│   └── awf_10m.pt                       # Trained AWF LLM (622K params, 2.1MB)
├── benchmarks/
│   ├── final_10x_benchmark.json        # 10x speedup (1M chars)
│   ├── verification_50x.json           # 54x speedup verified (200s)
│   ├── large_dataset_test.json         # 53x speedup on 5M chars (scales!)
│   ├── 10m_benchmark.json              # AWF vs Dense compression
│   └── ... (other benchmarks)
├── docs/
│   ├── TECHNICAL.md                     # Architecture whitepaper
│   └── BUSINESS_CASE.md                # Investment pitch
├── requirements.txt
├── LICENSE
└── README.md
```

## Reproducing the Results

```bash
# 1. Training speedup (10x — takes ~3 min)
python scripts/benchmark_output_cache.py --time_budget 100

# 2. AWF vs Dense compression (takes ~2 min)
python scripts/benchmark_10m.py

# 3. Chat with the model
python scripts/chat_v2.py --interactive

# 4. Train with output caching (10x speedup)
python scripts/train.py --resume --event_driven --time_budget 510

# 5. Train on GPU (Google Colab)
# Open scripts/AWF_Training_GPU.ipynb
```

## Progression of Speedups

| Version | Approach | Combined Speedup |
|---|---|---|
| v0.5 | Scout-based event training (layer-level) | 1.89× |
| v0.6 | Layer-level weight caching | 1.54× |
| v0.7 | Block-level weight caching | 1.81× |
| v0.8 | Output caching (skip blocks entirely) | 10.70× |
| **v0.9** | **Output caching (extreme, freeze after warmup)** | **53.09×** |

The key insight: caching block OUTPUTS (not just weights) skips the entire matmul — the most expensive operation. At high staleness, this effectively freezes the transformer blocks after brief warmup, yielding 50×+ speedup.

## Honest Limitations

1. **The 50× extreme config freezes transformer blocks after 5 steps.** This works on small models (622K) and small datasets (5M chars) because the blocks converge quickly. On very large models (100M+ params) with complex data, the frozen blocks may not provide good enough features — quality loss would be larger.

2. **The 10× conservative config is more robust.** Blocks recompute every 5 steps, maintaining freshness. This should scale better to larger models.

3. **GPU speedup may differ.** GPU backward is already fast; the overhead/savings tradeoff may be less favorable. The win is most pronounced on CPU.

4. **AWF's text quality is still rough.** The 622K model generates word-like output but not fluent English. Scaling to 2-5M params with GPU training (100K+ steps) is needed for production-quality text.

5. **Not yet tested at LLM scale** (>100M params). The theoretical 192× compression at GPT-3 scale requires GPU experiments.

## Roadmap

- [x] v0.1 — Core AWF library, char-level LLM demo
- [x] v0.2 — Sparse ternary corrections + int8 quantization
- [x] v0.3 — Real 10M LLM on TinyStories with chat
- [x] v0.4 — Resumable training, GPU support, Colab notebook
- [x] v0.5 — Gradient event-driven training (1.89×)
- [x] v0.6 — Layer-level weight caching (1.54×)
- [x] v0.7 — Block-level weight caching (1.81×)
- [x] v0.8 — Output caching — 10× speedup
- [x] **v0.9 — Output caching extreme — 53× speedup (verified on 5M chars)**
- [ ] v1.0 — Scale to 2M+ params with GPU for production-quality text
- [ ] v1.1 — QAT for int8 inference
- [ ] v2.0 — Production AWF compression for real LLMs (Llama, Mistral)

## License

Apache 2.0 — use commercially, modify freely.

## Citation

```bibtex
@misc{awf2026,
  title={Algorithmic Weight Fabric: Storing Neural Network Weights as Generative Programs},
  author={AWF Research},
  year={2026},
  url={https://github.com/Deexv/AWF}
}
```
