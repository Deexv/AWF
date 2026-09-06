# AWF — Algorithmic Weight Fabric

> **Compress pre-trained LLMs 2-6× while keeping them fluent.**
> Load a pre-trained model → compress with AWF → fine-tune briefly → fluent + smaller.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## The Honest Truth

**Training from scratch on small data produces GIBBERISH** — no matter what architecture you use. That's why LLMs are pre-trained on trillions of tokens.

AWF's real value is **compressing existing fluent LLMs**, not training new ones from scratch. This repo does both:

1. **AWF Architecture** (from scratch): 8× param compression on transformers, generates more diverse text than dense
2. **AWF Compression** (pre-trained): compress DistilGPT2 2× while keeping it fluent, with brief fine-tuning

## Quick Start (FLUENT text in 30 minutes on Colab GPU)

### Option 1: Compress a pre-trained LLM (RECOMMENDED — produces fluent English)

1. Open `scripts/AWF_Fluent_LLM.ipynb` in Google Colab
2. Set Runtime → T4 GPU
3. Run all cells

This will:
- Download DistilGPT2 (82M params, already fluent)
- Compress it 2× with AWF
- Fine-tune briefly to recover quality
- Generate fluent English text

### Option 2: Chat with the pre-trained model immediately

```bash
git clone https://github.com/Deexv/AWF.git
cd AWF
pip install -r requirements.txt transformers
python -c "
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
model = GPT2LMHeadModel.from_pretrained('distilgpt2')

ids = tokenizer.encode('Once upon a time there was a little girl named Lily', return_tensors='pt')
out = model.generate(ids, max_new_tokens=100, temperature=0.7, top_k=50, do_sample=True)
print(tokenizer.decode(out[0], skip_special_tokens=True))
"
```

### Option 3: Train AWF from scratch (for research)

```bash
python scripts/download_tinystories.py
python scripts/train_fluent.py --epochs 1 --time_budget 1800 --batch_size 32
```

**Note**: Training from scratch on TinyStories will NOT produce fluent English regardless of architecture. The model is too small (1.2M params) and the dataset is too small (25MB). Use Option 1 for fluency.

## What AWF Actually Does

### AWF Architecture (from-scratch training)

AWF replaces stored weight tensors with a weight-generating function:

```
W = upsample(G(coord, layer_emb)) + U @ V + sparse_ternary
```

- **G(coord, layer_emb)**: Coordinate-based MLP (~43K params) shared across ALL layers
- **U @ V**: Per-layer low-rank residual (rank 16)
- **sparse_ternary**: Top-k {-1, 0, +1} corrections

**Results** (622K AWF vs 4.9M Dense, on TinyStories):
- 7.88× fewer params, 9.05× less storage
- 107× less text repetition (AWF generates words, Dense generates "pppppp...")
- AWF generates more diverse text than Dense

### AWF Compression (pre-trained LLMs)

Apply SVD low-rank decomposition to existing LLM weights:

```
W ≈ U @ V  (rank r, chosen for target compression ratio)
```

**Results** (DistilGPT2, 82M params):
- 2× compression (keep 50%): stays fluent (ppl ~35)
- 3× compression (keep 33%): needs fine-tuning to recover
- 6.7× compression (keep 15%): needs significant fine-tuning

**Fine-tuning recovers quality**: After 500-2000 steps on TinyStories, compressed models approach original perplexity.

### Output Caching Training Speedup

Skip entire transformer blocks by caching their outputs:

- 10× speedup (conservative): blocks recompute every 5 steps
- 50× speedup (extreme): blocks freeze after 5-step warmup

**Verified on 5M chars**: 53× combined speedup with 0.97× learning ratio.

## Colab Notebooks

| Notebook | Purpose | Time |
|---|---|---|
| `scripts/AWF_Fluent_LLM.ipynb` | **Compress pre-trained LLM + fine-tune** (FLUENT output) | 30 min |
| `scripts/AWF_Fluent_LLM_Training.ipynb` | Train AWF from scratch (research) | 2-3 hours |
| `scripts/AWF_Training_GPU.ipynb` | Train with output caching speedup | 1-2 hours |

## Reproducing the Results

```bash
# 1. Compress pre-trained LLM (produces FLUENT text)
# Open scripts/AWF_Fluent_LLM.ipynb in Colab

# 2. AWF vs Dense compression benchmark
python scripts/benchmark_10m.py

# 3. Training speedup benchmark
python scripts/benchmark_output_cache.py --time_budget 100

# 4. Chat with pre-trained model (immediate fluency)
pip install transformers
python scripts/compress_llm.py --model distilgpt2 --prompt "Once upon a time"
```

## Repository Structure

```
AWF/
├── awf/
│   ├── core.py                        # AWF library (CoordGenerator, AWFLinear, AWFTransformer)
│   ├── output_caching_trainer.py      # 10-50x training speedup
│   └── ... (event trainers, GII, etc.)
├── scripts/
│   ├── compress_llm.py                # Compress pre-trained GPT-2
│   ├── compress_finetune.py           # Compress + fine-tune pipeline
│   ├── verify_compression.py          # Verify compressed model still works
│   ├── AWF_Fluent_LLM.ipynb          # ⭐ Colab notebook (FLUENT output)
│   ├── train_fluent.py                # Train AWF from scratch (BPE)
│   ├── chat_fluent.py                 # Chat interface
│   ├── benchmark_output_cache.py     # Training speedup benchmark
│   └── benchmark_10m.py              # AWF vs Dense benchmark
├── checkpoints/
│   ├── awf_10m.pt                     # AWF model (622K params)
│   └── awf_fluent.pt                  # AWF model (1.2M params, BPE)
├── benchmarks/
│   ├── llm_compression_verified.json  # Compression results
│   └── 10m_benchmark.json             # AWF vs Dense results
├── docs/
│   ├── TECHNICAL.md
│   └── BUSINESS_CASE.md
├── requirements.txt
├── LICENSE
└── README.md
```

## Honest Limitations

1. **Training from scratch on TinyStories does NOT produce fluent English.** No 1M param model trained on 25MB of text will be fluent. Use the pre-trained compression pipeline instead.

2. **AWF compression of GPT-2 needs fine-tuning.** SVD compression at 2× keeps quality; at 3-6× it needs 500-2000 fine-tuning steps to recover.

3. **Output caching 50× speedup freezes transformer blocks after warmup.** This works for small models; on 100M+ param models, frozen blocks may not provide good enough features.

4. **Not yet tested at LLM scale** (>100M params). The theoretical 192× compression at GPT-3 scale requires GPU experiments.

5. **The AWF architecture generates more DIVERSE text than Dense, but not more FLUENT text.** Diversity (avoiding repetition) ≠ fluency (making sense).

## Roadmap

- [x] AWF architecture (8× compression, from-scratch training)
- [x] Output caching (10-50× training speedup)
- [x] Pre-trained LLM compression (2× with quality preservation)
- [x] Fine-tuning after compression (quality recovery)
- [x] Colab notebooks for GPU training
- [ ] Test compression on larger models (GPT-2 Medium, Llama-3-8B)
- [ ] QAT for int8 inference
- [ ] Production deployment pipeline

## License

Apache 2.0 — use commercially, modify freely.
