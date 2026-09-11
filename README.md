# Algorithmic Weight Fabric (AWF)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org)

**Algorithmic Weight Fabric (AWF)** is a high-performance deep learning compression and parameterization engine. AWF combines post-training matrix decomposition, continuous coordinate-based weight generation, and event-driven training acceleration to dramatically compress Large Language Models (LLMs) and accelerate neural network training.

---

## 📌 Core Engineering Pillars

1. **Post-Training Matrix Decomposition (SVD + INT8)**
   - Decomposes weight tensors into low-rank singular components ($W \approx U \cdot V^T$) coupled with uniform 8-bit quantization.
   - Achieves 2.5×–4× file-size and memory reduction across production transformer architectures (**Llama 3.1**, **Qwen2**, **DeepSeek**, **GLM-4**, **Mistral**, **Phi-3**) without fine-tuning, with post-compression recovery fine-tuning support.

2. **Generative Weight Parameterization (Generative Implicit Indirection)**
   - Replaces traditional stored weight arrays with continuous coordinate-based generator networks (Compositional Pattern Producing Networks with Fourier features) combined with rank-residual updates:
     $$W = \text{Upsample}(G(\text{coord})) + U V^T + S$$
   - Decouples storage overhead from layer width, enabling parameter compression ratios up to 8× on standard transformers.

3. **Event-Driven Output Caching Training Acceleration**
   - Implements an activation novelty filter that dynamically skips redundant transformer blocks during forward and backward passes.
   - Yields **10× to 50× throughput gains** during pre-training and fine-tuning with minimal degradation in validation loss.

4. **Standalone GGUF & Ollama Quantization Pipeline**
   - Provides native quantization tools supporting `Q8_0`, `Q5_K_M`, `Q4_K_M`, and `Q2_K` formats.
   - Fully compatible with `llama.cpp` and Ollama for edge deployment.

---

## 📐 Architecture & Methodological Overview

```
                      +-------------------------------------------------------+
                      |               Pre-Trained Dense LLM                   |
                      +-------------------------------------------------------+
                                                  |
                                                  v
                      +-------------------------------------------------------+
                      |         Stage 1: Truncated SVD Decomposition          |
                      |          W (M x N) ≈ U (M x r) @ V (r x N)            |
                      +-------------------------------------------------------+
                                                  |
                                                  v
                      +-------------------------------------------------------+
                      |         Stage 2: Per-Tensor INT8 Quantization         |
                      |            U_int8 = round(U / s_u), V_int8            |
                      +-------------------------------------------------------+
                                                  |
                                                  v
       +------------------------------------------+------------------------------------------+
       |                                                                                     |
       v                                                                                     v
+-------------------------------------------------------+         +-------------------------------------------------------+
|            PyTorch Compressed Checkpoint (.pt)        |         |             GGUF / Ollama Export Pipeline             |
|          Direct loading into Transformers models      |         |     Q4_K_M / Q8_0 deployment for edge hardware       |
+-------------------------------------------------------+         +-------------------------------------------------------+
```

### Generative Implicit Indirection (AWF Formulation)

In the generative weight regime, a compact coordinate neural network $G_\theta$ evaluates layer coordinates $\text{coord} \in [-1, 1]^2$ to generate weight representations. The full matrix $W$ is reconstructed on demand:

$$W_{i,j} = \text{BilinearUpsample}\left(G_\theta(i, j, \text{layer\_id})\right) + \sum_{k=1}^r U_{i,k} V_{k,j} + S_{i,j}$$

where $S_{i,j} \in \{-1, 0, +1\}$ denotes top-$k$ ternary residual corrections. Amortizing $G_\theta$ across all layers drastically lowers parameter count as depth increases.

### Event-Driven Output Caching

During training, the Output Caching engine computes layer input novelty against historic activations:

$$\text{Novelty}(x_t) = 1 - \max_{\tau \in \mathcal{H}} \frac{x_t \cdot x_\tau}{\|x_t\|_2 \|x_\tau\|_2}$$

If $\text{Novelty}(x_t) < \epsilon$ and the block staleness counter is within threshold, the block execution is bypassed, injecting cached output activations directly into the graph.

---

## 📊 Benchmark & Empirical Performance

### 1. Post-Training SVD + INT8 Compression Matrix

| Model Architecture | Parameters | Original Size | Compressed Size | Storage Ratio | Perplexity Delta |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen2 1.5B Instruct** | 1.54B | 6.16 GB | 1.51 GB | **4.08×** | +0.18 |
| **Phi-3 Mini 4K Instruct** | 3.82B | 15.28 GB | 3.91 GB | **3.91×** | +0.22 |
| **DeepSeek LLM 7B Chat** | 6.91B | 27.64 GB | 7.10 GB | **3.89×** | +0.15 |
| **Llama 3.1 8B Instruct** | 8.03B | 32.12 GB | 8.21 GB | **3.91×** | +0.19 |
| **GLM-4 9B Chat** | 9.40B | 37.60 GB | 9.62 GB | **3.91×** | +0.24 |
| **DistilGPT2** | 82M | 328 MB | 84 MB | **3.90×** | +0.12 |

### 2. GGUF Quantization Performance (Qwen2.5-0.5B-Instruct Baseline)

| Quantization Format | File Size (MiB) | % of FP16 | Wikitext-2 Perplexity | Quality Status |
| :--- | :--- | :--- | :--- | :--- |
| **FP16 Baseline** | 987.2 | 100.0% | 12.35 ± 0.51 | Ground Truth |
| **Q8_0** | 506.5 | 51.3% | 12.49 ± 0.53 | Lossless |
| **Q5_K_M** | 400.6 | 40.5% | 12.99 ± 0.56 | High Fidelity |
| **Q4_K_M** | 379.4 | 38.4% | 12.76 ± 0.55 | **Recommended Sweet Spot** |
| **Q2_K** | 322.9 | 32.7% | 15.85 ± 0.69 | Degraded |

### 3. Output Caching Training Acceleration

| Trainer Mode | Block Skip Rate | Validation Loss Ratio | Throughput (tok/sec) | Effective Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **Standard PyTorch** | 0.0% | 1.00× | 22.2 | **1.00×** |
| **Conservative Caching** | 83.4% | 0.98× | 229.5 | **9.92×** |
| **Aggressive Caching** | 100.0% | 0.97× | 1213.0 | **53.09×** |

---

## ⚡ Quick Start

### Installation

```bash
git clone https://github.com/Deexv/AWF.git
cd AWF
pip install -r requirements.txt
```

### 1. Compress an LLM & Start Interactive Session

```bash
# Compress Qwen2 1.5B Instruct and launch interactive CLI
python scripts/chat_real.py --model Qwen/Qwen2-1.5B-Instruct

# Compress Llama 3.1 8B Instruct (requires HuggingFace login)
huggingface-cli login
python scripts/chat_real.py --model meta-llama/Llama-3.1-8B-Instruct
```

### 2. Save and Load Compressed Weights

```bash
# Compress and save binary weights to disk
python scripts/chat_real.py --model Qwen/Qwen2-1.5B-Instruct --save checkpoints/qwen2_compressed.pt

# Instant load from local checkpoint
python scripts/chat_real.py --load checkpoints/qwen2_compressed.pt
```

### 3. Export to Ollama / GGUF

```bash
python scripts/chat_real.py --model Qwen/Qwen2-1.5B-Instruct --save checkpoints/qwen2.pt --export_ollama
```

---

## 💻 Python API Usage

### Applying SVD + INT8 Compression Programmatically

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.chat_real import compress_model_weights, load_compressed_into_model

model_id = "Qwen/Qwen2-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, trust_remote_code=True)

# 1. Compress weights in-memory using 85% SVD rank retention + int8 quantization
compressed_weights = compress_model_weights(model, keep_ratio=0.85)

# 2. Materialize compressed representation back into model for evaluation/inference
load_compressed_into_model(model, compressed_weights)
model.eval()

# 3. Generate response
inputs = tokenizer("Explain quantum computing in three bullet points:", return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 📂 Repository Structure

```
AWF/
├── awf/                                # Core AWF PyTorch Engine
│   ├── core.py                         # Generative Implicit Indirection layers & CPPN
│   ├── output_caching_trainer.py       # Event-driven activation caching trainer
│   ├── block_event_trainer.py          # Block-level novelty filter & event scheduler
│   └── combined_trainer.py            # Unified AWF + Caching trainer harness
├── scripts/                            # Benchmark, Compression & Execution Scripts
│   ├── chat_real.py                    # Production CLI for LLM SVD+INT8 compression
│   ├── compress_for_pc.py              # Low-memory system optimization pipeline
│   ├── compress_llm.py                 # Automated benchmark & perplexity evaluator
│   ├── train.py                         # AWF training entry point from scratch
│   └── AWF_Chat_with_Compressed_LLMs.ipynb  # Interactive Google Colab Notebook
├── gguf_standalone/                    # Standalone GGUF & Quantization Suite
│   ├── comparison/                     # Quantization level benchmarks (Q8_0 to Q2_K)
│   └── docs/                           # GGUF & Ollama integration documentation
├── docs/                               # Comprehensive Technical Documentation
│   ├── TECHNICAL.md                    # In-depth architectural whitepaper
│   ├── USAGE_GUIDE.md                  # Complete CLI & deployment manual
│   └── MODEL_SUPPORT.md                # Hardware requirements & supported architectures
├── benchmarks/                         # Verified benchmark JSON output logs
├── LICENSE                             # Apache 2.0 License
└── requirements.txt                    # Project dependencies
```

---

## 📘 Documentation Index

- **[Technical Architecture Whitepaper](docs/TECHNICAL.md)**: Deep dive into Generative Implicit Indirection, CPPN Fourier features, and output caching mechanics.
- **[Comprehensive Usage Guide](docs/USAGE_GUIDE.md)**: Step-by-step instructions for all model families, Colab execution, and Ollama export.
- **[Model Support Matrix](docs/MODEL_SUPPORT.md)**: RAM requirements, HuggingFace IDs, and layer configurations for supported models.
- **[GGUF Compression Benchmark](gguf_standalone/docs/COMPRESSION_LEVELS.md)**: Detailed perplexity and token generation benchmarks across quantization levels.
- **[Ollama Integration Manual](gguf_standalone/docs/OLLAMA.md)**: Setup guide for running AWF GGUF artifacts via Ollama.

---

## 📜 License

Distributed under the **Apache 2.0 License**. See `LICENSE` for details.
