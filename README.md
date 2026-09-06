# AWF — Algorithmic Weight Fabric

> **Compress any LLM 2-4× and chat with it. REAL file compression. Instruct models for real chat.**
> Works with Qwen, GLM-4, DeepSeek, Llama, Mistral, Phi. Export to Ollama.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Quick Start (3 commands)

```bash
git clone https://github.com/Deexv/AWF.git
cd AWF
pip install -r requirements.txt transformers

# List supported models:
python scripts/chat_real.py --list

# Compress Qwen2 Instruct and chat (RECOMMENDED — real chat):
python scripts/chat_real.py --model Qwen/Qwen2-1.5B-Instruct
```

## Supported Models

| Model ID | Type | Params | Chat? | Token Needed? |
|---|---|---|---|---|
| `Qwen/Qwen2-1.5B-Instruct` | Qwen2 Instruct | 1.5B | ✅ YES | No |
| `microsoft/Phi-3-mini-4k-instruct` | Phi-3 Instruct | 3.8B | ✅ YES | No |
| `THUDM/glm-4-9b-chat` | GLM-4 Chat | 9B | ✅ YES | No |
| `deepseek-ai/deepseek-llm-7b-chat` | DeepSeek Chat | 7B | ✅ YES | No |
| `Qwen/Qwen2-7B-Instruct` | Qwen2 7B Instruct | 7B | ✅ YES | No |
| `mistralai/Mistral-7B-Instruct-v0.3` | Mistral Instruct | 7B | ✅ YES | Yes |
| `meta-llama/Llama-3.1-8B-Instruct` | Llama 3.1 Instruct | 8B | ✅ YES | Yes |
| `distilgpt2` | DistilGPT2 (base) | 82M | ❌ Text only | No |
| `gpt2` | GPT-2 (base) | 124M | ❌ Text only | No |

**For real chat, use Instruct models** (the ones marked ✅ YES).
Base models (distilgpt2, gpt2) only do text continuation — they won't respond to questions.

## How to Use

### Compress and Chat

```bash
# Qwen2 1.5B Instruct (small, fast, good chat) — RECOMMENDED
python scripts/chat_real.py --model Qwen/Qwen2-1.5B-Instruct

# GLM-4-9B-Chat (Chinese + English)
python scripts/chat_real.py --model THUDM/glm-4-9b-chat

# DeepSeek 7B Chat
python scripts/chat_real.py --model deepseek-ai/deepseek-llm-7b-chat

# Phi-3 Mini Instruct
python scripts/chat_real.py --model microsoft/Phi-3-mini-4k-instruct

# Mistral 7B Instruct (needs HF token)
huggingface-cli login
python scripts/chat_real.py --model mistralai/Mistral-7B-Instruct-v0.3

# Llama 3.1 8B Instruct (needs HF token)
python scripts/chat_real.py --model meta-llama/Llama-3.1-8B-Instruct
```

### Save Compressed Model

```bash
python scripts/chat_real.py --model Qwen/Qwen2-1.5B-Instruct --save qwen_compressed.pt
```

The saved file is **genuinely smaller** (not fake compression):
- Original: ~6 GB (fp32)
- Compressed: ~1.5 GB (int8 SVD factors)
- **Actual 2-4× file size reduction**

### Load Compressed Model

```bash
python scripts/chat_real.py --load checkpoints/qwen_compressed.pt
```

No re-compression needed — loads instantly.

### Single Prompt

```bash
python scripts/chat_real.py --model Qwen/Qwen2-1.5B-Instruct --prompt "What is the capital of France?"
```

### Export for Ollama

```bash
python scripts/chat_real.py --model Qwen/Qwen2-1.5B-Instruct --save qwen.pt --export_ollama
```

This creates an `ollama_model/` directory with the model + Modelfile.
See [Ollama Guide](docs/USAGE_GUIDE.md#ollama-export) for full instructions.

### Use in Python

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load compressed model
ckpt = torch.load("checkpoints/qwen_compressed.pt", weights_only=False)
model_name = ckpt["model_name"]
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# Load compressed weights
from chat_real import load_compressed_into_model
load_compressed_into_model(model, ckpt["compressed_weights"])
model.eval()

# Chat
messages = [{"role": "user", "content": "What is the capital of France?"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer.encode(text, return_tensors="pt")
output = model.generate(input_ids, max_new_tokens=200, temperature=0.7, top_k=50,
                       do_sample=True, repetition_penalty=1.2)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Google Colab

Open `scripts/AWF_Chat_with_Compressed_LLMs.ipynb` in Colab with T4 GPU.
The notebook:
- Lists all supported models
- Compresses with a slider
- Chats with proper instruct templates
- Exports for Ollama
- Saves to Google Drive

## How Compression Works

```
Pre-trained LLM (fluent, big)
       ↓
Stage 1: SVD Decomposition
  For each weight matrix W (M×N):
    W ≈ U @ V where U: (M, r), V: (r, N)
    Keep top 85% of singular values
       ↓
Stage 2: INT8 Quantization
  U and V stored as int8 codes (1 byte) instead of fp32 (4 bytes)
  Per-tensor scale factor preserves magnitude
       ↓
Saved file: SVD factors as int8 = 2-4× smaller than original
```

### Verified Results

| Model | Original | Compressed File | Compression | Quality |
|---|---|---|---|---|
| DistilGPT2 | 312 MB | 112 MB | 2.8× | Fluent (base model) |

## Documentation

| File | Contents |
|---|---|
| **[docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md)** | Complete step-by-step for every model + Ollama |
| **[docs/MODEL_SUPPORT.md](docs/MODEL_SUPPORT.md)** | All model families with RAM estimates |
| **[docs/TECHNICAL.md](docs/TECHNICAL.md)** | Architecture whitepaper |
| **[docs/BUSINESS_CASE.md](docs/BUSINESS_CASE.md)** | Investment pitch |

## Repository Structure

```
AWF/
├── scripts/
│   ├── chat_real.py                         # ⭐ MAIN: Compress + chat + Ollama export
│   ├── compress_for_pc.py                   # Compress for 4-8GB PC
│   ├── compress_llm.py                      # Compress + benchmark
│   ├── download_tinystories.py              # Dataset downloader
│   ├── train.py                              # Train AWF from scratch
│   ├── train_fluent.py                      # Train fluent AWF
│   ├── AWF_Chat_with_Compressed_LLMs.ipynb  # ⭐ Colab notebook
│   ├── AWF_Compress_Big_Models.ipynb        # Colab benchmark
│   └── archive/                              # Old scripts (research)
├── awf/                                      # AWF library
├── docs/                                     # Documentation
├── benchmarks/                               # Results
├── requirements.txt
└── LICENSE
```

## Chat Commands (interactive mode)

| Command | Effect |
|---|---|
| `temp 0.5` | Temperature (0.1=focused, 1.0=creative) |
| `tokens 200` | Max tokens per response |
| `quit` | Exit |
| `help` | Show commands |

## Honest Limitations

1. **Instruct models give real chat. Base models give text continuation.** Always use `*-Instruct` or `*-chat` models.
2. **Compression at 85% SVD + int8** is the sweet spot. More aggressive needs fine-tuning.
3. **Large models need RAM to compress.** 7B+ models: use Colab with T4 GPU.
4. **Ollama export needs llama.cpp** for GGUF conversion. The Modelfile is generated but GGUF conversion requires `llama-cpp-python`.
5. **Runtime memory in PyTorch** still loads as fp32 (reconstructed). For actual runtime savings, use Ollama/llama.cpp with GGUF format.

## License

Apache 2.0 — use commercially, modify freely.
