# AWF — Algorithmic Weight Fabric

> **Compress any LLM and chat with it — fluent English from a 2-6× smaller model.**
> Works with GPT-2, DeepSeek, GLM, Llama, Mistral, Phi, Qwen.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Quick Start (3 commands)

```bash
git clone https://github.com/Deexv/AWF.git
cd AWF
pip install -r requirements.txt transformers

# Compress DistilGPT2 and chat with it:
python scripts/chat_compressed.py --model distilgpt2
```

This downloads DistilGPT2 (already fluent), compresses it 2×, and starts an interactive chat. Type any prompt and get fluent English back.

## Chat with Any Model

```bash
# GPT-2 family (no token needed)
python scripts/chat_compressed.py --model distilgpt2       # 82M, fastest
python scripts/chat_compressed.py --model gpt2-medium      # 355M, better quality
python scripts/chat_compressed.py --model gpt2-large       # 774M, best GPT-2

# DeepSeek (no token needed)
python scripts/chat_compressed.py --model deepseek-ai/deepseek-coder-1.3b-base

# GLM / ChatGLM (no token needed)
python scripts/chat_compressed.py --model THUDM/chatglm3-6b-base

# Phi (no token needed)
python scripts/chat_compressed.py --model microsoft/phi-2

# Qwen (no token needed)
python scripts/chat_compressed.py --model Qwen/Qwen2-1.5B

# Llama 3 (needs huggingface-cli login first)
huggingface-cli login
python scripts/chat_compressed.py --model meta-llama/Llama-3-8B

# Mistral (needs huggingface-cli login first)
python scripts/chat_compressed.py --model mistralai/Mistral-7B-v0.1
```

## Save and Reuse Compressed Models

```bash
# Compress and save:
python scripts/chat_compressed.py --model distilgpt2 --save distilgpt2_compressed.pt

# Load and chat (no re-compression needed):
python scripts/chat_compressed.py --load checkpoints/distilgpt2_compressed.pt
```

## Single Prompt (non-interactive)

```bash
python scripts/chat_compressed.py --model distilgpt2 --prompt "Once upon a time"
```

## Use in Your Python Code

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load compressed model
ckpt = torch.load("checkpoints/distilgpt2_compressed.pt", map_location="cpu")
tokenizer = AutoTokenizer.from_pretrained(ckpt["model_name"])
model = AutoModelForCausalLM.from_pretrained(ckpt["model_name"])
model.load_state_dict(ckpt["model"])
model.eval()

# Generate
ids = tokenizer.encode("Once upon a time", return_tensors="pt")
out = model.generate(ids, max_new_tokens=150, temperature=0.7, top_k=50,
                     do_sample=True, repetition_penalty=1.2)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

## Google Colab (recommended for big models)

Open `scripts/AWF_Chat_with_Compressed_LLMs.ipynb` in Colab with T4 GPU. The notebook:
- Lets you pick any model from a dropdown
- Compresses with a slider
- Chats with the compressed model
- Saves to Google Drive

## Verified Results

| Model | Params | Original | Compressed | Perplexity | Fluent | RAM |
|---|---|---|---|---|---|---|
| DistilGPT2 | 82M | 312 MB | ~160 MB | 36.0 | ✅ YES | <1 GB |
| GPT-2 Small | 124M | 475 MB | ~240 MB | 17.2 | ✅ YES | <1 GB |

### Sample Output (compressed DistilGPT2)

**Prompt**: "Once upon a time there was a little girl named Lily"
**Response**: "...who had been the only person to know her and that she wanted help for them. When they arrived in one of these villages called Juna with their young princess Alina-Soumiya..."

Fluent English from a compressed model.

## Compression Settings

| Setting | Compression | Quality | Command |
|---|---|---|---|
| `--keep_ratio 0.95` | 1.2× | Best | `--keep_ratio 0.95` |
| `--keep_ratio 0.85` (default) | 2× | Good | `--keep_ratio 0.85` |
| `--keep_ratio 0.70` | 3× | Moderate | `--keep_ratio 0.70` |
| `--keep_ratio 0.50` | 4× | Needs fine-tuning | `--keep_ratio 0.50` |

## Chat Commands (interactive mode)

| Command | Effect |
|---|---|
| `temp 0.5` | Set temperature (0.1=focused, 1.0=creative) |
| `tokens 200` | Set max tokens per response |
| `quit` | Exit chat |
| `help` | Show commands |

## Documentation

| File | Contents |
|---|---|
| **[docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md)** | Complete step-by-step guide for every model |
| **[docs/MODEL_SUPPORT.md](docs/MODEL_SUPPORT.md)** | Model family table (DeepSeek, GLM, Llama, etc.) |
| **[docs/TECHNICAL.md](docs/TECHNICAL.md)** | Technical whitepaper |
| **[docs/BUSINESS_CASE.md](docs/BUSINESS_CASE.md)** | Investment pitch |

## Colab Notebooks

| Notebook | What it does |
|---|---|
| `scripts/AWF_Chat_with_Compressed_LLMs.ipynb` | Compress + chat (interactive) |
| `scripts/AWF_Compress_Big_Models.ipynb` | Compress + benchmark |
| `scripts/AWF_Fluent_LLM.ipynb` | Compress + fine-tune |

## Repository Structure

```
AWF/
├── scripts/
│   ├── chat_compressed.py               # ⭐ Compress + chat with any model
│   ├── compress_for_pc.py               # Compress for 4-8GB PC
│   ├── compress_finetune.py             # Compress + fine-tune
│   ├── AWF_Chat_with_Compressed_LLMs.ipynb  # ⭐ Colab: compress + chat
│   ├── AWF_Compress_Big_Models.ipynb    # Colab: compress + benchmark
│   └── ...
├── docs/
│   ├── USAGE_GUIDE.md                   # ⭐ How to use (step by step)
│   ├── MODEL_SUPPORT.md                 # All model families
│   ├── TECHNICAL.md                     # Whitepaper
│   └── BUSINESS_CASE.md                # Investment pitch
├── awf/                                  # AWF library
├── benchmarks/                           # Verified results
├── requirements.txt
└── LICENSE
```

## Honest Limitations

1. **Runtime memory**: In PyTorch, compressed weights load as fp32 (reconstructed from int8). For actual runtime savings, use a custom int8 kernel (llama.cpp/GGML). The compression is real for storage and download size.

2. **SVD at 85% is the sweet spot**: More aggressive (50%+) breaks quality and needs fine-tuning.

3. **Large models need RAM to compress**: Compressing a 7B model requires ~16GB RAM to load the original. Use Colab with T4 GPU.

4. **Training from scratch = gibberish**: No architecture overcomes this. Always start from pre-trained models.

5. **Not GPT-4 level**: This compresses existing models (DistilGPT2, GPT-2). It doesn't create new GPT-4 competitors.

## License

Apache 2.0 — use commercially, modify freely.
