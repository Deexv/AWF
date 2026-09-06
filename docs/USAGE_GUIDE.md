# AWF Usage Guide — How to Compress and Chat with Any LLM

## Overview

AWF lets you compress any pre-trained LLM and chat with it. The compressed model is fluent and runs on 4-8GB RAM PCs.

## Quick Start (3 commands)

```bash
git clone https://github.com/Deexv/AWF.git
cd AWF
pip install -r requirements.txt transformers

# Compress DistilGPT2 and chat with it:
python scripts/chat_compressed.py --model distilgpt2
```

That's it. The script will:
1. Download DistilGPT2 (82M params, already fluent)
2. Compress it ~2× with AWF (SVD + INT8)
3. Start an interactive chat

## How to Use Each Model

### GPT-2 Family (easiest, no token needed)

```bash
# DistilGPT2 (82M, fastest, <1GB RAM)
python scripts/chat_compressed.py --model distilgpt2

# GPT-2 Small (124M, better quality)
python scripts/chat_compressed.py --model gpt2

# GPT-2 Medium (355M, even better)
python scripts/chat_compressed.py --model gpt2-medium

# GPT-2 Large (774M, needs 8GB RAM)
python scripts/chat_compressed.py --model gpt2-large
```

### DeepSeek Family

```bash
# DeepSeek Coder 1.3B (code generation, <1GB compressed)
python scripts/chat_compressed.py --model deepseek-ai/deepseek-coder-1.3b-base

# DeepSeek LLM 7B (general purpose, needs Colab to compress)
# Run on Colab: python scripts/chat_compressed.py --model deepseek-ai/deepseek-llm-7b-base
```

### GLM Family (ZhipuAI / ChatGLM)

```bash
# ChatGLM3-6B (Chinese + English)
python scripts/chat_compressed.py --model THUDM/chatglm3-6b-base

# GLM-4-9B (latest, needs Colab to compress)
# Run on Colab: python scripts/chat_compressed.py --model THUDM/glm-4-9b-chat
```

### Llama 3 (needs HuggingFace token)

```bash
# Step 1: Get token from https://huggingface.co/settings/tokens
huggingface-cli login

# Step 2: Compress and chat
python scripts/chat_compressed.py --model meta-llama/Llama-3-8B
```

### Mistral (needs HuggingFace token)

```bash
huggingface-cli login
python scripts/chat_compressed.py --model mistralai/Mistral-7B-v0.1
```

### Phi (Microsoft, no token needed)

```bash
python scripts/chat_compressed.py --model microsoft/phi-2
```

### Qwen (Alibaba, no token needed)

```bash
python scripts/chat_compressed.py --model Qwen/Qwen2-1.5B
```

## Saving and Reusing Compressed Models

### Save a compressed model:

```bash
python scripts/chat_compressed.py --model distilgpt2 --save distilgpt2_compressed.pt
```

This saves to `checkpoints/distilgpt2_compressed.pt`. You can now load it without re-compressing.

### Load a saved compressed model:

```bash
python scripts/chat_compressed.py --load checkpoints/distilgpt2_compressed.pt
```

This loads the compressed weights and starts chatting immediately — no download, no compression.

### Use in your own Python code:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the compressed model
checkpoint = torch.load("checkpoints/distilgpt2_compressed.pt", map_location="cpu")
model_name = checkpoint["model_name"]
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.load_state_dict(checkpoint["model"])
model.eval()

# Generate text
prompt = "Once upon a time there was a little girl named Lily"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(input_ids, max_new_tokens=150, temperature=0.7,
                        top_k=50, do_sample=True, repetition_penalty=1.2)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Single Prompt (non-interactive)

```bash
python scripts/chat_compressed.py --model distilgpt2 --prompt "Once upon a time"
```

## Adjusting Generation Quality

```bash
# More focused/conservative (good for factual text)
python scripts/chat_compressed.py --model distilgpt2 --temperature 0.3 --top_k 30

# More creative/random (good for stories)
python scripts/chat_compressed.py --model distilgpt2 --temperature 0.9 --top_k 80

# More tokens per response
python scripts/chat_compressed.py --model distilgpt2 --tokens 300
```

## Chat Commands (in interactive mode)

Once chatting, type these commands:

| Command | Effect |
|---|---|
| `temp 0.5` | Set temperature (0.1=focused, 1.0=creative) |
| `tokens 200` | Set max tokens per response |
| `quit` | Exit chat |
| `help` | Show available commands |

## Using Google Colab (recommended for big models)

1. Open `scripts/AWF_Chat_with_Compressed_LLMs.ipynb` in Colab
2. Set Runtime → T4 GPU
3. Run all cells

The notebook lets you:
- Pick any model from a dropdown
- Compress it with a slider for compression ratio
- Chat with the compressed model
- Save to Google Drive for reuse

## Running Without Compression (for comparison)

```bash
# Chat with original model (no compression)
python scripts/chat_compressed.py --model distilgpt2 --no_compress
```

## Compression Settings

| Setting | Compression | Quality | Use When |
|---|---|---|---|
| `--keep_ratio 0.95` | 1.2× | Best | Quality is critical |
| `--keep_ratio 0.85` (default) | 2× | Good | **Recommended** |
| `--keep_ratio 0.70` | 3× | Moderate | RAM is tight |
| `--keep_ratio 0.50` | 4× | Needs fine-tuning | Maximum compression |

## Troubleshooting

### "CUDA out of memory"
- Use a smaller model (`distilgpt2` instead of `gpt2-large`)
- Run on Colab with T4 GPU (16GB RAM)
- Use `--keep_ratio 0.50` for more compression

### "Model generates gibberish"
- Use `--keep_ratio 0.95` (less compression, better quality)
- The pre-trained model is fluent — if output is gibberish, compression was too aggressive
- For 50%+ compression, fine-tune after: `python scripts/compress_finetune.py`

### "Model not found"
- Check the model ID at https://huggingface.co/models
- Some models need `huggingface-cli login` (Llama, Mistral)
- Use `--trust_remote_code` for GLM, DeepSeek

### "Generation is slow"
- Use GPU (Colab T4 is free)
- Use a smaller model
- Reduce `--tokens` (fewer tokens per response)
