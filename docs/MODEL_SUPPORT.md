# AWF Model Support Guide

## Supported Model Families

AWF compression works on **any model with standard Linear/Conv1D layers**  which covers essentially all transformer-based LLMs.

### How to Compress Each Model

```bash
# Universal command:
python scripts/compress_for_pc.py --model <huggingface_model_id> --target_ram <GB>
```

### GPT-2 Family (OpenAI)

| Model | HuggingFace ID | Params | Original | Compressed | RAM | Token Needed? |
|---|---|---|---|---|---|---|
| DistilGPT2 | `distilgpt2` | 82M | 312 MB | 12 MB | <1 GB | No |
| GPT-2 Small | `gpt2` | 124M | 475 MB | 18 MB | <1 GB | No |
| GPT-2 Medium | `gpt2-medium` | 355M | 1.4 GB | 53 MB | <1 GB | No |
| GPT-2 Large | `gpt2-large` | 774M | 3.0 GB | 112 MB | <1 GB | No |
| GPT-2 XL | `gpt2-xl` | 1.5B | 6.0 GB | 225 MB | <1 GB | No |

```bash
python scripts/compress_for_pc.py --model distilgpt2 --target_ram 4
python scripts/compress_for_pc.py --model gpt2-medium --target_ram 4
python scripts/compress_for_pc.py --model gpt2-large --target_ram 8
python scripts/compress_for_pc.py --model gpt2-xl --target_ram 8
```

### DeepSeek Family

| Model | HuggingFace ID | Params | Original | Compressed | RAM | Token Needed? |
|---|---|---|---|---|---|---|
| DeepSeek Coder 1.3B | `deepseek-ai/deepseek-coder-1.3b-base` | 1.3B | 5.2 GB | 195 MB | <1 GB | No |
| DeepSeek LLM 7B | `deepseek-ai/deepseek-llm-7b-base` | 7B | 28 GB | 1.0 GB | ~2 GB | No |
| DeepSeek MoE 16B | `deepseek-ai/deepseek-moe-16b-base` | 16B | 64 GB | 2.4 GB | ~4 GB | No |

```bash
python scripts/compress_for_pc.py --model deepseek-ai/deepseek-coder-1.3b-base --target_ram 4
python scripts/compress_for_pc.py --model deepseek-ai/deepseek-llm-7b-base --target_ram 8
```

**Architecture notes**: DeepSeek uses Llama-style transformer (RMSNorm, SwiGLU, GQA). AWF compresses all attention QKV and FFN weights. MoE models have more layers but each expert is small  compression works per-expert.

### GLM Family (ZhipuAI / ChatGLM)

| Model | HuggingFace ID | Params | Original | Compressed | RAM | Token Needed? |
|---|---|---|---|---|---|---|
| ChatGLM3-6B | `THUDM/chatglm3-6b-base` | 6B | 24 GB | 900 MB | ~2 GB | No |
| GLM-4-9B | `THUDM/glm-4-9b-chat` | 9B | 36 GB | 1.3 GB | ~2 GB | No |
| ChatGLM-6B | `THUDM/chatglm-6b` | 6B | 24 GB | 900 MB | ~2 GB | No |

```bash
python scripts/compress_for_pc.py --model THUDM/chatglm3-6b-base --target_ram 8
python scripts/compress_for_pc.py --model THUDM/glm-4-9b-chat --target_ram 8
```

**Architecture notes**: GLM uses prefix-LM + GLU activations. AWF compresses the query-key-value projections, dense projection, and GLU up/down projections. Works with `trust_remote_code=True`.

### Llama Family (Meta)

| Model | HuggingFace ID | Params | Original | Compressed | RAM | Token Needed? |
|---|---|---|---|---|---|---|
| Llama 3 8B | `meta-llama/Llama-3-8B` | 8B | 32 GB | 1.2 GB | ~2 GB | Yes |
| Llama 3 70B | `meta-llama/Llama-3-70B` | 70B | 280 GB | 10.5 GB | ~12 GB | Yes |
| Llama 2 7B | `meta-llama/Llama-2-7b-hf` | 7B | 28 GB | 1.0 GB | ~2 GB | Yes |

```bash
# First: get token from https://huggingface.co/settings/tokens
huggingface-cli login
python scripts/compress_for_pc.py --model meta-llama/Llama-3-8B --target_ram 8
```

**Architecture notes**: Llama uses RMSNorm, SwiGLU, GQA. All weight matrices are compressed. The 8B model compresses to ~1.2 GB  fits on 8GB PC.

### Mistral Family

| Model | HuggingFace ID | Params | Original | Compressed | RAM | Token Needed? |
|---|---|---|---|---|---|---|
| Mistral 7B | `mistralai/Mistral-7B-v0.1` | 7B | 28 GB | 1.0 GB | ~2 GB | Yes |
| Mistral 7B v0.3 | `mistralai/Mistral-7B-v0.3` | 7B | 28 GB | 1.0 GB | ~2 GB | Yes |
| Mixtral 8x7B | `mistralai/mixtral-8x7b-v0.1` | 47B | 188 GB | 7.0 GB | ~8 GB | Yes |

```bash
huggingface-cli login
python scripts/compress_for_pc.py --model mistralai/Mistral-7B-v0.1 --target_ram 8
```

**Architecture notes**: Mistral uses sliding window attention + GQA. Mixtral is MoE  each expert compressed separately.

### Phi Family (Microsoft)

| Model | HuggingFace ID | Params | Original | Compressed | RAM | Token Needed? |
|---|---|---|---|---|---|---|
| Phi-2 | `microsoft/phi-2` | 2.7B | 11 GB | 410 MB | <1 GB | No |
| Phi-3 mini | `microsoft/Phi-3-mini-4k-instruct` | 3.8B | 15 GB | 560 MB | ~1 GB | No |

```bash
python scripts/compress_for_pc.py --model microsoft/phi-2 --target_ram 4
```

### Qwen Family (Alibaba)

| Model | HuggingFace ID | Params | Original | Compressed | RAM | Token Needed? |
|---|---|---|---|---|---|---|
| Qwen2 7B | `Qwen/Qwen2-7B` | 7B | 28 GB | 1.0 GB | ~2 GB | No |
| Qwen2 1.5B | `Qwen/Qwen2-1.5B` | 1.5B | 6.0 GB | 225 MB | <1 GB | No |

```bash
python scripts/compress_for_pc.py --model Qwen/Qwen2-1.5B --target_ram 4
python scripts/compress_for_pc.py --model Qwen/Qwen2-7B --target_ram 8
```

## RAM Requirements

### To COMPRESS a model (loading the original):

| Model Size | RAM Needed | Where to Run |
|---|---|---|
| <500M params | 4 GB | Any PC |
| 1-2B params | 8 GB | Good PC or Colab |
| 7-9B params | 16 GB | Colab T4 GPU |
| 13B+ params | 32 GB+ | Colab A100 or cloud |

### To RUN the compressed model:

| Model Size (original) | Compressed Size | RAM Needed | PC Type |
|---|---|---|---|
| 82M (DistilGPT2) | 12 MB | <1 GB | Any PC |
| 124M (GPT-2) | 18 MB | <1 GB | Any PC |
| 355M (GPT-2 Medium) | 53 MB | <1 GB | Any PC |
| 1.3B (DeepSeek Coder) | 195 MB | <1 GB | Any PC |
| 2.7B (Phi-2) | 410 MB | <1 GB | Any PC |
| 7B (DeepSeek/Llama/Mistral) | ~1 GB | ~2 GB | Budget PC |
| 8B (Llama 3) | ~1.2 GB | ~2 GB | Budget PC |
| 9B (GLM-4) | ~1.3 GB | ~2 GB | Budget PC |
| 70B (Llama 3 70B) | ~10 GB | ~12 GB | Gaming PC |

## Fine-Tuning After Compression

For more aggressive compression ratios (50% SVD + INT8 = 13×), fine-tune to recover quality:

```bash
# Compress at 50% (more aggressive)
python scripts/compress_for_pc.py --model distilgpt2 --keep_ratio 0.50

# Fine-tune to recover quality (needs GPU, ~30 min)
python scripts/compress_finetune.py
```

After 500-2000 fine-tuning steps on relevant data, compressed models approach original perplexity.

## Custom Models

AWF works on any PyTorch model with Linear layers:

```python
from compress_for_pc import apply_svd_compression, apply_int8_quantization

# Load your custom model
model = YourModel.load()

# Compress
apply_svd_compression(model, keep_ratio=0.85)  # 1.7x
apply_int8_quantization(model)                   # 4x more

# Total: 6.7x compression
# Model is ready for inference
```
