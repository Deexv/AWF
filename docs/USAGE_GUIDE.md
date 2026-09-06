# AWF Usage Guide

## Quick Start

```bash
git clone https://github.com/Deexv/AWF.git
cd AWF
pip install -r requirements.txt transformers

# List supported models:
python scripts/chat_real.py --list

# Compress Qwen2 Instruct and chat:
python scripts/chat_real.py --model Qwen/Qwen2-1.5B-Instruct
```

## Supported Models (All Instruct-Tuned for Real Chat)

### Small Models (fit 4GB RAM, fast on CPU)

| Model | Command | Params | Compressed | Token? |
|---|---|---|---|---|
| Qwen2 1.5B Instruct | `--model Qwen/Qwen2-1.5B-Instruct` | 1.5B | ~400 MB | No |
| DistilGPT2 (base) | `--model distilgpt2` | 82M | ~112 MB | No |

### Medium Models (need 8GB RAM or Colab)

| Model | Command | Params | Compressed | Token? |
|---|---|---|---|---|
| Phi-3 Mini Instruct | `--model microsoft/Phi-3-mini-4k-instruct` | 3.8B | ~1 GB | No |
| GLM-4-9B-Chat | `--model THUDM/glm-4-9b-chat` | 9B | ~2.5 GB | No |
| DeepSeek 7B Chat | `--model deepseek-ai/deepseek-llm-7b-chat` | 7B | ~2 GB | No |
| Qwen2 7B Instruct | `--model Qwen/Qwen2-7B-Instruct` | 7B | ~2 GB | No |
| Mistral 7B Instruct | `--model mistralai/Mistral-7B-Instruct-v0.3` | 7B | ~2 GB | Yes |
| Llama 3.1 8B Instruct | `--model meta-llama/Llama-3.1-8B-Instruct` | 8B | ~2.2 GB | Yes |

## Commands

### Compress and Chat
```bash
python scripts/chat_real.py --model Qwen/Qwen2-1.5B-Instruct
```

### Compress and Save
```bash
python scripts/chat_real.py --model Qwen/Qwen2-1.5B-Instruct --save qwen_compressed.pt
```

### Load Saved Compressed Model
```bash
python scripts/chat_real.py --load checkpoints/qwen_compressed.pt
```

### Single Prompt
```bash
python scripts/chat_real.py --model Qwen/Qwen2-1.5B-Instruct --prompt "What is the capital of France?"
```

### Export for Ollama
```bash
python scripts/chat_real.py --model Qwen/Qwen2-1.5B-Instruct --save qwen.pt --export_ollama
```

### List All Models
```bash
python scripts/chat_real.py --list
```

## GLM-4-9B-Chat

GLM-4 is ZhipuAI's latest model. Supports Chinese + English chat.

```bash
# Compress and chat with GLM-4
python scripts/chat_real.py --model THUDM/glm-4-9b-chat

# Save compressed GLM-4
python scripts/chat_real.py --model THUDM/glm-4-9b-chat --save glm4_compressed.pt

# Load and chat
python scripts/chat_real.py --load checkpoints/glm4_compressed.pt
```

**Note**: GLM-4-9B needs ~36 GB RAM to load the original model. Use Colab with A100 GPU.

## DeepSeek

```bash
# DeepSeek 7B Chat (general chat)
python scripts/chat_real.py --model deepseek-ai/deepseek-llm-7b-chat

# DeepSeek Coder 1.3B (code generation, smaller)
python scripts/chat_real.py --model deepseek-ai/deepseek-coder-1.3b-instruct
```

## Ollama Export

AWF can export compressed models for use with [Ollama](https://ollama.ai):

```bash
# Compress + save + export for Ollama
python scripts/chat_real.py --model Qwen/Qwen2-1.5B-Instruct --save qwen.pt --export_ollama
```

This creates:
- `ollama_Qwen2-1.5B-Instruct/` directory with model + tokenizer
- `Modelfile` configured for the model

### Using with Ollama

After export:

```bash
# 1. Install Ollama: https://ollama.ai
# 2. Install llama.cpp for GGUF conversion:
pip install llama-cpp-python

# 3. Convert to GGUF:
python -m llama_cpp.convert ollama_Qwen2-1.5B-Instruct --outtype q8_0 --outfile ollama_Qwen2-1.5B-Instruct/model.gguf

# 4. Update Modelfile to point to GGUF:
echo "FROM ollama_Qwen2-1.5B-Instruct/model.gguf" > ollama_Qwen2-1.5B-Instruct/Modelfile
echo "PARAMETER temperature 0.7" >> ollama_Qwen2-1.5B-Instruct/Modelfile
echo "PARAMETER top_k 50" >> ollama_Qwen2-1.5B-Instruct/Modelfile
echo "SYSTEM You are a helpful AI assistant." >> ollama_Qwen2-1.5B-Instruct/Modelfile

# 5. Create Ollama model:
ollama create qwen2-awf -f ollama_Qwen2-1.5B-Instruct/Modelfile

# 6. Run:
ollama run qwen2-awf
```

### Alternative: Use without GGUF

If you don't want to convert to GGUF, you can use the compressed model directly in Python:

```bash
python scripts/chat_real.py --load checkpoints/qwen_compressed.pt
```

## Chat Commands (Interactive Mode)

| Command | Effect |
|---|---|
| `temp 0.5` | Set temperature (0.1=focused, 1.0=creative) |
| `tokens 200` | Set max tokens per response |
| `quit` | Exit chat |
| `help` | Show commands |

## Using in Your Python Code

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys; sys.path.insert(0, 'scripts')
from chat_real import load_compressed_into_model

# Load compressed model
ckpt = torch.load("checkpoints/qwen_compressed.pt", weights_only=False)
model_name = ckpt["model_name"]
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
load_compressed_into_model(model, ckpt["compressed_weights"])
model.eval()

# Chat using instruct template
messages = [{"role": "user", "content": "Write a poem about the ocean."}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer.encode(text, return_tensors="pt")

with torch.no_grad():
    output = model.generate(input_ids, max_new_tokens=200,
                            temperature=0.7, top_k=50, do_sample=True,
                            repetition_penalty=1.2)

response = tokenizer.decode(output[0], skip_special_tokens=True)
# Extract assistant response
if "assistant" in response:
    response = response.split("assistant")[-1].strip()
print(response)
```

## Google Colab

Open `scripts/AWF_Chat_with_Compressed_LLMs.ipynb` in Colab with T4 GPU.

The notebook:
1. Lists all supported models
2. Compresses with a slider for compression ratio
3. Chats with proper instruct templates
4. Exports for Ollama
5. Saves to Google Drive

## Troubleshooting

### "Model generates gibberish"
- Use **Instruct** models (`Qwen2-1.5B-Instruct`), not base models (`Qwen2-1.5B`)
- Base models do text continuation; instruct models do Q&A
- Run `--list` to see which models are instruct

### "Compressed file is same size as original"
- You're using the old script. Use `chat_real.py` (not `chat_compressed.py`)
- The old script saved reconstructed fp32 weights (no savings)
- `chat_real.py` stores int8 SVD factors (actual savings)

### "CUDA out of memory"
- Use smaller model (`Qwen/Qwen2-1.5B-Instruct` instead of 7B)
- Run on Colab with T4 GPU (16GB RAM)
- Use `--keep_ratio 0.50` for more compression

### "Model not found"
- Check model ID at https://huggingface.co/models
- Llama and Mistral need `huggingface-cli login`
- GLM and DeepSeek need `trust_remote_code=True` (handled automatically)
