# Using Compressed GGUF Models with Ollama

**Yes — the `.gguf` files this repo produces work directly with Ollama.** No conversion, no re-bundling, no extra steps.

Ollama uses llama.cpp under the hood, and GGUF is llama.cpp's native format. So any `.gguf` file you produce with the included Colab notebook (or `llama-quantize` directly) can be loaded into Ollama with a 3-line `Modelfile`.

---

## Quick start

### 1. Get a compressed `.gguf` file

Either:
- Run [`notebooks/compress_model_to_gguf_q4_k_m.ipynb`](../notebooks/compress_model_to_gguf_q4_k_m.ipynb) in Colab and download the resulting `.gguf`, or
- Use a pre-quantized one from [HuggingFace's GGUF models](https://huggingface.co/models?other=gguf).

### 2. Install Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: download from https://ollama.com/download
```

### 3. Create a Modelfile

Create a file named `Modelfile` (no extension) in the same directory as your `.gguf`:

```modelfile
FROM ./compressed/your-model-Q4_K_M.gguf

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER stop "</s>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
```

The `FROM` line is the only required line — it points to your compressed `.gguf`.
The `TEMPLATE` and `PARAMETER` blocks are optional; Ollama will fall back to
defaults baked into the GGUF metadata if you omit them.

> **Note for Qwen2.5 models**: The template above is the Qwen chat template.
> If you compressed a different model family (Llama, Mistral, Phi, etc.),
> use that model's chat template. The model card on HuggingFace will tell you.

### 4. Register the model with Ollama

```bash
ollama create my-compressed-model -f Modelfile
```

This registers the model in Ollama's local registry. It does NOT copy the `.gguf`
file — Ollama references it in place. If you move or delete the `.gguf`, you'll
need to re-run `ollama create`.

### 5. Run it

```bash
# Interactive chat
ollama run my-compressed-model

# One-shot
ollama run my-compressed-model "What is the capital of France?"

# Via the API (curl)
curl http://localhost:11434/api/generate -d '{
  "model": "my-compressed-model",
  "prompt": "Write a haiku about the ocean.",
  "stream": false
}'
```

### 6. Verify it's truly standalone

```bash
# After `ollama create`, Ollama copies the .gguf into its blobs/ dir.
ls ~/.ollama/models/blobs/

# Test from a totally empty directory:
cd /tmp
ollama run my-compressed-model "Hello!"
```

---

## Using with the AWF Companion

The PCCA Companion (`companion/`) now supports `.gguf` models directly via the `--gguf` flag:

```bash
# Direct GGUF (uses llama-cpp-python, no separate process)
python -m companion.demo --gguf gguf_standalone/compressed/your-model-Q4_K_M.gguf

# Or via Ollama (uses Ollama's HTTP API, requires `ollama serve`)
python -m companion.demo --ollama my-compressed-model
```

Memory persists across model swaps — you can switch from a template-mode companion to a GGUF-backed one without losing conversation history.

---

## Performance comparison: `llama-cpp-python` vs `ollama`

| Aspect              | `llama-cpp-python` (in-process) | `ollama` (separate server) |
|---------------------|----------------------------------|-----------------------------|
| Startup latency     | ~0.5s (model load)              | 0s after server is up         |
| Per-request overhead| Minimal (direct C call)          | HTTP round-trip (~5ms local) |
| Memory footprint    | In your process                 | Separate process             |
| Concurrency         | Single-threaded by default      | Built-in queueing            |
| GPU support         | Manual (`n_gpu_layers` param)   | Auto-detected                |
| Best for            | CLI tools, batch scripts        | Long-running services, web UIs |

For the 10K conversation test in this repo, we used `llama-cpp-python` directly
to minimize overhead. For production deployments (web UI, multi-user), Ollama is
usually the better choice.

---

## Troubleshooting

### "Error: model not found" after `ollama create`

The path in `FROM` must be relative to where you run `ollama create`, OR an absolute path:

```bash
# If your Modelfile says:
FROM ./compressed/my-model.gguf

# You must run `ollama create` from the directory CONTAINING `compressed/`:
cd /path/to/gguf_standalone
ollama create my-model -f Modelfile
```

### Garbled or empty responses

This means the chat template doesn't match the model. Common templates:

**Qwen2 / Qwen2.5:**
```modelfile
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
```

**Llama-3:**
```modelfile
TEMPLATE """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|><|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|start_header_id|>"
```

**Mistral:**
```modelfile
TEMPLATE """[INST] {{ if .System }}{{ .System }}

{{ end }}{{ .Prompt }} [/INST] {{ .Response }}"""
PARAMETER stop "[INST]"
PARAMETER stop "[/INST]"
```

If unsure, check the model card on HuggingFace — it lists the chat template in the "Chat template" section.

### Model runs slowly

If you have a GPU but it's not being used:

```bash
ollama ps                              # check what Ollama sees
# Force GPU usage in the Modelfile:
PARAMETER num_gpu 99
```

For `llama-cpp-python`, set `n_gpu_layers=99` when constructing the `Llama` object (requires building llama-cpp-python with CUDA or Metal support).

### "signal: killed" / OOM

The model needs RAM roughly equal to 1.5× the `.gguf` file size at runtime. For a 380 MiB GGUF, plan for ~600 MiB free RAM.

If hitting OOM, use a more aggressive quantization level — see [`docs/COMPRESSION_LEVELS.md`](COMPRESSION_LEVELS.md).

---

## Programmatic access (Python)

```python
import requests

response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "my-compressed-model",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        "stream": False,
    },
)
print(response.json()["message"]["content"])
```

Or use the official Ollama Python client:

```bash
pip install ollama
```

```python
import ollama

response = ollama.chat(
    model="my-compressed-model",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response["message"]["content"])
```

---

## Verifying the model is the compressed one

After `ollama create`, you can confirm Ollama is using your compressed file (not pulling the original from the Ollama registry):

```bash
ollama list                              # list registered models
ollama show my-compressed-model --modelfile  # show details
ls -lh compressed/your-model-Q4_K_M.gguf # your file
du -sh ~/.ollama/models/blobs/          # Ollama's copy
```

The blob size should roughly match your `.gguf` file size. If it's much larger
(e.g. ~1 GB for a 380 MiB GGUF), Ollama pulled the original model from the
registry instead of using your compressed file — check the `FROM` line.
