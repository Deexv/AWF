# Usage Guide

How to compress your own HuggingFace instruct model into a standalone `.gguf` file, and use it without ever touching the original model again.

---

## Option 1: Google Colab (zero setup)

1. Open [`notebooks/compress_model_to_gguf_q4_k_m.ipynb`](../notebooks/compress_model_to_gguf_q4_k_m.ipynb) in Google Colab (free CPU tier is fine).
2. In the **Configuration** cell, set:
   - `MODEL_ID` to your HuggingFace model ID (e.g. `meta-llama/Llama-3.2-1B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3`, `Qwen/Qwen2.5-7B-Instruct`, etc.)
   - `QUANT_TYPE` to `Q4_K_M` (default  best quality/size balance) or pick another from the dropdown.
3. Run all cells top-to-bottom. Total time on free Colab CPU:
   - 0.5B model: ~5 minutes
   - 1-3B model: ~10 minutes
   - 7B model: ~20 minutes
4. The last cell triggers a browser download of the `.gguf` file.

The notebook also includes an optional cell to upload the compressed file to your own HuggingFace repo.

---

## Option 2: Local build

### Step 1  Build llama.cpp once

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
pip install -r requirements/requirements-convert_hf_to_gguf.txt
mkdir build && cd build
cmake .. -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build . --target llama-quantize -j
```

### Step 2  Download the HF model

```bash
pip install huggingface_hub
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/Qwen2.5-0.5B-Instruct
```

### Step 3  Convert HF  F16 GGUF (lossless intermediate)

```bash
cd /path/to/llama.cpp
python convert_hf_to_gguf.py /path/to/models/Qwen2.5-0.5B-Instruct \
    --outfile model-F16.gguf --outtype f16
```

### Step 4  Quantize F16  Q4_K_M (the standalone compressed file)

```bash
./build/bin/llama-quantize model-F16.gguf model-Q4_K_M.gguf Q4_K_M
```

### Step 5  Delete the originals

```bash
rm model-F16.gguf
rm -rf /path/to/models/Qwen2.5-0.5B-Instruct
```

From this point on, `model-Q4_K_M.gguf` is **all you need**.

---

## Using the compressed model

### With the included demo

```bash
pip install llama-cpp-python

# Interactive chat
python demo.py --model path/to/model-Q4_K_M.gguf

# One-shot
python demo.py --model path/to/model-Q4_K_M.gguf \
    --prompt "Explain recursion in one sentence."

# Custom system prompt
python demo.py --model path/to/model-Q4_K_M.gguf \
    --system "You are a terse pirate." \
    --prompt "How do I bake bread?"
```

### In Python

```python
from llama_cpp import Llama

llm = Llama(
    model_path="path/to/model-Q4_K_M.gguf",
    n_ctx=2048,        # context window
    n_threads=4,       # CPU threads
    n_gpu_layers=0,    # set to 99 if you compiled llama.cpp with CUDA/Metal
    verbose=False,
)

response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the capital of France?"},
    ],
    max_tokens=64,
    temperature=0.7,
)
print(response["choices"][0]["message"]["content"])
print(response["usage"])  # prompt_tokens, completion_tokens, total_tokens
```

### With Ollama

Create a file named `Modelfile`:

```
FROM ./path/to/model-Q4_K_M.gguf
```

Then:

```bash
ollama create my-model -f Modelfile
ollama run my-model
```

### With the llama.cpp CLI directly

```bash
./llama-cli -m path/to/model-Q4_K_M.gguf -p "Hello!" -n 128 --temp 0.7
```

---

## Choosing a quantization type

| Quant   | Approx % of F16 | Quality     | Use case                              |
|---------|-----------------|-------------|---------------------------------------|
| Q8_0    | ~50%            | Best        | When size is not a concern            |
| Q6_K    | ~45%            | Excellent   | Production, balanced                   |
| Q5_K_M  | ~40%            | Very good   | Tighter size, still good quality      |
| **Q4_K_M** | **~35%**     | **Good**    | **Recommended default**               |
| Q4_K_S  | ~33%            | Good        | Slightly smaller variant of Q4_K_M    |
| Q3_K_M  | ~28%            | Acceptable  | When size matters more than quality   |
| Q2_K    | ~22%            | Lossy       | Last resort for very constrained envs |

To use a different quant, change `QUANT_TYPE` in the notebook or pass it to `llama-quantize`:

```bash
./build/bin/llama-quantize model-F16.gguf model-Q5_K_M.gguf Q5_K_M
```

---

## Verifying the compressed file is truly standalone

If you want to be sure the compressed file isn't secretly relying on a base model somewhere:

```bash
# 1. Compress the model
./build/bin/llama-quantize model-F16.gguf model-Q4_K_M.gguf Q4_K_M

# 2. Delete EVERYTHING except the compressed .gguf
rm model-F16.gguf
rm -rf models/Qwen2.5-0.5B-Instruct
rm -rf ~/.cache/huggingface

# 3. The compressed file still loads and generates:
./build/bin/llama-cli -m model-Q4_K_M.gguf -p "Hello!" -n 64
```

If the compressed file can load and generate with no other model files on disk, it's standalone. That's the whole point.

---

## Running the 10K conversation test

To prove your compressed model actually works at scale (not just on a single prompt):

```bash
pip install llama-cpp-python

# Full 10K test (chunked, resumable)
python scripts/run_chunks.py --model path/to/model-Q4_K_M.gguf \
    --chunks 20 --chunk-size 500 --max-tokens 24

# Aggregate results
python scripts/finalize_results.py

# View
cat results/final_summary.json
```

Each chunk of 500 takes ~9 minutes on a 2-core CPU. The test is resumable  if interrupted, re-run `run_chunks.py` and it picks up from `results/progress.json`.
