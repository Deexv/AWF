# Architecture: Why Standalone Compression Works

This document explains why the compressed `.gguf` files produced by this repo are truly standalone — i.e., why they do **not** require the original (uncompressed) model at runtime — and contrasts this with compression schemes that do.

---

## TL;DR

| Compression family | How it works | Needs base at runtime? |
|--------------------|--------------|------------------------|
| **Quantization (standalone)** | Rewrite all weights to lower-precision format, save to a self-contained file | **NO** ✅ |
| Adapter / delta | Compute a small "correction" tensor to apply on top of the base model | **YES** ❌ |
| Pruning (sparse) | Zero out some weights, but ship them as a delta/diff | **YES** ❌ (unless dense-pruned) |
| Distillation | Train a smaller model from scratch | **NO** ✅ (but expensive to produce) |

This repo only uses **standalone quantization** (specifically llama.cpp's GGUF Q4_K_M format).

---

## How standalone quantization works

### Step 1: Load the original model (once, at compression time)

The original HuggingFace model has weights stored as 16-bit floats (`fp16` or `bf16`). For a 0.5B-parameter model, that's roughly:

```
0.5 × 10^9 params × 2 bytes/param = ~1 GB on disk
```

### Step 2: Re-encode each weight tensor

`llama-quantize` reads each tensor and re-encodes the values using a block-wise quantization scheme. For `Q4_K_M`:

- Most weights are encoded in **4-bit** blocks of 32 elements, with one 6-bit scale and one 4-bit min per block.
- Some "important" tensors (like the FFN down-projection) are encoded in **6-bit** to preserve quality.
- A few small tensors (norms, biases) are kept in `f32`.

The result is a new weight blob where the average bits-per-weight (BPW) is ~6.35 for Q4_K_M, vs 16 for the original F16.

### Step 3: Write everything to ONE file

The GGUF file format is a single binary that bundles:

1. **Model architecture metadata** (number of layers, hidden size, vocabulary size, attention heads, etc.) — so the runtime knows how to construct the computation graph.
2. **The tokenizer** (vocabulary, merges, special tokens).
3. **The chat template** (Jinja2 template string, e.g. `<|im_start|>{role}\n{content}<|im_end|>`).
4. **All the quantized weight tensors**, in a flat key-value store.

This is the key difference from adapter-based compression: **all the information needed to run the model lives inside the compressed file**. There's no "external reference" the runtime needs to resolve.

### Step 4: At inference time

The runtime (`llama.cpp`, `llama-cpp-python`, Ollama, etc.) opens the `.gguf` file, reads the metadata to construct the model graph, mmaps the weight tensors, and runs inference. **At no point does it look for the original HF model.**

You can prove this by deleting every other file on disk and running inference — it'll still work.

---

## Why adapter-based compression isn't standalone

### LoRA

A LoRA adapter contains two small low-rank matrices `A` and `B` such that the modified weight is:

```
W' = W + α · B · A
```

Where `W` is the **original** weight. Without `W`, you can't compute `W'`. So you must have the base model on disk at inference time.

This is fine for fine-tuning (the adapter is the only thing you swap), but it is **not** compression in the user-facing sense: the total download size (base + adapter) is *larger* than the base alone.

### Sparse delta / diff patches

Some "compression" schemes work by:

1. Identifying weights that can be set to zero without much quality loss.
2. Shipping only the *diff* (which weights changed, and to what).

But to reconstruct the model, you need the original weights to apply the diff to. So again: **base model required at runtime.**

### Low-rank approximation (SVD-based)

If you decompose `W ≈ U · V` where `U` is `n×k` and `V` is `k×m`, and ship only `U` and `V`, then technically the original `W` is gone. **This CAN be standalone** — but only if you ship the full low-rank factors and reconstruct `W = U · V` at load time.

The catch: for the approximation to be any good, `k` has to be close to `min(n, m)`, so the compression ratio is usually poor (50-70% of original). That's worse than Q4_K_M (~35% of original) and worse than Q8_0 (~50%).

That's why this repo uses quantization rather than low-rank approximation.

---

## Why Q4_K_M specifically

llama.cpp offers a family of quantization types, each making a different quality/size tradeoff:

| Quant   | Bits/weight (avg) | Size vs F16 | Perplexity degradation |
|---------|-------------------|-------------|-------------------------|
| F16     | 16.0              | 100%        | baseline                |
| Q8_0    | 8.5               | ~53%        | negligible              |
| Q6_K    | 5.5-6.0           | ~45%        | tiny                    |
| Q5_K_M  | 5.5               | ~40%        | small                   |
| **Q4_K_M** | **4.85**       | **~35%**    | **small-moderate**      |
| Q3_K_M  | 3.9               | ~30%        | moderate                |
| Q2_K    | 2.6               | ~22%        | significant              |

`Q4_K_M` is the recommended default in the llama.cpp community because:

1. **Size reduction**: 65% smaller than the original F16 model.
2. **Quality**: Minimal perplexity increase, well below the threshold where outputs become noticeably worse for typical instruction-following tasks.
3. **Speed**: 4-bit integer matmuls are well-optimized on modern CPUs (often faster than F16 due to better cache utilization).
4. **Compatibility**: Universally supported across llama.cpp, Ollama, LM Studio, kobold.cpp, and others.

For a deeper dive, see the [llama.cpp quantization docs](https://github.com/ggerganov/llama.cpp/tree/master/examples/quantize).

---

## What this repo does NOT do

To be very explicit, because there's been confusion:

- **Does NOT use LoRA.** No adapters, no base-model-plus-delta.
- **Does NOT use sparse diff.** The compressed file is dense.
- **Does NOT need the HF model directory at runtime.** Once the `.gguf` is produced, you can `rm -rf` the HF directory.
- **Does NOT need transformers / PyTorch at runtime.** Only `llama-cpp-python` (or any other GGUF-aware runtime).
- **Does NOT need any "demo mode" or "template mode".** The 10K test runs real inference against the compressed weights.

---

## The math of standalone-ness

Formally: a compressed artifact `C` is **standalone** if there exists a decoder function `D` such that `D(C) → M'` produces a runnable model `M'`, where `D` depends only on `C` and not on any external file or registry.

For GGUF Q4_K_M:

```
D(.gguf) = parse_gguf_metadata() + reconstruct_quantized_tensors() + apply_chat_template()
```

All inputs to `D` come from inside the `.gguf` file. Therefore, GGUF Q4_K_M is standalone. QED.

For LoRA:

```
D(lora_adapter) = apply_delta(W_base, lora_adapter)
                                  ↑
                          requires external base model
```

`D` depends on `W_base`, which is not in the LoRA file. Therefore, LoRA is not standalone.

---

## Frequently confused points

### "But I saw a compression demo that needed the base model!"

You probably saw a LoRA / sparse-diff / adapter-based demo. Those genuinely need the base model. This repo is different — it uses quantization, which doesn't.

### "Then why does the Colab notebook download the base model at all?"

Because you need the original weights **once**, at compression time, to read them and re-encode them as 4-bit. After the compressed file is written, the original can be (and is) deleted. The notebook does this explicitly in Step 6 to make the point.

### "Can I use the compressed file in production?"

Yes. The `.gguf` format is the standard interchange format for llama.cpp-based runtimes, which are widely deployed in production (Ollama, LM Studio, llamafile, LocalAI, etc.). The file format is stable across versions.

### "Does this work for non-instruct models too?"

Yes, but the chat template won't be useful. For base models (no instruct tuning), you'd use the `.gguf` for raw text completion rather than chat. The compression step is identical.

### "What about vision models / multimodal models?"

Some are supported (e.g. LLaVA, Qwen-VL, Gemma-3 vision). The `convert_hf_to_gguf.py` script handles them, and the resulting `.gguf` is similarly standalone. The included 10K test in this repo only covers text instruct models because that's the common case.

---

## References

- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [GGUF file format spec](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [Quantization types reference](https://github.com/ggerganov/llama.cpp/wiki/Local-LLM-Inference-On-CPU-+-GPU#quantization)
- [HuggingFace GGUF docs](https://huggingface.co/docs/hub/gguf)
