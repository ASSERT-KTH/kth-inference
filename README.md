# VLLM serving for fast inference on a single H100

Perfect for use on KTH's DGX H100 since it mostly sits idle ¯\\_(ツ)_/¯

## Quick Start

1. Start the vLLM server:
```bash
ls serve*sh
# example
sh serve-QwQ-32B-AWQ.sh
```

2. In a separate terminal, start the chat interface (configure the correct model name):
```bash
uv run chat.py
```

## Configuration

### Chat Interface

You can customize the chat interface with these parameters:

```bash
# Set a custom system prompt
uv run chat.py --system "You are a Python programming expert"
```

## Usage

- Type your message and press Enter for new lines
- Press Ctrl+J or ESC+Enter to send the message
- Supports pasting multi-line text

## Recent Qwen3 models

Howto and findings from serving the Qwen3 family, measured 2026-09-06 on gpu1
(vLLM **0.15.1**, torch 2.9.1+cu128, single H100 MIG slice).

### How to serve

```bash
sh serve-Qwen3-Coder.sh          # Qwen3-Coder-30B-A3B-Instruct (MoE)
```

The recipe that works for a recent Qwen3 model on this box:

```bash
uv run vllm serve <model> \
    --host 0.0.0.0 --port 8000 \
    --kv-cache-dtype fp8 \        # halve KV memory (MIG slice is tight)
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder   # for Qwen3-Coder; see parser note below
```

### Findings

- **Qwen3-Coder-30B-A3B-Instruct** (MoE, ~3B active): **~178 tok/s** single-stream
  decode (167 tok/s @128 tokens, 178 tok/s @512). 57 GiB weights, engine init ~41s,
  fp8 KV cache = 274,816 tokens on the slice (~12.6 GiB KV). Fits comfortably at
  `--gpu-memory-utilization 0.90`.
- **Tool calling** works: `--enable-auto-tool-choice --tool-call-parser qwen3_coder`
  returns a proper `tool_calls` message (`finish_reason: tool_calls`). Qwen3-Coder
  uses its own XML tool-call format, so it needs the `qwen3_coder` parser
  specifically — the `llama3_json` parser used by the older serve scripts will not
  parse its calls. For non-Coder Qwen3 chat models use `--tool-call-parser hermes`.
- **vLLM 0.15.1 architecture support** (`ModelRegistry.get_supported_archs()`):
  supported — `Qwen3ForCausalLM`, `Qwen3MoeForCausalLM`, `Qwen3NextForCausalLM`,
  `Qwen3VLForConditionalGeneration`, `Qwen3VLMoeForConditionalGeneration`,
  `Qwen3OmniMoeForConditionalGeneration`. **Not supported**:
  `Qwen3_5ForConditionalGeneration` / `Qwen3_5MoeForConditionalGeneration` (the
  Qwen3.5/3.6/3.8 generation) — see below.

### Throughput reference

| Model | Quant / dtype | Tool parser | Single-stream decode |
|-------|---------------|-------------|----------------------|
| Qwen3-Coder-30B-A3B-Instruct | bf16 + fp8 KV | `qwen3_coder` | ~178 tok/s |
| QwQ-32B (see below) | AWQ-Marlin | — | ~78 tok/s |

## Qwen3.5 / 3.6 / 3.8 generation (the `qwen3_5` architecture)

The newest Qwen open models — **Qwen3.5**, **Qwen3.6** and **Qwen3.8** — are all
native vision-language models on the same architecture family:
`model_type: qwen3_5` (dense: `Qwen3_5ForConditionalGeneration`; MoE:
`Qwen3_5MoeForConditionalGeneration`). There is **no Qwen3.7** — Qwen went
3.5 → 3.6 → 3.8 (0 hits on the Hub for `Qwen3.7`).

Cached on gpu1: `Qwen/Qwen3.8-27B`. Available on the Hub include `Qwen/Qwen3.6-27B`,
`Qwen/Qwen3.6-27B-FP8`, `Qwen/Qwen3.6-35B-A3B(-FP8)`, and the full `Qwen/Qwen3.5-*`
range (0.8B → 397B-A17B).

### How to run (transformers, not vLLM)

vLLM 0.15.1 cannot serve this generation and transformers 4.57.6 does not know
`qwen3_5`. Use a throwaway venv on **transformers ≥ 5.8** (the model card asks for
`5.8.0.dev0`; tested on 5.9.0):

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python transformers==5.9.0 accelerate \
    safetensors pillow torchvision torch --torch-backend=cu128
```

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
mid = "Qwen/Qwen3.6-27B"                    # or Qwen/Qwen3.8-27B
proc = AutoProcessor.from_pretrained(mid)   # torchvision required (video preproc)
model = AutoModelForImageTextToText.from_pretrained(mid, dtype=torch.bfloat16,
                                                    device_map="cuda")
msgs = [{"role":"user","content":[{"type":"text","text":"What is KTH?"}]}]
inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                  return_dict=True, return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=80)   # thinking mode on by default
```

### Findings

| Model | dtype | Load | Peak VRAM | Decode | Status |
|-------|-------|------|-----------|--------|--------|
| Qwen3.8-27B | bf16 | ~34s | 54.9 GiB | ~2.7 tok/s | runs (eager path) |
| Qwen3.6-27B | bf16 | ~36s | 54.9 GiB | ~2.7 tok/s | runs (eager path) |
| Qwen3.6-27B-FP8 | fp8 dynamic | — | — | — | **blocked** (see below) |

- **It works, but only on the slow eager path.** The `qwen3_5` text stack uses
  linear/hybrid attention whose fast kernels (`flash-linear-attention`,
  `causal-conv1d`) are not installed, so transformers falls back to pure torch:
  ~2.7 tok/s single-stream. Install those kernels (and use vLLM once it supports the
  arch) for usable speed. bf16 dense-27B needs ~55 GiB — fits the H100 slice.
- **FP8 is blocked in this container.** `Qwen3.6-27B-FP8` (dynamic e4m3) loads its
  weights, but the finegrained-FP8 matmul is a JIT-compiled DeepGEMM kernel (via the
  `kernels` package) that needs a real CUDA toolkit — `nvcc` is absent in the
  JupyterHub container (`std::filesystem::exists(nvcc_path)` assertion), and the pip
  `nvidia-cuda-nvcc-cu12` wheel ships only `ptxas`, not `nvcc`. So run the **bf16**
  weights here, or serve FP8 from a container that has the CUDA toolkit.
- **Version pinning matters.** transformers 5.9.0 requires `kernels>=0.12,<0.13`;
  the current `kernels` (0.16.x) breaks the transformers import
  (`LayerRepository ... revision or a version must be specified`).
- **To serve properly** you need a newer vLLM than 0.15.1 (latest on PyPI is 0.28.0;
  verify `Qwen3_5ForConditionalGeneration` is in `get_supported_archs()` before the
  multi-GB download) in its own venv, not the pinned 0.15.1 of the other serve scripts.

## Performance Notes

For Qwen/QwQ-32B, these are the observed generation speeds, each subsequent item includes the changes of the ones above:

- AWQ: ~38 tokens/s
- AWQ-Marlin instead: ~78 tokens/s
  - Adding --enforce-eager here brings it down to: 66 tokens/s
- everything else I can think of doesnt help any more
