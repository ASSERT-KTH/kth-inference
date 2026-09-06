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
  `Qwen3_5ForConditionalGeneration` (Qwen3.5 family) — see below.

### Throughput reference

| Model | Quant / dtype | Tool parser | Single-stream decode |
|-------|---------------|-------------|----------------------|
| Qwen3-Coder-30B-A3B-Instruct | bf16 + fp8 KV | `qwen3_coder` | ~178 tok/s |
| QwQ-32B (see below) | AWQ-Marlin | — | ~78 tok/s |

## Qwen3.5 family (Qwen3.8-27B)

The cached `models--Qwen--Qwen3.8-27B` is **Qwen3.8-27B**, a native vision-language
dense model built on the **Qwen3.5 architecture** (`model_type: qwen3_5`, arch
`Qwen3_5ForConditionalGeneration`). Findings (2026-09-06):

- **Not servable on the installed stack.** vLLM 0.15.1 does not register
  `Qwen3_5ForConditionalGeneration`, and transformers 4.57.6 does not know
  `qwen3_5`. The model card asks for `transformers 5.8.0.dev0`.
- **Runs via transformers ≥ 5.8** (tested with a throwaway venv on 5.9.0 +
  torchvision + torch 2.11+cu128):
  ```bash
  uv venv --python 3.11 .venv
  uv pip install --python .venv/bin/python transformers==5.9.0 accelerate \
      safetensors pillow torchvision torch --torch-backend=cu128
  ```
  Load with `AutoModelForImageTextToText` + `AutoProcessor` (torchvision is
  required by the video preprocessor). Text-only smoke test generated coherent
  output; thinking mode is on by default (emits a `</think>` block).
- **Numbers:** loaded in ~34s, peak **54.9 GiB** (bf16, 27B dense), fits the H100
  slice. Decode was only **~2.7 tok/s** — this is the *slow eager path*: the
  `qwen3_5` architecture uses linear/hybrid attention whose fast kernels
  (`flash-linear-attention`, `causal-conv1d`) were not installed, so it fell back
  to the pure-torch implementation. Install those (and use vLLM once it supports
  the arch) for usable speed.
- **To serve it properly** you need a newer vLLM than 0.15.1 (latest on PyPI is
  0.28.0; verify `Qwen3_5ForConditionalGeneration` is in
  `ModelRegistry.get_supported_archs()` before committing to the download) in its
  own venv, not the pinned 0.15.1 used by the other serve scripts.

## Performance Notes

For Qwen/QwQ-32B, these are the observed generation speeds, each subsequent item includes the changes of the ones above:

- AWQ: ~38 tokens/s
- AWQ-Marlin instead: ~78 tokens/s
  - Adding --enforce-eager here brings it down to: 66 tokens/s
- everything else I can think of doesnt help any more
