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
  `Qwen3_5ForConditionalGeneration` (Qwen3.5-VL) — serving it needs a newer vLLM.
- **Cache gotcha**: the cached dir `models--Qwen--Qwen3.8-27B` is *not* a 27B dense
  model — its config is `Qwen3_5ForConditionalGeneration` (Qwen3.5-VL, multimodal,
  52 GiB), which the current vLLM cannot load. Don't trust the directory name; check
  `config.json` `architectures`.

### Throughput reference

| Model | Quant / dtype | Tool parser | Single-stream decode |
|-------|---------------|-------------|----------------------|
| Qwen3-Coder-30B-A3B-Instruct | bf16 + fp8 KV | `qwen3_coder` | ~178 tok/s |
| QwQ-32B (see below) | AWQ-Marlin | — | ~78 tok/s |

## Performance Notes

For Qwen/QwQ-32B, these are the observed generation speeds, each subsequent item includes the changes of the ones above:

- AWQ: ~38 tokens/s
- AWQ-Marlin instead: ~78 tokens/s
  - Adding --enforce-eager here brings it down to: 66 tokens/s
- everything else I can think of doesnt help any more
