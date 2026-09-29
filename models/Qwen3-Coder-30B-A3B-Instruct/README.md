# Qwen3-Coder-30B-A3B-Instruct (and recent Qwen3 models)

```bash
bash models/Qwen3-Coder-30B-A3B-Instruct/serve.sh
```

Howto and findings from serving the Qwen3 family, measured 2026-09-06 on gpu1
(vLLM **0.15.1**, torch 2.9.1+cu128, single H100 MIG slice).

## How to serve

The recipe that works for a recent Qwen3 model on this box:

```bash
vllm serve <model> \
    --host 0.0.0.0 --port 8000 \
    --kv-cache-dtype fp8 \        # halve KV memory (MIG slice is tight)
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder   # for Qwen3-Coder; see parser note below
```

## Findings

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
  Qwen3.5/3.6/3.8 generation), see [`../Qwen3.8-27B`](../Qwen3.8-27B).

## Throughput reference

| Model | Quant / dtype | Tool parser | Single-stream decode |
|-------|---------------|-------------|----------------------|
| Qwen3-Coder-30B-A3B-Instruct | bf16 + fp8 KV | `qwen3_coder` | ~178 tok/s |
| QwQ-32B ([`../QwQ-32B-AWQ`](../QwQ-32B-AWQ)) | AWQ-Marlin | — | ~78 tok/s |
