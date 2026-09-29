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
- **Newer vLLM doesn't run on this driver either.** vLLM 0.24.0 and 0.28.0 do list
  `Qwen3_5ForConditionalGeneration`, but their wheels are built on torch CUDA 13.0 and
  the gpu1 driver is CUDA 12.4: the engine dies with `The NVIDIA driver on your system
  is too old (found version 12040)`. torch **cu128** runs fine on this driver (training
  and the 0.15.1 serve scripts use it). Until there's a cu12x vLLM build that knows
  `qwen3_5`, generate with transformers.

### LoRA fine-tuning (TRL + peft)

Tested 2026-09-29 on `Qwen/Qwen3.8-27B`, bf16 LoRA r=16 on all linear layers,
torch 2.9.0+cu128, transformers 5.17.0, TRL 1.13.0, peft 0.21.0. Running it takes
four workarounds:

1. **TRL loads the image processor**, so the venv needs `pillow` and a `torchvision`
   that matches torch (`torchvision==0.24.0` from the cu128 index for torch 2.9.0).
   Otherwise: `ImportError: Qwen2VLImageProcessor requires the PIL library`.
2. **Load the text-only causal LM yourself** and pass the object to `SFTTrainer`.
   `AutoModelForCausalLM` gives `Qwen3_5ForCausalLM` (26.9B params, 50 GiB bf16, fits
   the 80 GB H100). With a model *name*, TRL takes its VLM path, `device_map="auto"`
   offloads to CPU ("Some parameters are on the meta device"), and backward fails
   (`MmBackward0 returned an invalid gradient ... expected device meta`). Forcing
   `device_map={"": 0}` on that path then OOMs at load. Pass
   `processing_class=AutoTokenizer...` so no processor is built.
3. **`loss_type="nll"`.** The default chunked loss patches `lm_head` and assumes a
   bound `forward`; here it's a `functools.partial`:
   `AttributeError: 'functools.partial' object has no attribute '__func__'`.
4. **Disable thinking in the chat template.** By default it injects a "Reasoning effort
   is set to xhigh..." system text and an open `<think>` block, so prompt and
   prompt+completion tokenize differently (TRL: `Mismatch between tokenized prompt and
   the start of tokenized prompt+completion`) and the completion-only loss mask is
   wrong. Pre-render plain-text prompts with
   `apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)`
   (they end in an empty `<think>\n\n</think>\n\n`), and pass
   `chat_template_kwargs={"enable_thinking": False}` at inference. Qwen2.5 templates
   ignore the flag.

```python
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.8-27B", dtype=torch.bfloat16,
                                             device_map={"": 0})
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B")
cfg = SFTConfig(loss_type="nll", bf16=True, gradient_checkpointing=True,
                per_device_train_batch_size=2, gradient_accumulation_steps=16, ...)
SFTTrainer(model=model, processing_class=tok, args=cfg, train_dataset=ds,  # plain-text
           peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear",
                                  task_type="CAUSAL_LM"))                  # prompt/completion
```

Throughput: ~1.7 samples/s (prompts ~300 tokens median, ~19 s per 32-example
step) vs ~10.8 samples/s for Qwen2.5-Coder-7B with the same data, so about 8 h
per epoch of 48k examples. Also check with transformers 5: `warmup_ratio` is gone
from `TrainingArguments`; use `warmup_steps=<float in [0,1)>` for a ratio.

## Performance Notes

For Qwen/QwQ-32B, these are the observed generation speeds, each subsequent item includes the changes of the ones above:

- AWQ: ~38 tokens/s
- AWQ-Marlin instead: ~78 tokens/s
  - Adding --enforce-eager here brings it down to: 66 tokens/s
- everything else I can think of doesnt help any more
