# Qwen3.8-27B: the Qwen3.5 / 3.6 / 3.8 generation (`qwen3_5` architecture)

The newest Qwen open models — **Qwen3.5**, **Qwen3.6** and **Qwen3.8** — are all
native vision-language models on the same architecture family:
`model_type: qwen3_5` (dense: `Qwen3_5ForConditionalGeneration`; MoE:
`Qwen3_5MoeForConditionalGeneration`). There is **no Qwen3.7** — Qwen went
3.5 → 3.6 → 3.8 (0 hits on the Hub for `Qwen3.7`).

Cached on gpu1: `Qwen/Qwen3.8-27B`. Available on the Hub include `Qwen/Qwen3.6-27B`,
`Qwen/Qwen3.6-27B-FP8`, `Qwen/Qwen3.6-35B-A3B(-FP8)`, and the full `Qwen/Qwen3.5-*`
range (0.8B → 397B-A17B).

## How to run (transformers, not vLLM)

vLLM 0.15.1 cannot serve this generation and transformers 4.57.6 does not know
`qwen3_5`: it needs **transformers ≥ 5.8** (the model card asks for `5.8.0.dev0`;
first tested on 5.9.0). `requirements.txt` pins the stack that runs inference *and*
LoRA training here (torch 2.9.0+cu128, transformers 5.17.0, TRL 1.13.0, peft 0.21.0,
flash-linear-attention 0.5.2). `setup.sh` creates that venv and adds `causal-conv1d`,
built from source (see *User-space CUDA toolkit and the fast kernels*):

```bash
VENV=$(bash models/Qwen3.8-27B/setup.sh)     # prints ~/.venvs/kth-<hash>
$VENV/bin/python your_script.py
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

## Findings

| Model | dtype | Load | Peak VRAM | Decode | Status |
|-------|-------|------|-----------|--------|--------|
| Qwen3.8-27B | bf16 | ~34s | 54.9 GiB | ~2.7 tok/s | runs (eager path) |
| Qwen3.6-27B | bf16 | ~36s | 54.9 GiB | ~2.7 tok/s | runs (eager path) |
| Qwen3.6-27B-FP8 | fp8 dynamic | — | — | — | **blocked** (see below) |

- **It works, but only on the slow eager path.** The `qwen3_5` text stack uses
  linear/hybrid attention whose fast kernels (`flash-linear-attention`,
  `causal-conv1d`) are not installed, so transformers falls back to pure torch:
  ~2.7 tok/s single-stream. Install those kernels (see *User-space CUDA toolkit and the
  fast kernels* below; and use vLLM once it supports the arch) for usable speed. bf16
  dense-27B needs ~55 GiB — fits the H100 slice.
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
  and the 0.15.1 serve scripts use it). The prebuilt wheels are out, so build from
  source: see *vLLM 0.24.0 built from source* below.

## vLLM 0.24.0 built from source (2026-09-29/30)

```bash
VENV=$(bash models/Qwen3.8-27B/build-vllm.sh)                  # ~20 min, once
python3 models/Qwen3.8-27B/lora_for_vllm.py my-lora my-lora-vllm   # adapters from TRL
bash models/Qwen3.8-27B/serve-vllm.sh my-lora-vllm sig            # :8000, "Qwen/Qwen3.8-27B" + "sig"
```

**Outcome: it builds and serves, with the LoRA applied, ~4.6x faster than transformers.
Don't send `n>1` requests: they intermittently hang the engine; send `n` separate `n=1`
requests instead (same speed, no hang).** Details:

- **Version choice.** vLLM 0.28 pins torch 2.13 (cu126/cu129/cu130 wheels only). vLLM
  **0.24.0** pins torch **2.11.0**, the last torch with a cu128 wheel, and already lists
  `Qwen3_5ForConditionalGeneration`. `requirements-vllm.txt` is that venv.
- **Build (`build-vllm.sh`).** Around `uv build --wheel --no-build-isolation`:
  `use_existing_torch.py` (keep torch cu128); the runtime requirements' `[cu13]` extras
  swapped (`nvidia-cutlass-dsl` plain, `humming-kernels[cu12]`); a user-space Rust
  toolchain via rustup (vLLM 0.24 has a Rust frontend); `TORCH_CUDA_ARCH_LIST=9.0`.
  Result: `~/wheels/vllm-0.24.0+cu128-cp311-cp311-linux_x86_64.whl` (286 MB, ~18 min).
- **The toolkit needs more than nvcc.** Torch's CMake config wants cublas, cufft, curand,
  cusparse, cusolver, nvtx, cupti, nvjitlink (`WITH_LIBS=1`, ~2.2 GB of tarballs, 5.6 GB
  installed). CUDA 12.8's CCCL 2.7 lacks the `cuda::ptx` overloads vLLM's
  `cooperative_topk.cuh` uses (`mbarrier_try_wait_parity`/`mbarrier_arrive_expect_tx`
  with `sem_relaxed`, `cp_async_bulk` to shared memory): `CCCL_FROM=12.9.1` overlays the
  header-only CCCL 2.8.2 from CUDA 12.9.1, and nvcc 12.8 (PTX ISA 8.7) compiles it.
- **Network.** The pod's DNS drops lookups (github.com, pypi.org, static.rust-lang.org in
  turn), and CMake FetchContent clones ~10 repos + submodules at configure time. Every
  download retries; the build retries only on network errors (fetched deps stay in `.deps/`).
- **Serving.** FlashInfer JIT-compiles kernels at startup (sampling, the gated-delta-net
  prefill), so `serve-vllm.sh` sources `tools/cuda-env.sh` (else it looks for
  `/usr/local/cuda/bin/nvcc`). **The pod has 32 CPUs and 128 GiB** (`/sys/fs/cgroup/cpu.max`,
  `memory.max`) while `nproc` says 224: the unbounded JIT (~70 nvcc at once) got the pod
  restarted, so `MAX_JOBS=8`. First start ~10 min (JIT), later ~1.5 min (cached in
  `~/.cache/flashinfer`); weights 51.3 GiB, load 10 s, torch.compile 62 s.
- **LoRA naming trap.** A TRL/peft adapter trained on `Qwen3_5ForCausalLM` is saved as
  `base_model.model.model.layers.N...`; vLLM's `Qwen3_5ForConditionalGeneration` expects
  `model.language_model.layers.N...`. vLLM logs `Loaded new LoRA adapter` and then matches
  **nothing**: outputs equal the base model's (measured: base 15% vs unrenamed adapter 14%
  hit@60 on 100 eval examples). `lora_for_vllm.py` renames the keys (992/992 here); then
  the adapter scores 28% hit@60 on those 100, same as transformers + peft on the full 500
  (28.4%). The packed GDN projections (`in_proj_qkvz`, `in_proj_ba`) take LoRA fine.
- **Speed.** n=60 samples, ≤64 tokens, ~500-token prompts: 3.4 s per request with one
  request in flight (transformers `generate()`: ~13–16 s, even with the fast kernels).
- **Bug: `n>1` hangs; workarounds measured.** With `"n": 60`, the engine eventually stops
  generating (`generation throughput: 0.0 tokens/s`, requests stuck as Running/Waiting with
  the KV cache ~45% used, EngineCore at 100% CPU, no compiler running): after ~10–25 min at
  4–16 concurrent requests, and after 379 of 500 requests even at one request in flight.
  Only a server restart recovers. On the same 500 prompts (with the adapter):
  - **60 × `n=1` requests** (60 in flight, one prompt at a time): 500/500, **no hang**,
    3.3 s per prompt, task hit@60 30.4%. This is what to use.
  - **`n=60` with `--enforce-eager`**: 500/500, no hang, 5.1 s per prompt, hit@60 29.4%.
  So parallel sampling + CUDA graphs is the trigger (prime suspect: graph replay of the GDN
  state update for forked sequences). Not diagnosed further: `ptrace` and NVML are blocked
  in the pod (a process can opt in to ptrace with `prctl(PR_SET_PTRACER,
  PR_SET_PTRACER_ANY)`, which works here). vLLM also warns about Triton JIT *during*
  inference for the GDN kernels (`_causal_conv1d_fwd_kernel`,
  `fused_sigmoid_gating_delta_rule_update_kernel`).

## User-space CUDA toolkit and the fast kernels

The container has no `nvcc`, but you don't need root to get one.
`tools/install-cuda-toolkit.sh` unpacks NVIDIA's redist tarballs (nvcc, cudart, cccl, nvrtc
for 12.8.1, 791 MB) into `~/cuda-12.8` and adds the `lib64 -> lib` link that nvcc and
torch's `cpp_extension` expect. `source tools/cuda-env.sh [venv]` sets
`CUDA_HOME`/`PATH`/`LD_LIBRARY_PATH` and puts the header dirs of torch's `nvidia-*` pip
wheels (cublas, cusparse, ...) on `CPATH`, so those libraries don't have to be downloaded
again. Code compiled by nvcc 12.8 runs on the CUDA 12.4 driver (minor-version
compatibility; checked with an sm_90 test kernel).

The two kernels the `qwen3_5` fallback warns about:
- `flash-linear-attention` + `fla-core` (Triton only, needs no nvcc): in `requirements.txt`;
- `causal-conv1d`: `setup.sh` **builds it from source** against the venv's torch
  (`--no-binary --no-cache --no-build-isolation --no-deps`; `setuptools`/`wheel`/`ninja`
  come from `requirements.txt`). A cached or prebuilt wheel imports with `undefined symbol:
  _ZN3c104cuda29c10_cuda_check_implementation...` (built for another torch).
  `setup.sh <existing venv>` adds it to a venv you already have.

Afterwards `transformers.utils.import_utils.is_flash_linear_attention_available()` and
`is_causal_conv1d_available()` are both True. Tested with torch 2.9.0+cu128, fla 0.5.2,
causal-conv1d 1.7.0. This doesn't make the CUDA-13 vLLM wheels run: they fail on the
driver, not on the toolkit.

## LoRA fine-tuning (TRL + peft)

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

**Train without the fast kernels.** Once `flash-linear-attention` is installed (`setup.sh`),
transformers uses it for the gated-delta-net layers in training too, and fla 0.5.2 refuses the
backward pass: `RuntimeError: Triton >= 3.4.0 and < 3.7.1 on Hopper GPUs produces incorrect
results for gated chunk_bwd_dqkwg`. torch 2.9 pins Triton 3.5, so don't upgrade Triton: keep fla
for inference only and make the training process unable to import it, *before* transformers
loads the model (it tries hub kernels, then the original package, then the torch path; the
`USE_HUB_KERNELS=NO` environment variable only disables the first):

```python
import sys
sys.modules["fla"] = None            # `import fla` now raises ImportError -> torch path
sys.modules["causal_conv1d"] = None
```

Speed is the same as without fla installed (~1.6 samples/s here).
