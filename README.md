# VLLM serving for fast inference on a single H100

Perfect for use on KTH's DGX H100 since it mostly sits idle ¯\\_(ツ)_/¯

## Models

One folder per model: `requirements.txt` is its venv spec, `serve.sh` (or `setup.sh`) runs
it, and `README.md` has its howto and findings. Oldest first; the date is when the served
Hugging Face repo was created (`createdAt` in the Hub API).

- [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) (2024-09-17): [`models/Qwen2.5-Coder-7B-Instruct`](models/Qwen2.5-Coder-7B-Instruct), vLLM 0.15.1, base + LoRA adapters; cheap to fine-tune (~75 min / 48k examples)
- [Llama-3.3-70B-Instruct-FP8-dynamic](https://huggingface.co/RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic) (2024-12-11): [`models/Llama-3.3-70B-Instruct-FP8-dynamic`](models/Llama-3.3-70B-Instruct-FP8-dynamic), vLLM 0.15.1, FP8
- [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) (2025-01-26): [`models/Qwen2.5-VL-7B-Instruct`](models/Qwen2.5-VL-7B-Instruct), vLLM 0.15.1, vision-language
- [QwQ-32B-AWQ](https://huggingface.co/Qwen/QwQ-32B-AWQ) (2025-03-05): [`models/QwQ-32B-AWQ`](models/QwQ-32B-AWQ), vLLM 0.15.1, AWQ-Marlin, ~78 tok/s
- [Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) (2025-07-31): [`models/Qwen3-Coder-30B-A3B-Instruct`](models/Qwen3-Coder-30B-A3B-Instruct), vLLM 0.15.1, ~178 tok/s, `qwen3_coder` tool calls; recipe for recent Qwen3 models
- [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) (2025-08-04): [`models/gpt-oss-20b`](models/gpt-oss-20b), vLLM 0.15.1
- [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (2026-08-05): [`models/Qwen3.8-27B`](models/Qwen3.8-27B), the `qwen3_5` generation (Qwen3.5/3.6/3.8): transformers + fast kernels, LoRA fine-tuning, and vLLM 0.24.0 built from source against cu128 (serves with LoRA; `n>1` sampling intermittently hangs)

## Quick Start

1. Start a server (the first run creates the model's venv):
```bash
bash models/QwQ-32B-AWQ/serve.sh
```

2. In a separate terminal, start the chat interface (configure the correct model name):
```bash
uv run chat.py
uv run chat.py --system "You are a Python programming expert"   # custom system prompt
```
Type your message and press Enter for new lines; Ctrl+J or ESC+Enter sends it. Pasting
multi-line text works.

## Venvs

`venv.sh models/<Model>/requirements.txt` creates the venv once and prints its path:
`~/.venvs/kth-<hash of the spec>`. Models with the same spec share one venv (a vLLM venv
is ~10 GB and the shared home is ~98% full), so all the vLLM 0.15.1 models use the same one.
To change a model's stack, edit its `requirements.txt`: the new spec gets a new venv.

The root `pyproject.toml` / `uv.lock` still pin the vLLM 0.15.1 stack for `uv run chat.py`
and `uv run vllm ...`, and the old `serve-*.sh` in the root forward to the model folders.

## Tools

- `chat.py`: terminal chat client for the OpenAI-compatible server on :8000
- `check-limit-max-tokens.py`: probe the max tokens a served model accepts
- `test_tool_calling.py`, `test_tool_calling_v2.py`: tool-calling smoke tests
- `tools/install-cuda-toolkit.sh`, `tools/cuda-env.sh`: user-space CUDA 12.8 toolkit (no root),
  to build CUDA extensions against torch cu128 (`WITH_LIBS=1` adds cublas & co., `CCCL_FROM=12.9.1`
  a newer CCCL, e.g. to build vLLM); see [`models/Qwen3.8-27B`](models/Qwen3.8-27B)
- `tools/sitecustomize_ptrace.py`, `tools/hang_dump.sh`: stack dumps of hung processes (below)

## The gpu1 pod (what the machine really is)

Measured 2026-09/10 inside the JupyterHub pod:
- **1× H100 80GB as a MIG 7g.80gb instance** (the whole GPU), driver 535.247.01 = **CUDA 12.4**:
  wheels built for CUDA 13 fail (`The NVIDIA driver on your system is too old`); cu12x wheels
  (torch cu128, our vLLM builds) run fine.
- **32 CPUs and 128 GiB RAM**, from `/sys/fs/cgroup/cpu.max` and `memory.max`, although
  `nproc` / `free` show the host's 224 cores and 2 TB. Tools that size parallelism from `nproc`
  (ninja, JIT compilers) oversubscribe the pod; an unbounded JIT got it OOM-restarted. Set
  `MAX_JOBS` explicitly.
- **A restart of the pod wipes `/tmp`** (and anything outside `$HOME`). Keep builds' outputs,
  wheels, checkpoints and adapters in `$HOME`.
- **The network is unreliable**: DNS lookups fail for minutes at a time (github.com, pypi.org,
  static.rust-lang.org in turn), sometimes the whole network. Retry downloads
  (`curl --retry 8 --retry-all-errors`), and start servers of cached models with
  `HF_HUB_OFFLINE=1` (vLLM otherwise fails engine start-up asking the Hub for a file list).
- **No root, no `nvcc`** (see `tools/install-cuda-toolkit.sh`), **no `ptrace` between
  processes** (Yama `ptrace_scope=1`, no `CAP_SYS_PTRACE`, `NoNewPrivs=1`), and **NVML can't
  read MIG memory** (`nvidia-smi --query-gpu=memory.used` → `[Insufficient Permissions]`;
  GPU utilization is `Not Supported` on MIG in NVML anyway; `nvidia-smi`'s table does show
  per-MIG memory).
- `/home/jovyan` is shared and ~98% full: share venvs (`venv.sh`), don't duplicate models.

## Debugging a hung server in this pod

ptrace works for processes that opt in with `prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY)`.
Install the opt-in into a venv once, so every Python process of it (and their forks / spawns)
can be traced:

```bash
SP=$(VENV/bin/python -c 'import site; print(site.getsitepackages()[0])')
cp tools/sitecustomize_ptrace.py "$SP/sitecustomize.py"
bash tools/hang_dump.sh VENV ~/vllm.log      # py-spy --native of the API server and EngineCore
```

Processes started before the install aren't opted in (`Permission Denied`). This is how the
Qwen3.8-27B vLLM hang was located (a GPU-side stall seen from a blocked cuBLAS launch,
[`models/Qwen3.8-27B`](models/Qwen3.8-27B)).
