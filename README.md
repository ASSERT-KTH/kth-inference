# VLLM serving for fast inference on a single H100

Perfect for use on KTH's DGX H100 since it mostly sits idle ¯\\_(ツ)_/¯

## Models

One folder per model: `requirements.txt` is its venv spec, `serve.sh` (or `setup.sh`) runs
it, and `README.md` has its howto and findings.

- [`models/Qwen3-Coder-30B-A3B-Instruct`](models/Qwen3-Coder-30B-A3B-Instruct): vLLM 0.15.1, ~178 tok/s, `qwen3_coder` tool calls; recipe for recent Qwen3 models
- [`models/QwQ-32B-AWQ`](models/QwQ-32B-AWQ): vLLM 0.15.1, AWQ-Marlin, ~78 tok/s
- [`models/Llama-3.3-70B-Instruct-FP8-dynamic`](models/Llama-3.3-70B-Instruct-FP8-dynamic): vLLM 0.15.1, FP8
- [`models/Qwen2.5-VL-7B-Instruct`](models/Qwen2.5-VL-7B-Instruct): vLLM 0.15.1, vision-language
- [`models/Qwen2.5-Coder-7B-Instruct`](models/Qwen2.5-Coder-7B-Instruct): vLLM 0.15.1, base + LoRA adapters; cheap to fine-tune (~75 min / 48k examples)
- [`models/gpt-oss-20b`](models/gpt-oss-20b): vLLM 0.15.1
- [`models/Qwen3.8-27B`](models/Qwen3.8-27B): the `qwen3_5` generation (Qwen3.5/3.6/3.8), transformers only (no vLLM on this driver), fast kernels, LoRA fine-tuning

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
  to build CUDA extensions against torch cu128; see [`models/Qwen3.8-27B`](models/Qwen3.8-27B)
