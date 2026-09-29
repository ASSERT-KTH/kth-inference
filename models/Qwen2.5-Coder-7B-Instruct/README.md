# Qwen2.5-Coder-7B-Instruct (+ LoRA adapters)

```bash
bash models/Qwen2.5-Coder-7B-Instruct/serve.sh                       # base only
bash models/Qwen2.5-Coder-7B-Instruct/serve.sh ~/adapters/my-lora sig  # base + adapter "sig"
```

vLLM 0.15.1 (same spec as the other vLLM models, so the same shared venv), 8k context.
With an adapter, requests pick it with `"model": "<name>"`, and the base stays
available as `"model": "Qwen/Qwen2.5-Coder-7B-Instruct"` on the same server.
`--max-lora-rank` defaults to 16 (`MAX_LORA_RANK=...` for bigger adapters).

## Findings (2026-09-25..29, gpu1)

- **LoRA serving works on vLLM 0.15.1**: `--enable-lora --lora-modules name=dir` with a
  peft adapter (r=16, all linear layers) saved by TRL; ~40 s to start.
- **Fine-tuning this 7B is cheap**: LoRA SFT with TRL 1.13 / transformers 5.17 (the
  [`../Qwen3.8-27B`](../Qwen3.8-27B) venv spec also trains this model) runs at ~10.8
  samples/s (prompts ~300 tokens), so ~75 min for 48k examples, 1 epoch, on the H100.
  That's ~6.5x faster than the same LoRA on Qwen3.8-27B.
- **Sampling many candidates is fast**: `"n": 60` per request with 16–48 concurrent
  requests; 500 prompts × 60 samples (≤64 tokens) in ~40 s.
- On a narrow task (predicting a masked function signature from its siblings,
  keccak-verified), the LoRA 7B reached hit@60 22.2% vs 4.4% for the base 7B and 2.0%
  for zero-shot Qwen3-Coder-30B-A3B. With transformers + peft generation instead of vLLM,
  the same adapter gives the same 22.2%, so the vLLM LoRA path doesn't change results.
- Qwen2.5 chat templates ignore `chat_template_kwargs={"enable_thinking": False}`, so
  one client can send it to both this model and the `qwen3_5` ones.
