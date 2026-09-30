#!/usr/bin/env python3
"""Rename a qwen3_5 LoRA trained on the text-only Qwen3_5ForCausalLM for vLLM.

transformers' AutoModelForCausalLM gives Qwen3_5ForCausalLM (weights `model.layers.N...`),
so peft saves `base_model.model.model.layers.N...`. vLLM serves the checkpoint's
Qwen3_5ForConditionalGeneration, whose mapper expects `model.language_model.layers.N...`
(-> `language_model.model.layers.N...`). Unrenamed, vLLM logs "Loaded new LoRA adapter"
and then matches none of the tensors: outputs equal the base model's.

Usage: python3 lora_for_vllm_qwen3_5.py adapters/sig-lora-27b adapters/sig-lora-27b-vllm
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

OLD, NEW = "base_model.model.model.layers.", "base_model.model.model.language_model.layers."


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and f.name != "adapter_model.safetensors":
            shutil.copy2(f, dst / f.name)
    t = load_file(str(src / "adapter_model.safetensors"))
    out = {(NEW + k[len(OLD):] if k.startswith(OLD) else k): v for k, v in t.items()}
    renamed = sum(1 for k in t if k.startswith(OLD))
    save_file(out, str(dst / "adapter_model.safetensors"), metadata={"format": "pt"})
    print(f"{renamed}/{len(t)} tensors renamed -> {dst}")


if __name__ == "__main__":
    main()
