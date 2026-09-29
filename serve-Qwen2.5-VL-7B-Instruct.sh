#!/bin/bash
# Moved to models/Qwen2.5-VL-7B-Instruct/serve.sh (one folder per model); kept for backward compatibility.
exec bash "$(dirname "$0")/models/Qwen2.5-VL-7B-Instruct/serve.sh" "$@"
