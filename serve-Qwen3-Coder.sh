#!/bin/bash
# Moved to models/Qwen3-Coder-30B-A3B-Instruct/serve.sh (one folder per model); kept for backward compatibility.
exec bash "$(dirname "$0")/models/Qwen3-Coder-30B-A3B-Instruct/serve.sh" "$@"
