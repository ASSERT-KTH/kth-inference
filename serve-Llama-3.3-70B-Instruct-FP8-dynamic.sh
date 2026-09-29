#!/bin/bash
# Moved to models/Llama-3.3-70B-Instruct-FP8-dynamic/serve.sh (one folder per model); kept for backward compatibility.
exec bash "$(dirname "$0")/models/Llama-3.3-70B-Instruct-FP8-dynamic/serve.sh" "$@"
