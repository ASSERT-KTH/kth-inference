#!/bin/bash
# Moved to models/QwQ-32B-AWQ/serve.sh (one folder per model); kept for backward compatibility.
exec bash "$(dirname "$0")/models/QwQ-32B-AWQ/serve.sh" "$@"
