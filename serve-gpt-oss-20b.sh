#!/bin/bash
# Moved to models/gpt-oss-20b/serve.sh (one folder per model); kept for backward compatibility.
exec bash "$(dirname "$0")/models/gpt-oss-20b/serve.sh" "$@"
