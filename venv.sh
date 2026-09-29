#!/bin/bash
# Create (once) or reuse the venv of a model spec and print its path.
# The venv is ~/.venvs/kth-<sha256 of the spec without comments/blank lines, 10 chars>, so
# models with the same spec share one venv (a vLLM venv is ~10 GB; the shared home is ~98% full).
# Usage: VENV=$(bash venv.sh models/<Model>/requirements.txt)
set -euo pipefail
REQ=$(realpath "$1")
UV=$(command -v uv || echo "$HOME/.local/bin/uv")
if [ ! -x "$UV" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh >&2
    UV=$HOME/.local/bin/uv
fi
HASH=$(grep -vE '^\s*(#|$)' "$REQ" | sha256sum | cut -c1-10)
VENV=$HOME/.venvs/kth-$HASH
if [ ! -f "$VENV/.kth-complete" ]; then
    echo "creating $VENV from $REQ" >&2
    "$UV" venv -q --python 3.11 "$VENV" >&2
    VIRTUAL_ENV=$VENV "$UV" pip install -r "$REQ" --index-strategy unsafe-best-match >&2
    cp "$REQ" "$VENV/kth-requirements.txt"
    touch "$VENV/.kth-complete"
fi
echo "$VENV"
