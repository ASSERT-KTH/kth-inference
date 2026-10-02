#!/bin/bash
# Python + native stacks (py-spy) of every vLLM process, e.g. when an engine hangs, plus the
# tail of its log. Needs the ptrace opt-in (tools/sitecustomize_ptrace.py installed in the
# venv as sitecustomize.py) in this pod (Yama ptrace_scope=1, no CAP_SYS_PTRACE).
# Usage: bash tools/hang_dump.sh <venv> [vllm log] [out dir=~/hang-dumps]
VENV=${1:?venv}
LOG=${2:-}
D=${3:-$HOME/hang-dumps}/$(date +%Y%m%dT%H%M%S)
mkdir -p "$D"
[ -n "$LOG" ] && tail -200 "$LOG" > "$D/vllm-tail.log"
[ -x "$VENV/bin/py-spy" ] || VIRTUAL_ENV=$VENV "$(command -v uv || echo ~/.local/bin/uv)" pip install -q py-spy
for pid in $(pgrep -f 'vllm serve'; pgrep -f 'VLLM::'); do
    { echo "# pid $pid: $(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | cut -c1-80)"
      timeout 60 "$VENV/bin/py-spy" dump --native --pid "$pid"; } > "$D/pyspy-$pid.txt" 2>&1
    grep -q 'Permission Denied' "$D/pyspy-$pid.txt" && echo "pid $pid: not opted in to ptrace" >&2
done
echo "$D"
