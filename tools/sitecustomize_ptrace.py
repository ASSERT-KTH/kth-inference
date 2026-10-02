"""Opt every Python process of this venv in to being ptraced by any process (same uid).

Installed as `sitecustomize.py` in the vLLM venv on gpu1, so that `py-spy dump --pid` can
take stacks of the vLLM API server and EngineCore when they hang. The pod runs with Yama
`ptrace_scope=1` (only descendants may be traced) and no CAP_SYS_PTRACE; a tracee may still
whitelist tracers with prctl(PR_SET_PTRACER). The exception is per task and isn't
inherited across fork, so it's re-applied in forked children too.
"""

import ctypes
import os

_PR_SET_PTRACER = 0x59616D61
_PR_SET_PTRACER_ANY = ctypes.c_ulong(-1)


def _optin() -> None:
    try:
        ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_PTRACER, _PR_SET_PTRACER_ANY, 0, 0, 0)
    except Exception:  # noqa: BLE001  (never break interpreter start-up)
        pass


_optin()
os.register_at_fork(after_in_child=_optin)
