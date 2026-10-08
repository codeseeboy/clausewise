"""Per-request timings, token counts and errors.

A request starts a trace; each step inside it adds its duration. The finished trace is
appended to a JSON lines file, which is enough to compute per-step latency and error rates.
"""

import json
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_current: ContextVar[dict | None] = ContextVar("trace", default=None)


def start_trace(name: str, **attrs: str) -> dict:
    trace = {"trace_id": uuid.uuid4().hex[:12], "name": name, **attrs, "steps": {}, "tokens": {}, "error": None}
    _current.set(trace)
    return trace


@contextmanager
def step(name: str):
    """Time a block and add it to the current trace (no-op outside a trace)."""
    start = time.perf_counter()
    try:
        yield
    except Exception as exc:
        trace = _current.get()
        if trace is not None:
            trace["error"] = f"{name}: {type(exc).__name__}: {exc}"
        raise
    finally:
        trace = _current.get()
        if trace is not None:
            ms = (time.perf_counter() - start) * 1000
            trace["steps"][name] = round(trace["steps"].get(name, 0.0) + ms, 2)


def add_tokens(**counts: int) -> None:
    trace = _current.get()
    if trace is not None:
        for key, value in counts.items():
            trace["tokens"][key] = trace["tokens"].get(key, 0) + value


def finish_trace(path: Path | None = None, error: Exception | None = None) -> dict:
    """Close the current trace and append it to ``path`` if given.

    ``error`` marks the request as failed when the failure happened outside a step.
    """
    trace = _current.get() or {}
    _current.set(None)
    if trace and error is not None and not trace["error"]:
        trace["error"] = f"{type(error).__name__}: {error}"
    if path is not None and trace:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(trace) + "\n")
    return trace
