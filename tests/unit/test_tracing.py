import json

import pytest

from app.tracing import add_tokens, finish_trace, start_trace, step


def test_steps_and_tokens_are_recorded(tmp_path):
    path = tmp_path / "traces.jsonl"
    start_trace("ask", strategy="hybrid")
    with step("embed"):
        pass
    add_tokens(prompt=100, completion=20)
    add_tokens(prompt=5)
    trace = finish_trace(path)
    assert trace["strategy"] == "hybrid"
    assert "embed" in trace["steps"]
    assert trace["tokens"] == {"prompt": 105, "completion": 20}
    assert json.loads(path.read_text())["trace_id"] == trace["trace_id"]


def test_failing_step_is_recorded_as_the_error():
    start_trace("search")
    with pytest.raises(ValueError):
        with step("dense"):
            raise ValueError("qdrant down")
    trace = finish_trace()
    assert trace["error"] == "dense: ValueError: qdrant down"
    assert "dense" in trace["steps"]


def test_error_outside_a_step_is_recorded_on_finish():
    start_trace("search")
    trace = finish_trace(error=KeyError("strategy"))
    assert trace["error"].startswith("KeyError")


def test_step_outside_a_trace_does_nothing():
    with step("embed"):
        pass
    assert finish_trace() == {}
