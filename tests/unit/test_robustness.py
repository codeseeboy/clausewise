import httpx

from evaluation.robustness import check_case, run_robustness

CASE = {"id": "r01", "type": "unsupported", "query": "maternity leave?", "expected_status": "refused"}


def _body(status: str, citations: list[str]) -> dict:
    return {
        "status": status,
        "answer": "text",
        "citations": citations,
        "hits": [],
        "guardrail_notes": [],
        "tokens": {"prompt": 10, "completion": 2},
    }


def test_check_case_reports_each_problem():
    case = {"expected_status": "conflict", "must_cite_all": ["PRC-3.1", "PRCA-1.1"], "must_not_cite": ["TEP-6.2"]}
    reasons = check_case(case, _body("answered", ["PRCA-1.1", "TEP-6.2"]))
    assert len(reasons) == 3
    assert check_case(case, _body("conflict", ["PRC-3.1", "PRCA-1.1"])) == []


def test_a_timeout_fails_that_case_and_the_run_continues():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 2:  # first case after the warm-up
            raise httpx.ReadTimeout("model too slow")
        return httpx.Response(200, json=_body("refused", []))

    client = httpx.Client(base_url="http://backend", transport=httpx.MockTransport(handler))
    results = run_robustness(client, [CASE])
    assert results["vector"]["cases"][0]["reasons"] == ["request failed: ReadTimeout"]
    assert results["hybrid"]["cases"][0]["passed"]
    assert results["hybrid_rerank"]["cases"][0]["passed"]
