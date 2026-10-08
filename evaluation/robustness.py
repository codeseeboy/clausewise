"""LLM robustness: unsupported, conflicting, distracting and injected cases, through the /ask API.

Each case states the expected status and which clauses must or must not be cited, so the
check is plain code and gives the same verdict for the same reply.
"""

import time
from statistics import mean

import httpx

from evaluation.benchmark import STRATEGIES


def check_case(case: dict, body: dict) -> list[str]:
    """Reasons the reply fails the case; empty when it passes."""
    reasons = []
    if body["status"] != case["expected_status"]:
        reasons.append(f"status '{body['status']}', expected '{case['expected_status']}'")
    cited = set(body["citations"])
    if case.get("must_cite_any") and not cited & set(case["must_cite_any"]):
        reasons.append(f"cites none of {case['must_cite_any']}")
    missing = set(case.get("must_cite_all", [])) - cited
    if missing:
        reasons.append(f"does not cite {sorted(missing)}")
    forbidden = cited & set(case.get("must_not_cite", []))
    if forbidden:
        reasons.append(f"cites {sorted(forbidden)}")
    return reasons


def run_robustness(client: httpx.Client, cases: list[dict]) -> dict:
    # Load the model into memory first, so the first case does not carry the load time.
    try:
        client.post("/ask", json={"query": cases[0]["query"], "strategy": "vector"}, timeout=900)
    except httpx.HTTPError:
        pass  # the cases below report the failure if the model stays unavailable

    results = {}
    for strategy in STRATEGIES:
        rows: list[dict] = []
        for case in cases:
            started = time.perf_counter()
            try:
                response = client.post("/ask", json={"query": case["query"], "strategy": strategy})
            except httpx.HTTPError as exc:  # timeout or connection error: record it, keep going
                response, error = None, f"request failed: {type(exc).__name__}"
            else:
                error = None if response.status_code == 200 else f"HTTP {response.status_code}"
            latency_ms = round((time.perf_counter() - started) * 1000, 1)
            if response is None or error:
                rows.append({**case, "passed": False, "reasons": [error], "latency_ms": latency_ms})
                continue
            body = response.json()
            reasons = check_case(case, body)
            rows.append(
                {
                    **case,
                    "passed": not reasons,
                    "reasons": reasons,
                    "status": body["status"],
                    "answer": body["answer"],
                    "citations": body["citations"],
                    "retrieved": [hit["chunk"]["chunk_id"] for hit in body["hits"]],
                    "guardrail_notes": body["guardrail_notes"],
                    "tokens": body["tokens"],
                    "latency_ms": latency_ms,
                }
            )
        unsupported = [r for r in rows if r["type"] == "unsupported"]
        results[strategy] = {
            "pass_rate": round(mean(r["passed"] for r in rows), 3),
            "correct_refusal_rate": round(mean(r.get("status") == "refused" for r in unsupported), 3) if unsupported else None,
            "mean_prompt_tokens": round(mean(r.get("tokens", {}).get("prompt", 0) for r in rows), 1),
            "mean_completion_tokens": round(mean(r.get("tokens", {}).get("completion", 0) for r in rows), 1),
            "mean_latency_ms": round(mean(r["latency_ms"] for r in rows), 1),
            "cases": rows,
        }
    return results
