"""Run the retrieval benchmark and the LLM robustness check against a running backend.

Writes raw JSON and readable Markdown tables to results/.
Usage: python -m evaluation.run [--skip-llm]
"""

import json
import os
import sys
import time
from pathlib import Path
from statistics import mean

import httpx

from evaluation.benchmark import METRICS, STRATEGIES, load_jsonl, pareto_front, plot_pareto, run_split
from evaluation.robustness import run_robustness

ROOT = Path(__file__).resolve().parents[1]
QUERIES = ROOT / "data" / "queries"
RESULTS = ROOT / "results"
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
RETRIEVAL_STEPS = ["embed", "dense", "bm25", "rrf", "rerank"]


def wait_for_backend(client: httpx.Client, timeout_s: int = 900) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if client.get("/health").status_code == 200:
                return
        except httpx.TransportError:
            pass
        time.sleep(5)
    raise SystemExit(f"backend at {BACKEND_URL} did not become healthy")


def summarize_traces(path: Path, skip_lines: int) -> dict:
    """Request count, error rate and mean step time per endpoint and strategy, for this run only."""
    if not path.exists():
        return {}
    traces = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()[skip_lines:] if line]
    groups: dict[str, list[dict]] = {}
    for trace in traces:
        groups.setdefault(f"{trace['name']}:{trace.get('strategy', '')}", []).append(trace)
    summary = {}
    for key, items in sorted(groups.items()):
        errors = sum(1 for t in items if t["error"])
        steps = sorted({s for t in items for s in t["steps"]})
        summary[key] = {
            "requests": len(items),
            "errors": errors,
            "error_rate": round(errors / len(items), 3),
            "mean_step_ms": {s: round(mean(t["steps"].get(s, 0.0) for t in items), 1) for s in steps},
        }
    return summary


def _table(header: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines + [""]


def write_benchmark_md(dev: dict, evl: dict, index_report: dict, front: list[str], path: Path) -> None:
    lines = ["# Retrieval benchmark", ""]
    for title, split in (("Development set", dev), ("Evaluation set", evl)):
        lines += [f"## {title}", ""]
        rows = [
            [s, *(f"{split[s]['summary'][m]:.3f}" for m in METRICS), split[s]["summary"]["p50_ms"], split[s]["summary"]["p95_ms"]]
            for s in STRATEGIES
        ]
        lines += _table(["strategy", *METRICS, "p50 ms", "p95 ms"], rows)

    types = sorted(evl[STRATEGIES[0]]["by_type"])
    lines += ["## NDCG@5 by query type (evaluation set)", ""]
    lines += _table(["strategy", *types], [[s, *(f"{evl[s]['by_type'][t]['NDCG@5']:.3f}" for t in types)] for s in STRATEGIES])

    lines += ["## Mean time per step in the backend (ms, evaluation set)", ""]
    steps = RETRIEVAL_STEPS
    lines += _table(["strategy", *steps], [[s, *(evl[s]["summary"]["mean_step_ms"].get(st, "-") for st in steps)] for s in STRATEGIES])

    per_chunk_ms = 1000 * index_report["embed_seconds"] / max(index_report["chunks"], 1)
    lines += [
        "## Index cost",
        "",
        f"- files: {index_report['files']}, chunks: {index_report['chunks']}, failed: {index_report['failed'] or 'none'}",
        f"- embedding all chunks: {index_report['embed_seconds']:.2f} s ({per_chunk_ms:.1f} ms per chunk)",
        f"- BM25 build: {index_report['bm25_build_seconds'] * 1000:.1f} ms",
        f"- total indexing: {index_report['total_seconds']:.2f} s",
        "",
        f"Pareto front (p95 latency vs NDCG@5, evaluation set): **{', '.join(front)}**",
        "",
        "## Evaluation queries where the first result was not relevant",
        "",
    ]
    misses = [
        [r["id"], r["type"], s, ", ".join(r["relevant"]), ", ".join(r["retrieved"][:3])]
        for s in STRATEGIES
        for r in evl[s]["queries"]
        if r["MRR"] < 1.0
    ]
    lines += _table(["query", "type", "strategy", "expected", "top 3 retrieved"], misses) if misses else ["None.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_robustness_md(results: dict, path: Path) -> None:
    lines = ["# LLM robustness", ""]
    rows = [
        [s, r["pass_rate"], r["correct_refusal_rate"], r["mean_prompt_tokens"], r["mean_completion_tokens"], r["mean_latency_ms"]]
        for s, r in results.items()
    ]
    lines += _table(["strategy", "pass rate", "correct refusal rate", "prompt tokens", "completion tokens", "mean ms"], rows)
    lines += ["## Failed cases", ""]
    failed = [
        [c["id"], c["type"], s, c.get("status", "-"), ", ".join(c.get("citations", [])), "; ".join(c["reasons"])]
        for s, r in results.items()
        for c in r["cases"]
        if not c["passed"]
    ]
    lines += _table(["case", "type", "strategy", "status", "citations", "why"], failed) if failed else ["None.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _dump(data: dict, name: str) -> None:
    (RESULTS / name).write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    trace_file = RESULTS / "traces.jsonl"
    skip = len(trace_file.read_text(encoding="utf-8").splitlines()) if trace_file.exists() else 0

    with httpx.Client(base_url=BACKEND_URL, timeout=300) as client:
        wait_for_backend(client)
        index_report = client.get("/index/report").json()
        dev = run_split(client, load_jsonl(QUERIES / "dev.jsonl"))
        evl = run_split(client, load_jsonl(QUERIES / "eval.jsonl"))

        points = {s: (evl[s]["summary"]["p95_ms"], evl[s]["summary"]["NDCG@5"]) for s in STRATEGIES}
        front = pareto_front(points)
        plot_pareto(points, front, RESULTS / "pareto.png")
        _dump({"index": index_report, "results": dev}, "benchmark_dev.json")
        _dump({"index": index_report, "pareto_front": front, "results": evl}, "benchmark_eval.json")
        write_benchmark_md(dev, evl, index_report, front, RESULTS / "benchmark.md")
        print((RESULTS / "benchmark.md").read_text(encoding="utf-8"))

        if "--skip-llm" not in sys.argv:
            robust = run_robustness(client, load_jsonl(QUERIES / "robustness.jsonl"))
            _dump(robust, "robustness.json")
            write_robustness_md(robust, RESULTS / "robustness.md")
            print((RESULTS / "robustness.md").read_text(encoding="utf-8"))

    _dump(summarize_traces(trace_file, skip), "observability.json")


if __name__ == "__main__":
    main()
