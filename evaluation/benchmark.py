"""Retrieval benchmark: every query of a split, against each strategy, through the /search API."""

import json
import time
from pathlib import Path
from statistics import mean
from typing import Any

import httpx
import numpy as np

from evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank

STRATEGIES = ["vector", "hybrid", "hybrid_rerank"]
KS = (1, 3, 5)
METRICS = [f"P@{k}" for k in KS] + [f"R@{k}" for k in KS] + ["MRR", "NDCG@5"]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score(retrieved: list[str], relevant: set[str]) -> dict[str, float]:
    row = {}
    for k in KS:
        row[f"P@{k}"] = precision_at_k(retrieved, relevant, k)
        row[f"R@{k}"] = recall_at_k(retrieved, relevant, k)
    row["MRR"] = reciprocal_rank(retrieved, relevant)
    row["NDCG@5"] = ndcg_at_k(retrieved, relevant, 5)
    return row


def summarize(rows: list[dict]) -> dict:
    out: dict = {name: round(mean(r[name] for r in rows), 4) for name in METRICS}
    for label, key in (("", "latency_ms"), ("server_", "server_ms")):
        values = [v for r in rows for v in r[key]]
        out[f"{label}p50_ms"] = round(float(np.percentile(values, 50)), 1)
        out[f"{label}p95_ms"] = round(float(np.percentile(values, 95)), 1)
    runs = [steps for r in rows for steps in r["steps_ms"]]
    names = sorted({s for steps in runs for s in steps})
    out["mean_step_ms"] = {s: round(mean(steps.get(s, 0.0) for steps in runs), 1) for s in names}
    return out


def run_split(client: httpx.Client, queries: list[dict], k: int = 5, warmup: int = 3, repeats: int = 3) -> dict:
 
    for query in queries[:warmup]:
        for strategy in STRATEGIES:
            client.post("/search", json={"query": query["query"], "strategy": strategy, "k": k})

    rows: dict[str, dict[str, dict[str, Any]]] = {s: {} for s in STRATEGIES}  # strategy -> query id -> row
    for _ in range(repeats):
        for query in queries:
            for strategy in STRATEGIES:
                started = time.perf_counter()
                response = client.post("/search", json={"query": query["query"], "strategy": strategy, "k": k})
                latency_ms = (time.perf_counter() - started) * 1000
                response.raise_for_status()
                body = response.json()
                row: dict[str, Any] | None = rows[strategy].get(query["id"])
                if row is None:
                    retrieved = [hit["chunk"]["chunk_id"] for hit in body["hits"]]
                    row = rows[strategy][query["id"]] = {
                        "id": query["id"],
                        "type": query["type"],
                        "query": query["query"],
                        "relevant": query["relevant"],
                        "retrieved": retrieved,
                        **score(retrieved, set(query["relevant"])),
                        "latency_ms": [],
                        "server_ms": [],
                        "steps_ms": [],
                    }
                row["latency_ms"].append(round(latency_ms, 1))
                row["server_ms"].append(round(sum(body["timings_ms"].values()), 1))
                row["steps_ms"].append(body["timings_ms"])

    results = {}
    for strategy, by_id in rows.items():
        strategy_rows = list(by_id.values())
        types = sorted({r["type"] for r in strategy_rows})
        results[strategy] = {
            "summary": summarize(strategy_rows),
            "by_type": {t: summarize([r for r in strategy_rows if r["type"] == t]) for t in types},
            "queries": strategy_rows,
        }
    return results


def pareto_front(points: dict[str, tuple[float, float]]) -> list[str]:
    front = []
    for name, (lat, quality) in points.items():
        dominated = any(
            other_lat <= lat and other_q >= quality and (other_lat < lat or other_q > quality)
            for other, (other_lat, other_q) in points.items()
            if other != name
        )
        if not dominated:
            front.append(name)
    return front


def plot_pareto(points: dict[str, tuple[float, float]], front: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, (lat, quality) in points.items():
        ax.scatter(lat, quality, s=80, color="tab:green" if name in front else "tab:gray")
        ax.annotate(name, (lat, quality), textcoords="offset points", xytext=(6, 6))
    ordered = sorted((points[n] for n in front), key=lambda p: p[0])
    ax.plot([p[0] for p in ordered], [p[1] for p in ordered], color="tab:green", linestyle="--")
    ax.set_xlabel("p95 latency (ms, end to end)")
    ax.set_ylabel("NDCG@5 (eval set)")
    ax.set_title("Latency vs quality (green = Pareto front)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
