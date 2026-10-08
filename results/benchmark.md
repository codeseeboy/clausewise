# Retrieval benchmark

## Development set

| strategy | P@1 | P@3 | P@5 | R@1 | R@3 | R@5 | MRR | NDCG@5 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| vector | 0.875 | 0.333 | 0.200 | 0.875 | 1.000 | 1.000 | 0.938 | 0.954 | 25.2 | 28.7 |
| hybrid | 0.875 | 0.333 | 0.200 | 0.875 | 1.000 | 1.000 | 0.917 | 0.938 | 25.9 | 32.5 |
| hybrid_rerank | 0.625 | 0.333 | 0.200 | 0.625 | 1.000 | 1.000 | 0.812 | 0.862 | 1007.9 | 1141.9 |

## Evaluation set

| strategy | P@1 | P@3 | P@5 | R@1 | R@3 | R@5 | MRR | NDCG@5 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| vector | 0.750 | 0.361 | 0.233 | 0.667 | 0.917 | 1.000 | 0.854 | 0.878 | 54.9 | 69.7 |
| hybrid | 0.667 | 0.278 | 0.200 | 0.625 | 0.792 | 0.917 | 0.767 | 0.794 | 62.9 | 82.7 |
| hybrid_rerank | 0.833 | 0.389 | 0.233 | 0.750 | 1.000 | 1.000 | 0.903 | 0.928 | 954.3 | 1113.5 |

## NDCG@5 by query type (evaluation set)

| strategy | exact_id | negation | numeric | semantic |
|---|---|---|---|---|
| vector | 0.754 | 0.973 | 1.000 | 0.783 |
| hybrid | 1.000 | 0.959 | 0.796 | 0.421 |
| hybrid_rerank | 1.000 | 1.000 | 0.833 | 0.877 |

## Mean time per step in the backend (ms, evaluation set)

| strategy | embed | dense | bm25 | rrf | rerank |
|---|---|---|---|---|---|
| vector | 42.1 | 3.1 | - | - | - |
| hybrid | 46.4 | 3.2 | 0.3 | 0.0 | - |
| hybrid_rerank | 49.3 | 3.2 | 0.2 | 0.0 | 913.7 |

## Index cost

- files: 6, chunks: 52, failed: none
- embedding all chunks: 1.07 s (20.6 ms per chunk)
- BM25 build: 10.4 ms
- total indexing: 1.21 s

Pareto front (p95 latency vs NDCG@5, evaluation set): **vector, hybrid_rerank**

## Evaluation queries where the first result was not relevant

| query | type | strategy | expected | top 3 retrieved |
|---|---|---|---|---|
| e01 | exact_id | vector | ACP-4.2 | PRC-4.1, ACP-4.2, PRC-3.2 |
| e03 | exact_id | vector | PRC-4.1 | PRC-1.1, PRC-4.1, PRC-3.1 |
| e11 | semantic | vector | IRS-5.1 | IRS-2.1, IRS-2.3, IRS-1.1 |
| e09 | numeric | hybrid | TEP-4.2.3 | TEP-4.1, TEP-1.1, TEP-6.2 |
| e10 | semantic | hybrid | ACP-4.2 | ACP-3.2, ACP-4.2, ACP-4.3 |
| e11 | semantic | hybrid | IRS-5.1 | IRS-2.3, IRS-5.1, IRS-2.1 |
| e12 | semantic | hybrid | PRC-3.1, PRCA-1.1 | PRC-4.1, PRCA-1.2, PRC-2.1 |
| e09 | numeric | hybrid_rerank | TEP-4.2.3 | TEP-4.2.2, TEP-4.2.1, TEP-4.2.3 |
| e11 | semantic | hybrid_rerank | IRS-5.1 | IRS-2.3, IRS-5.1, IRS-2.1 |
