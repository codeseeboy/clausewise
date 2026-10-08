"""Ranking metrics with binary relevance: a retrieved id is either relevant or not."""

import math


def _relevant_in_top_k(retrieved: list[str], relevant: set[str], k: int) -> int:
    return sum(1 for item in retrieved[:k] if item in relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Share of the top k results that are relevant: (relevant in top k) / k."""
    return _relevant_in_top_k(retrieved, relevant, k) / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Share of all relevant ids found in the top k: (relevant in top k) / len(relevant).

    Returns 0.0 when there are no relevant ids.
    """
    if not relevant:
        return 0.0
    return _relevant_in_top_k(retrieved, relevant, k) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant result (rank counts from 1), or 0.0 if none is relevant.

    MRR is the mean of this value over all queries.
    """
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Normalised discounted cumulative gain at k.

    DCG  = sum over the top k positions i (from 1) of rel_i / log2(i + 1), rel_i is 1 or 0.
    IDCG = the DCG of a perfect ranking: all relevant ids first, at most k of them.
    NDCG = DCG / IDCG, or 0.0 when there are no relevant ids.
    """
    if not relevant:
        return 0.0
    dcg = sum(1 / math.log2(i + 1) for i, item in enumerate(retrieved[:k], start=1) if item in relevant)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
    return dcg / idcg
