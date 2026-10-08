"""Reciprocal Rank Fusion: merge several ranked lists of ids into one list."""


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked lists of ids.

    Each list is ordered best first. An id at rank r (counting from 1) in a list gets
    1 / (k + r) from that list, and its scores from all lists are added. An id that is
    missing from a list gets nothing from that list.

    Returns (id, score) pairs sorted by score, highest first. When two ids have the
    same score, the one seen first (first list first, then by rank) comes first.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    # sorted() is stable, so ids with equal scores keep the order they were first seen in.
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
