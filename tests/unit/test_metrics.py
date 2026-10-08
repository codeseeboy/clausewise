import math

from pytest import approx

from evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank

RETRIEVED = ["a", "x", "b", "y", "z"]
RELEVANT = {"a", "b", "c"}


def test_precision_at_k():
    assert precision_at_k(RETRIEVED, RELEVANT, 1) == 1.0
    assert precision_at_k(RETRIEVED, RELEVANT, 3) == approx(2 / 3)
    assert precision_at_k(RETRIEVED, RELEVANT, 5) == approx(2 / 5)


def test_precision_divides_by_k_even_when_fewer_results():
    assert precision_at_k(["a"], RELEVANT, 5) == approx(1 / 5)


def test_recall_at_k():
    assert recall_at_k(RETRIEVED, RELEVANT, 1) == approx(1 / 3)
    assert recall_at_k(RETRIEVED, RELEVANT, 5) == approx(2 / 3)
    assert recall_at_k(RETRIEVED, set(), 5) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(RETRIEVED, RELEVANT) == 1.0
    assert reciprocal_rank(["x", "y", "b"], RELEVANT) == approx(1 / 3)
    assert reciprocal_rank(["x", "y"], RELEVANT) == 0.0


def test_ndcg_perfect_and_worst():
    assert ndcg_at_k(["a", "b", "c"], RELEVANT, 5) == approx(1.0)
    assert ndcg_at_k(["x", "y"], RELEVANT, 5) == 0.0
    assert ndcg_at_k(["a"], set(), 5) == 0.0


def test_ndcg_by_hand():
    # Relevant at positions 1 and 3; three relevant ids exist.
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(RETRIEVED, RELEVANT, 5) == approx(dcg / idcg)
