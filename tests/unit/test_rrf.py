from pytest import approx

from app.retrieval.rrf import reciprocal_rank_fusion


def test_single_list_scores_by_rank():
    result = reciprocal_rank_fusion([["a", "b"]], k=60)
    assert result == [("a", approx(1 / 61)), ("b", approx(1 / 62))]


def test_id_in_both_lists_beats_top_of_one_list():
    result = dict(reciprocal_rank_fusion([["a", "b", "c"], ["c", "d"]], k=60))
    assert result["c"] == approx(1 / 63 + 1 / 61)
    assert result["c"] > result["a"]


def test_id_missing_from_a_list_gets_nothing_from_it():
    result = dict(reciprocal_rank_fusion([["a"], ["b", "d"]], k=60))
    assert result["d"] == approx(1 / 62)


def test_output_is_sorted_best_first():
    # a = 1/61, b = 1/62 + 1/61, c = 1/63 + 1/62
    result = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c"]], k=60)
    assert [item for item, _ in result] == ["b", "c", "a"]


def test_ties_keep_first_seen_order():
    assert [i for i, _ in reciprocal_rank_fusion([["a"], ["b"]], k=60)] == ["a", "b"]


def test_empty_input():
    assert reciprocal_rank_fusion([], k=60) == []
    assert reciprocal_rank_fusion([[], []], k=60) == []
