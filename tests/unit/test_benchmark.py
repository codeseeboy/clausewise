from pytest import approx

from evaluation.benchmark import pareto_front, score, summarize


def test_pareto_front_drops_points_that_are_slower_and_worse():
    points = {"fast": (50.0, 0.80), "best": (1000.0, 0.95), "slow_and_worse": (400.0, 0.70)}
    assert pareto_front(points) == ["fast", "best"]


def test_equal_points_both_stay_on_the_front():
    assert pareto_front({"a": (10.0, 0.5), "b": (10.0, 0.5)}) == ["a", "b"]


def test_score_has_every_metric():
    row = score(["a", "x"], {"a"})
    assert row["P@1"] == 1.0
    assert row["MRR"] == 1.0
    assert set(row) == {"P@1", "P@3", "P@5", "R@1", "R@3", "R@5", "MRR", "NDCG@5"}


def test_summarize_uses_every_timed_run():
    base = score(["a"], {"a"})
    rows = [
        {**base, "latency_ms": [10.0, 30.0], "server_ms": [8.0, 9.0], "steps_ms": [{"embed": 5.0}, {"embed": 7.0}]},
        {**base, "latency_ms": [20.0, 40.0], "server_ms": [8.0, 9.0], "steps_ms": [{"embed": 6.0}, {"embed": 6.0}]},
    ]
    summary = summarize(rows)
    assert summary["p50_ms"] == approx(25.0)
    assert summary["mean_step_ms"] == {"embed": 6.0}
    assert summary["NDCG@5"] == 1.0
