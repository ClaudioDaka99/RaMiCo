"""Tests for ramico.benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent / "golden"))
import _harness as h

from ramico import benchmark
from ramico.exceptions import DataError


class FittedToy(h.ToyModel):
    """The deterministic toy model with a no-op fit, as run_benchmark expects."""

    def fit(self, X, y):
        return self


def _data():
    case = h.Case("bench", "mixed", 2, "random", n_samples=120)
    X, y = h.make_dataset(case)
    return X, y


def test_paper_seeds():
    seeds = benchmark.paper_seeds()
    assert len(seeds) == 30 and len(set(seeds)) == 30
    assert seeds[:6] == [42, 123, 456, 789, 101112, 131415]
    assert benchmark.paper_seeds(8) == seeds[:8]


def test_splits_are_disjoint_and_reproducible():
    X, y = _data()
    train, test = benchmark.balanced_split(0.1)(X, y, 3)
    assert len(np.intersect1d(train, test)) == 0 and len(train) + len(test) == len(y)
    assert (pd.Series(y[test]).value_counts() == 12).all()
    again = benchmark.balanced_split(0.1)(X, y, 3)
    np.testing.assert_array_equal(test, again[1])
    pytest.importorskip("sklearn")
    train, test = benchmark.stratified_split(0.25)(X, y, 3)
    assert len(test) == 30 and len(np.intersect1d(train, test)) == 0


def test_run_benchmark_and_cache(tmp_path, monkeypatch):
    X, y = _data()
    kwargs = dict(
        strategies=["random", "boundary"],
        seeds=[1, 2],
        n_factuals=3,
        n_counterfactuals=3,
        split=benchmark.balanced_split(0.1),
        k=3,
        isolation_forest="native",
        categorical_features=["color", "segment", "flag"],
        cache_dir=tmp_path,
        progress=False,
    )
    results = benchmark.run_benchmark(lambda seed: FittedToy(2), X, y, **kwargs)
    assert len(results) == 4
    assert set(results["strategy"]) == {"random", "boundary"}
    assert results[["fidelity", "incomprehensibility", "implausibility"]].notna().all().all()
    assert len(list(tmp_path.glob("*.json"))) == 4

    def fail(*args, **kwargs):
        raise AssertionError("the cached results should have been reused")

    monkeypatch.setattr(benchmark.RaMiCo, "fit", fail)
    cached = benchmark.run_benchmark(lambda seed: FittedToy(2), X, y, **kwargs)
    pd.testing.assert_frame_equal(cached, results)
    with pytest.raises(AssertionError, match="should have been reused"):  # other settings: recomputed
        benchmark.run_benchmark(lambda seed: FittedToy(2), X, y, **{**kwargs, "mode": "or"})


def test_benchmark_rejects_mismatched_labels():
    X, y = _data()
    with pytest.raises(DataError):
        benchmark.run_benchmark(lambda seed: FittedToy(2), X, y[:-1], seeds=[1], progress=False)


def test_benjamini_hochberg_matches_statsmodels():
    multitest = pytest.importorskip("statsmodels.stats.multitest")
    p = np.array([0.001, 0.04, 0.03, 0.2, 0.04, 0.9, 0.0001])
    expected = multitest.multipletests(p, alpha=0.05, method="fdr_bh")[1]
    np.testing.assert_allclose(benchmark.benjamini_hochberg(p), expected, rtol=1e-12)
    assert benchmark.benjamini_hochberg([]).size == 0


def test_compare_strategies():
    scipy_stats = pytest.importorskip("scipy.stats")
    rs = np.random.RandomState(0)
    rows = []
    for strategy, shift in [("ensemble", 0.0), ("random", -10.0), ("prototype", 0.1)]:
        for seed in range(15):
            rows.append(
                {
                    "seed": seed,
                    "strategy": strategy,
                    "fidelity": 80 + shift + rs.normal(),
                    "incomprehensibility": 3 + shift / 10 + rs.normal(scale=0.1),
                    "implausibility": 0.5 + rs.normal(scale=0.01),
                }
            )
    results = pd.DataFrame(rows)
    table = benchmark.compare_strategies(results)
    assert len(table) == 6
    random_fidelity = table[(table.metric == "fidelity") & (table.strategy == "random")].iloc[0]
    assert random_fidelity.significant and not random_fidelity.better
    random_readability = table[(table.metric == "incomprehensibility") & (table.strategy == "random")].iloc[0]
    assert random_readability.significant and random_readability.better
    raw = scipy_stats.mannwhitneyu(
        results.loc[results.strategy == "ensemble", "fidelity"],
        results.loc[results.strategy == "random", "fidelity"],
        alternative="two-sided",
    ).pvalue
    assert random_fidelity.p_value == pytest.approx(raw)
    with pytest.raises(DataError):
        benchmark.compare_strategies(results, reference="nope")
