"""Tests for the numpy Isolation Forest that follows YDF's algorithm."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ramico._isolation_forest import IsolationForest, average_path_length


def _mixed(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rs = np.random.RandomState(seed)
    return pd.DataFrame(
        {
            "x": rs.normal(size=n),
            "y": rs.normal(size=n),
            "colour": pd.Series(rs.choice(["red", "green", "blue"], size=n).tolist(), dtype=object),
        }
    )


def test_average_path_length_matches_ydf_formula():
    assert average_path_length(1) == 0.0
    assert average_path_length(2) == 1.0
    assert math.isclose(float(average_path_length(3)), 2 * (math.log(2) + 0.5772156649) - 4 / 3)
    assert math.isclose(float(average_path_length(256)), 10.2448, abs_tol=1e-4)


def test_scores_are_probabilities_and_deterministic():
    X = _mixed()
    first = IsolationForest(n_trees=200, random_state=0).fit(X).predict(X)
    second = IsolationForest(n_trees=200, random_state=0).fit(X).predict(X)
    other = IsolationForest(n_trees=200, random_state=1).fit(X).predict(X)
    assert ((first > 0) & (first <= 1)).all()
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)
    assert np.corrcoef(first, other)[0, 1] > 0.95


def test_isolated_points_score_higher():
    X = _mixed()
    X.loc[len(X)] = [8.0, -8.0, "red"]
    scores = IsolationForest(n_trees=200, random_state=0).fit(X).predict(X)
    assert scores[-1] == scores.max()


def test_categorical_values_rare_unseen_and_missing():
    X = _mixed()
    X.loc[0, "colour"] = "purple"  # seen once: out of dictionary
    forest = IsolationForest(n_trees=50, random_state=0).fit(X)
    assert "purple" not in forest.vocabulary_[2]
    query = pd.DataFrame({"x": [0.0, np.nan], "y": [0.0, 0.0], "colour": ["never-seen", None]})
    scores = forest.predict(query)
    assert np.isfinite(scores).all()


def test_degenerate_datasets():
    constant = pd.DataFrame({"x": [1.0] * 10, "c": ["a"] * 10})
    np.testing.assert_allclose(IsolationForest(n_trees=10).fit(constant).predict(constant), 0.5)
    tiny = pd.DataFrame({"x": [1.0, 2.0, 3.0]})  # fewer than min_examples: every root is a leaf
    np.testing.assert_allclose(IsolationForest(n_trees=10).fit(tiny).predict(tiny), 0.5)
    single = pd.DataFrame({"x": [1.0]})
    np.testing.assert_array_equal(IsolationForest(n_trees=10).fit(single).predict(single), [0.0])
    with pytest.raises(ValueError, match="empty"):
        IsolationForest().fit(pd.DataFrame({"x": []}))


def test_depth_limit():
    X = _mixed()
    scores = IsolationForest(n_trees=20, max_depth=0, random_state=0).fit(X).predict(X)
    np.testing.assert_allclose(scores, 0.5)


def test_missing_column_at_predict():
    X = _mixed()
    forest = IsolationForest(n_trees=10, random_state=0).fit(X)
    with pytest.raises(ValueError, match="missing"):
        forest.predict(X.drop(columns="y"))


def test_agrees_with_ydf():
    """Same algorithm as YDF: scores rank the samples alike (needs ydf)."""
    ydf = pytest.importorskip("ydf")
    from scipy.stats import spearmanr

    X = _mixed(n=400)
    ours = IsolationForest(random_state=0).fit(X).predict(X)
    theirs = np.asarray(
        ydf.IsolationForestLearner(num_trees=1000, max_depth=-1).train(X, verbose=0).predict(X),
        dtype=float,
    )
    assert spearmanr(ours, theirs).statistic > 0.98
