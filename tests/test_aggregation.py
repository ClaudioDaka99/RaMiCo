"""Numerical features are summarised by mean ± std, categorical ones by mode."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent / "golden"))
import _harness as h

from ramico import RaMiCo, metrics
from ramico._validation import numerical_features

NUMERICAL = {"age", "income", "score", "hours"}
CATEGORICAL = list(h.CATEGORICAL)


def fit(X, y, **kwargs):
    explainer = RaMiCo(
        h.ToyModel(2), "random", n_factuals=6, n_counterfactuals=5, random_state=3, isolation_forest="native"
    )
    return explainer.fit(X, y, **kwargs)


@pytest.fixture(scope="module")
def mixed():
    X, y = h.make_dataset(h.Case("aggregation", "mixed", 2, "random", n_samples=140))
    return X, y, fit(X, y, categorical_features=CATEGORICAL)


def test_numerical_features_get_an_interval_on_mixed_data(mixed):
    _, _, rules = mixed
    kinds = rules.to_frame().groupby("feature")["aggregation"].unique().map(tuple).to_dict()
    assert {f for f, kind in kinds.items() if kind == ("mean",)} == NUMERICAL & set(kinds)
    assert {f for f, kind in kinds.items() if kind == ("mode",)} == set(CATEGORICAL) & set(kinds)
    assert rules.metadata["numerical_features"] == ["age", "income", "score", "hours"]
    assert "legacy_aggregation" not in rules.metadata


def test_interval_is_the_mean_and_std_of_the_records(mixed):
    _, _, rules = mixed
    for position, rule in enumerate(rules):
        if rule.aggregation != "mean" or rule.n_records < 2:
            continue
        behind = rules.records_of(position)
        before, after = behind["from_value"].astype(float), behind["to_value"].astype(float)
        assert rule.from_value == pytest.approx(round(before.mean(), 2))
        assert rule.from_std == pytest.approx(round(before.std(), 2))
        assert rule.to_value == pytest.approx(round(after.mean(), 2))
        assert rule.to_std == pytest.approx(round(after.std(), 2))
        assert np.isnan(rule.from_confidence)


def test_the_metrics_of_a_ruleset_and_of_its_records_agree(mixed):
    X, _, rules = mixed
    model = h.ToyModel(2)
    assert metrics.fidelity(model, rules.records, X, k=5) == metrics.fidelity(model, rules, X, k=5)
    assert metrics.evaluate(model, rules, X, k=5).fidelity == metrics.fidelity(model, rules, X, k=5)


def test_numerical_data_give_intervals_only():
    X, y = h.make_dataset(h.Case("aggregation", "numeric", 2, "random", n_samples=120))
    rules = fit(X, y)
    assert len(rules) > 0 and set(rules.to_frame()["aggregation"]) == {"mean"}


def test_booleans_and_declared_categories_are_categorical():
    X = pd.DataFrame({"a": [1.5, 2.0], "b": [1, 2], "flag": [True, False], "code": pd.Series([1, 2], dtype=object)})
    assert numerical_features(X) == ["a", "b"]
