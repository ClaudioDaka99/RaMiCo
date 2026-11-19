"""Tests of the metrics: accepted inputs, errors and consistency between the functions.

Their values are frozen by the golden tests.
"""

from __future__ import annotations

import copy
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "golden"))
sys.path.insert(0, str(Path(__file__).parent))
import _harness as h

import _metrics_reference as reference
from ramico import RaMiCo, metrics
from ramico.exceptions import DataError


@pytest.fixture(scope="module")
def setup():
    X, y = h.make_dataset(h.Case("metrics", "mixed", 2, "random", n_samples=120))
    model = h.ToyModel(2)
    rules = RaMiCo(
        model, strategy="random", n_factuals=5, n_counterfactuals=4, random_state=0, isolation_forest="native"
    ).fit(X, y, categorical_features=list(h.CATEGORICAL))
    return model, rules, X


@pytest.mark.parametrize("mode", metrics.MODES)
def test_evaluate_gathers_the_single_metrics(setup, mode):
    model, rules, X = setup
    scores = metrics.evaluate(model, rules, X, k=3, mode=mode)
    assert scores["fidelity"] == metrics.fidelity(model, rules, X, k=3, mode=mode)
    assert scores["incomprehensibility"] == metrics.incomprehensibility(model, rules, X, k=3)
    assert scores["implausibility"] == metrics.implausibility(rules)
    assert 0 <= scores["fidelity"] <= 100 and scores["incomprehensibility"] >= 1


def test_scores_read_by_name(setup):
    model, rules, X = setup
    scores = metrics.evaluate(model, rules, X, k=3)
    assert isinstance(scores, metrics.Scores) and list(scores) == ["fidelity", "incomprehensibility", "implausibility"]
    assert scores.fidelity == scores["fidelity"] and scores.incomprehensibility == scores["incomprehensibility"]
    assert scores.implausibility == scores["implausibility"]
    assert all(type(value) is float for value in scores.values())  # plain floats, not numpy scalars
    assert scores == dict(scores) and len(scores) == 3
    assert {"fidelity", "incomprehensibility", "implausibility"} <= set(dir(scores))  # editors complete them
    assert repr(scores) == (
        f"Scores(fidelity={scores.fidelity!r}, incomprehensibility={scores.incomprehensibility!r}, "
        f"implausibility={scores.implausibility!r})"
    )
    with pytest.raises(AttributeError, match="no metric named 'fidelty' \\(available: fidelity, "):
        _ = scores.fidelty
    with pytest.raises(AttributeError, match="read-only"):
        scores.fidelity = 0.0
    with pytest.raises(TypeError):
        scores["fidelity"] = 0.0
    assert pickle.loads(pickle.dumps(scores)) == scores and copy.deepcopy(scores) == scores


def test_scores_with_any_names():
    scores = metrics.Scores({"fidelity": 90.0}, k=7)
    assert scores.k == 7 and dict(scores) == {"fidelity": 90.0, "k": 7}
    assert repr(metrics.Scores({"mode or": 1.0})) == "Scores({'mode or': 1.0})"
    assert len(metrics.Scores()) == 0 and repr(metrics.Scores()) == "Scores()"
    with pytest.raises(AttributeError, match="available: none"):
        _ = metrics.Scores().fidelity


@pytest.mark.parametrize("case", h.CASES, ids=lambda case: case.name)
def test_batched_metrics_equal_one_sample_at_a_time(case):
    X, y = h.make_dataset(case)
    model = h.ToyModel(case.n_classes)
    rules = RaMiCo(
        model,
        strategy=h.STRATEGY_NAMES[case.strategy],
        n_factuals=case.n_factuals,
        n_counterfactuals=case.n_counterfactuals,
        random_state=case.seed,
        isolation_forest="native",
    ).fit(X, y, categorical_features=h.categorical_features(case))
    for k in (1, 3, 7):
        for mode in metrics.MODES:
            expected = reference.scores(model, rules, X, k=k, mode=mode)
            got = (metrics.fidelity(model, rules, X, k=k, mode=mode), metrics.incomprehensibility(model, rules, X, k=k))
            assert got == expected, (k, mode)


def test_batched_metrics_on_heart_disease():
    from sklearn.compose import make_column_transformer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    from ramico import datasets

    data = datasets.load_heart_disease(download=False) if _cached("heart_disease") else None
    if data is None:
        pytest.skip("Heart Disease is not in the local cache")
    X, y = data
    numerical = [c for c in X.columns if c not in data.categorical_features]
    model = make_pipeline(
        make_column_transformer(
            (OneHotEncoder(handle_unknown="ignore"), data.categorical_features), (StandardScaler(), numerical)
        ),
        DecisionTreeClassifier(max_depth=4, random_state=0),
    ).fit(X, y)
    rules = RaMiCo(model, n_factuals=20, n_counterfactuals=10, random_state=0, isolation_forest="native").fit(X, y)
    for mode in metrics.MODES:
        expected = reference.scores(model, rules, X, mode=mode)
        assert (metrics.fidelity(model, rules, X, mode=mode), metrics.incomprehensibility(model, rules, X)) == expected


def _cached(name: str) -> bool:
    from ramico import datasets

    try:
        getattr(datasets, f"load_{name}")(download=False)
    except Exception:
        return False
    return True


@pytest.mark.parametrize("seed", range(5))
def test_first_k_is_the_head_of_a_stable_argsort(seed):
    rng = np.random.default_rng(seed)
    distances = rng.choice([0.1, 0.2, 0.3, np.nan], size=50).astype(np.float32)  # many ties
    for k in (1, 7, 49, 50, 60):
        expected = np.argsort(distances, kind="stable")[:k]
        np.testing.assert_array_equal(metrics._first_k(distances, k), expected)


def test_blocks_do_not_change_the_scores(setup, monkeypatch):
    model, rules, X = setup
    reference = metrics.evaluate(model, rules, X)
    original = metrics._nearest_records
    monkeypatch.setattr(
        metrics, "_nearest_records", lambda *args: original(*args, max_cells=50)
    )  # blocks of one or two rows
    assert metrics.evaluate(model, rules, X) == reference


def test_records_give_the_same_scores_as_the_rules(setup):
    model, rules, X = setup
    from_records, from_rules = metrics.evaluate(model, rules.records, X, k=3), metrics.evaluate(model, rules, X, k=3)
    assert from_records["fidelity"] == from_rules["fidelity"]
    assert from_records["incomprehensibility"] == from_rules["incomprehensibility"]
    # the rules round the mean implausibility of each rule to 3 decimals, as the original aggregation
    assert from_records["implausibility"] == pytest.approx(from_rules["implausibility"], abs=5e-4)


def test_category_columns_are_categorical(setup):
    model, rules, X = setup
    as_category = X.astype({"color": "category", "segment": "category"})
    assert metrics.evaluate(model, rules, as_category, k=3) == pytest.approx(metrics.evaluate(model, rules, X, k=3))


def test_rules_without_records(setup):
    model, rules, X = setup
    empty = rules.filter(feature="no such feature")
    assert metrics.fidelity(model, empty, X) == 0
    assert metrics.incomprehensibility(model, empty, X) == 0
    assert np.isnan(metrics.implausibility(empty))


@pytest.mark.parametrize("function", [metrics.evaluate, metrics.fidelity, metrics.incomprehensibility])
@pytest.mark.parametrize("k", [0, 2.5])
def test_invalid_k(setup, function, k):
    model, rules, X = setup
    with pytest.raises(ValueError, match="k must be a positive integer"):
        function(model, rules, X, k=k)


def test_invalid_inputs(setup):
    model, rules, X = setup
    with pytest.raises(ValueError, match="mode must be one of"):
        metrics.fidelity(model, rules, X, mode="xor")
    with pytest.raises(DataError, match="X must be a DataFrame, or an array"):
        metrics.fidelity(model, rules, X.to_numpy()[:, :-1])  # one column short
    with pytest.raises(DataError, match="X must be a DataFrame, or an array"):
        metrics.fidelity(model, rules.records, X.to_numpy())  # records alone do not name the columns
    with pytest.raises(TypeError, match="rules must be a RuleSet or a DataFrame"):
        metrics.fidelity(model, [1, 2], X)
    with pytest.raises(DataError, match=r"the records lack the columns \['to_value'\]"):
        metrics.fidelity(model, rules.records.drop(columns="to_value"), X)
    with pytest.raises(DataError, match="'implausibility' column"):
        metrics.implausibility(rules.records.drop(columns="implausibility"))
    with pytest.raises(TypeError, match="rules must be a RuleSet or a DataFrame"):
        metrics.implausibility("rules")
