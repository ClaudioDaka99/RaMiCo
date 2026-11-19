"""Tests of the mining strategies and of their registry."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent / "golden"))
import _harness as h

from ramico import RaMiCo, strategies
from ramico._distance import gower_matrix
from ramico.strategies import (
    BoundaryStrategy,
    EnsembleStrategy,
    MiningStrategy,
    NearestBoundaryStrategy,
    Pair,
    PrototypeStrategy,
    RandomPairsStrategy,
    RandomStrategy,
    available_strategies,
    get_strategy,
    register_strategy,
)

SINGLE_STRATEGIES = [RandomStrategy, BoundaryStrategy, NearestBoundaryStrategy, PrototypeStrategy, RandomPairsStrategy]


@pytest.fixture(scope="module")
def data():
    return h.make_dataset(h.Case("strategies", "mixed", 2, "random", n_samples=80))


@pytest.fixture(scope="module")
def spaces(data):
    X, y = data
    return X[y == 0], X[y != 0]


@pytest.fixture
def registry(monkeypatch):
    """A copy of the registry, restored after the test."""
    monkeypatch.setattr(strategies, "_REGISTRY", dict(strategies._REGISTRY))


def select(strategy, spaces, s=4, k=3, rng=None):
    X_c, X_not_c = spaces
    return strategy.select(X_c, X_not_c, s, k, gower_matrix, rng if rng is not None else np.random.RandomState(0))


@pytest.mark.parametrize("factory", SINGLE_STRATEGIES)
def test_pairs_point_into_the_two_spaces(factory, spaces):
    X_c, X_not_c = spaces
    pairs = select(factory(), spaces)
    assert len(pairs) == 4 and len({pair.factual for pair in pairs}) == 4
    for pair in pairs:
        assert isinstance(pair, Pair) and 0 <= pair.factual < len(X_c)
        assert len(pair.counterfactuals) == 3
        assert ((pair.counterfactuals >= 0) & (pair.counterfactuals < len(X_not_c))).all()


@pytest.mark.parametrize("factory", SINGLE_STRATEGIES)
def test_same_seed_same_pairs(factory, spaces):
    first = select(factory(), spaces, rng=np.random.RandomState(3))
    again = select(factory(), spaces, rng=np.random.RandomState(3))
    assert [p.factual for p in first] == [p.factual for p in again]
    assert all((p.counterfactuals == q.counterfactuals).all() for p, q in zip(first, again, strict=True))


def test_numpy_generators_are_accepted(spaces):
    assert len(select(RandomStrategy(), spaces, rng=np.random.default_rng(0))) == 4


def test_counterfactuals_are_the_nearest_samples_of_the_other_classes(spaces):
    X_c, X_not_c = spaces
    pairs = select(RandomStrategy(), spaces)
    distances = gower_matrix(X_not_c, X_c.iloc[[p.factual for p in pairs]])
    for i, pair in enumerate(pairs):
        np.testing.assert_array_equal(pair.counterfactuals, np.argsort(distances[:, i], kind="stable")[:3])


def test_boundary_factuals_come_from_the_closest_candidates(spaces):
    X_c, X_not_c = spaces
    closest = np.argsort(gower_matrix(X_c, X_not_c).min(axis=1), kind="stable")
    assert {p.factual for p in select(BoundaryStrategy(), spaces)} <= set(closest[:6])  # int(1.5 * s)
    assert [p.factual for p in select(NearestBoundaryStrategy(), spaces)] == list(closest[:4])


def test_boundary_with_too_few_candidates(spaces, caplog):
    with caplog.at_level(logging.WARNING, logger="ramico.strategies"):
        pairs = select(BoundaryStrategy(n_candidates=2), spaces)
    assert len(pairs) == 2 and "Only 2 candidate factuals" in caplog.text
    assert repr(BoundaryStrategy(n_candidates=2)) == "BoundaryStrategy(n_candidates=2)"
    assert repr(BoundaryStrategy()) == "BoundaryStrategy()"


def test_prototypes_share_their_counterfactuals(spaces):
    pairs = select(PrototypeStrategy(), spaces)
    assert all((p.counterfactuals == pairs[0].counterfactuals).all() for p in pairs)


def test_sizes_are_clipped_to_the_spaces(spaces):
    X_c, X_not_c = spaces
    assert len(select(RandomStrategy(), spaces, s=len(X_c) + 5)) == len(X_c)
    pairs = select(RandomPairsStrategy(), spaces, k=len(X_not_c) + 5)
    assert all(len(p.counterfactuals) == len(X_not_c) for p in pairs)


def test_ensemble_components(spaces):
    ensemble = EnsembleStrategy()
    assert [name for name, _ in ensemble.components()] == ["prototype", "nearest-boundary", "random-pairs"]
    assert len(select(ensemble, spaces)) == 12  # 4 factuals per component
    nested = EnsembleStrategy([RandomStrategy(), EnsembleStrategy([BoundaryStrategy()])])
    assert [name for name, _ in nested.components()] == ["random", "boundary"]
    assert repr(nested) == (
        "EnsembleStrategy(strategies=[RandomStrategy(), EnsembleStrategy(strategies=[BoundaryStrategy()])])"
    )


def test_registry(registry):
    assert available_strategies() == ["boundary", "ensemble", "prototype", "random"]
    assert isinstance(get_strategy("prototype"), PrototypeStrategy)
    instance = RandomStrategy()
    assert get_strategy(instance) is instance
    with pytest.raises(ValueError, match="already registered"):
        register_strategy("random", RandomPairsStrategy)
    register_strategy("random", RandomPairsStrategy, overwrite=True)
    assert isinstance(get_strategy("random"), RandomPairsStrategy)


def test_custom_strategy_by_name(registry, data):
    class FirstSamples(MiningStrategy):
        name = "first"

        def select(self, X_c, X_not_c, s, k, distance, rng):
            counterfactuals = np.arange(min(k, len(X_not_c)))
            return [Pair(i, counterfactuals) for i in range(min(s, len(X_c)))]

    register_strategy("first", FirstSamples)
    assert "first" in available_strategies() and repr(FirstSamples()) == "FirstSamples()"
    X, y = data
    rules = RaMiCo(
        h.ToyModel(2), strategy="first", n_factuals=3, n_counterfactuals=4, random_state=0, isolation_forest="native"
    ).fit(X, y, categorical_features=list(h.CATEGORICAL))
    assert rules.strategy == "first" and set(rules.to_frame()["strategy"]) == {"first"}


def test_pairs_are_the_pairs_of_fit(data):
    X, y = data
    explainer = RaMiCo(
        h.ToyModel(2), "ensemble", n_factuals=4, n_counterfactuals=3, random_state=5, isolation_forest="native"
    )
    pairs = explainer.pairs(X, y, categorical_features=list(h.CATEGORICAL))
    assert list(pairs.columns) == ["strategy", "class", "factual", "counterfactual"]
    assert set(pairs["strategy"]) == {"prototype", "nearest-boundary", "random-pairs"}
    labels = pd.Series(np.asarray(y), index=X.index)
    assert (labels[pairs["factual"]].to_numpy() == pairs["class"].to_numpy()).all()
    assert (labels[pairs["counterfactual"]].to_numpy() != pairs["class"].to_numpy()).all()
    rules = explainer.fit(X, y, categorical_features=list(h.CATEGORICAL))
    assert list(dict.fromkeys(pairs["factual"])) == rules.mined_indices  # same factuals, same order
