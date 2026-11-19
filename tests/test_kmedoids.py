"""Tests for the vendored k-medoids implementation."""

from __future__ import annotations

import gower
import numpy as np
import pandas as pd
import pytest

from ramico._kmedoids import kmedoids


def _random_distances(rs: np.random.RandomState, n: int, dtype: type) -> np.ndarray:
    points = rs.normal(size=(n, 3))
    D = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2))
    return D.astype(dtype)


def _gower_distances(rs: np.random.RandomState, n: int) -> np.ndarray:
    frame = pd.DataFrame(
        {
            "a": rs.randint(0, 100, size=n),
            "b": np.round(rs.normal(size=n), 3),
            "c": pd.Series(rs.choice(["x", "y", "z"], size=n).tolist(), dtype=object),
        }
    )
    return gower.gower_matrix(frame)  # float32, as in the mining strategies


def _matrices() -> list[np.ndarray]:
    rs = np.random.RandomState(0)
    matrices = []
    for n in (8, 23, 57, 120):
        matrices.append(_random_distances(rs, n, np.float64))
        matrices.append(_random_distances(rs, n, np.float32))
        matrices.append(_gower_distances(rs, n))
    return matrices


@pytest.mark.parametrize("n_clusters", [1, 2, 5, 7])
def test_returns_distinct_valid_indices(n_clusters):
    for D in _matrices():
        medoids = kmedoids(D, n_clusters)
        assert medoids.shape == (n_clusters,)
        assert len(set(medoids.tolist())) == n_clusters
        assert medoids.min() >= 0 and medoids.max() < len(D)


def test_is_deterministic():
    D = _matrices()[5]
    np.testing.assert_array_equal(kmedoids(D, 6), kmedoids(D, 6))


def test_single_cluster_is_the_most_central_point():
    D = _matrices()[3]
    assert kmedoids(D, 1).tolist() == [int(np.argmin(D.sum(axis=1)))]


def test_all_points_as_medoids():
    D = _matrices()[0]
    assert sorted(kmedoids(D, len(D)).tolist()) == list(range(len(D)))


@pytest.mark.parametrize(
    ("distances", "n_clusters", "message"),
    [
        (np.zeros((3, 4)), 1, "square"),
        (np.array([[0.0, np.nan], [np.nan, 0.0]]), 1, "NaN"),
        (np.array([[0.0, -1.0], [-1.0, 0.0]]), 1, "negative"),
        (np.zeros((3, 3)), 0, "n_clusters"),
        (np.zeros((3, 3)), 4, "n_clusters"),
    ],
)
def test_invalid_input(distances, n_clusters, message):
    with pytest.raises(ValueError, match=message):
        kmedoids(distances, n_clusters)


def test_matches_scikit_learn_extra():
    """Same medoids as scikit-learn-extra 0.3.0 (reference environment only)."""
    cluster = pytest.importorskip("sklearn_extra.cluster")

    def sorted_init(self, D, n_clusters, random_state_, X=None):
        return np.argsort(np.sum(D, axis=1), kind="stable")[:n_clusters]

    for D in _matrices():
        for n_clusters in (1, 2, 5, 7):
            original = cluster.KMedoids(n_clusters=n_clusters, metric="precomputed", random_state=42)
            expected_set = sorted(original.fit(D).medoid_indices_.tolist())
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(cluster.KMedoids, "_initialize_medoids", sorted_init)
                same_init = cluster.KMedoids(n_clusters=n_clusters, metric="precomputed").fit(D)
            ours = kmedoids(D, n_clusters)
            # identical once the initial medoids are listed in the same order
            np.testing.assert_array_equal(ours, same_init.medoid_indices_)
            # and the same set of medoids as the unmodified implementation
            assert sorted(ours.tolist()) == expected_set
