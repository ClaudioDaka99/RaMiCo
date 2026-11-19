"""k-medoids clustering on a precomputed distance matrix.

Adapted from ``sklearn_extra.cluster.KMedoids`` (scikit-learn-extra 0.3.0,
authors: Timo Erkkilä, Antti Lehmussola, Kornel Kiełczewski, Zane Dufour),
restricted to the configuration RaMiCo uses: ``metric="precomputed"``,
``method="alternate"``, ``init="heuristic"``, ``max_iter=300``.

One deliberate difference: the heuristic initialisation lists the initial
medoids sorted by total distance (ties by index). scikit-learn-extra takes
them from ``np.argpartition``, whose order depends on the numpy version; the
set of medoids is the same.

The original code is distributed under the following licence:

    Copyright (c) 2016, scikit-learn-contrib contributors
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the name of project-template nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def kmedoids(distances: np.ndarray, n_clusters: int, max_iter: int = 300) -> np.ndarray:
    """Return the indices of ``n_clusters`` medoids of a distance matrix.

    Parameters
    ----------
    distances : ndarray of shape (n_samples, n_samples)
        Symmetric matrix of non-negative distances. float32 and float64 are
        used as given; other dtypes are converted to float64.
    n_clusters : int
        Number of medoids, between 1 and ``n_samples``.
    max_iter : int, default=300
        Maximum number of assignment/update rounds.

    Returns
    -------
    ndarray of shape (n_clusters,)
        Row indices of the medoids. The order follows the initial medoids,
        sorted by total distance to the other points.

    Raises
    ------
    ValueError
        If the matrix is not square, contains NaN, infinite or negative
        values, or if ``n_clusters`` is out of range.
    """
    D = np.asarray(distances)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"distances must be a square matrix, got shape {D.shape}")
    if D.dtype not in (np.float32, np.float64):
        D = D.astype(np.float64)
    if not np.all(np.isfinite(D)):
        raise ValueError("distances contains NaN or infinite values")
    if np.any(D < 0):
        raise ValueError("distances contains negative values")
    n_samples = D.shape[0]
    if not 1 <= n_clusters <= n_samples:
        raise ValueError(f"n_clusters must be between 1 and {n_samples}, got {n_clusters}")

    # heuristic initialisation: the points with the smallest total distance
    medoids = np.argsort(np.sum(D, axis=1), kind="stable")[:n_clusters]

    for iteration in range(max_iter):
        previous = np.copy(medoids)
        labels = np.argmin(D[medoids, :], axis=0)
        _update_medoids_in_place(D, labels, medoids)
        if np.all(previous == medoids):
            break
        if iteration == max_iter - 1:
            logger.warning("k-medoids reached max_iter=%d before converging", max_iter)
    return medoids


def _update_medoids_in_place(D: np.ndarray, labels: np.ndarray, medoids: np.ndarray) -> None:
    """Move each medoid to the point with the lowest total in-cluster distance."""
    for k in range(len(medoids)):
        members = np.where(labels == k)[0]
        if len(members) == 0:
            logger.debug("k-medoids cluster %d is empty; its medoid is kept", k)
            continue
        # same indexing as scikit-learn-extra (transposed block, D is symmetric)
        costs = np.sum(D[members, members[:, np.newaxis]], axis=1)
        best = np.argmin(costs)
        current = costs[np.argmax(members == medoids[k])]
        if costs[best] < current:
            medoids[k] = members[best]
