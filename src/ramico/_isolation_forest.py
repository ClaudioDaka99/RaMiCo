"""Isolation Forest following the algorithm of YDF, in numpy.

The forest reproduces the algorithm and the configuration that the paper used
through YDF (``ydf.IsolationForestLearner(num_trees=1000, max_depth=-1)``):

* every tree grows on ``subsample`` examples (256 by default, at most the size
  of the dataset) drawn without replacement;
* a node becomes a leaf when it holds fewer than ``min_examples`` examples,
  when it reaches ``max_depth`` or when all its features are constant;
* otherwise one non-constant feature is drawn uniformly. Numerical features
  are split at a threshold drawn uniformly between the minimum (excluded)
  and the maximum of the node, examples at or above the threshold going to
  the positive branch. Categorical features send one observed value to each
  branch and every other value to either branch with probability 1/2;
* the anomaly score of an example is ``2 ** (-E[h] / c(subsample))``, where
  ``h`` is the depth of the leaf it reaches plus ``c(examples in the leaf)``
  and ``c(n) = 2 (ln(n - 1) + 0.5772156649) - 2 (n - 1) / n`` (``c(2) = 1``,
  ``c(1) = 0``);
* columns whose dtype is not numeric are categorical; values seen fewer than
  ``min_vocab_frequency`` times share an out-of-dictionary code, as do values
  never seen in training. Missing numbers take the training mean, missing
  categories the most frequent one.

Unlike YDF, training is deterministic for a given ``random_state``. The trees
of all the forest grow together, one depth level at a time, so that every
step is a vectorised numpy operation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EULER_GAMMA = 0.5772156649
_LEAF = -1


def average_path_length(n: int | np.ndarray) -> np.ndarray:
    """Average path length ``c(n)`` of an unsuccessful search in a binary tree."""
    n = np.asarray(n, dtype=np.float64)
    out = np.zeros_like(n)
    large = n > 2
    out[large] = 2.0 * (np.log(n[large] - 1.0) + _EULER_GAMMA) - 2.0 * (n[large] - 1.0) / n[large]
    out[n == 2] = 1.0
    return out


class IsolationForest:
    """Isolation Forest for mixed numerical and categorical data.

    Parameters
    ----------
    n_trees : int, default=1000
        Number of trees.
    subsample : int, default=256
        Examples drawn, without replacement, to grow each tree.
    max_depth : int, optional
        Maximum depth of the trees; unlimited by default.
    min_examples : int, default=5
        Nodes with fewer examples become leaves.
    min_vocab_frequency : int, default=5
        Categorical values seen fewer times share one out-of-dictionary code.
    random_state : int, optional
        Seed of the random generator; None draws fresh entropy.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> X = pd.DataFrame({"x": np.r_[np.random.RandomState(0).normal(size=300), 8.0]})
    >>> forest = IsolationForest(n_trees=200, random_state=0).fit(X)
    >>> scores = forest.predict(X)
    >>> bool(scores[-1] == scores.max())  # the far point is the most anomalous
    True
    """

    def __init__(
        self,
        n_trees: int = 1000,
        subsample: int = 256,
        max_depth: int | None = None,
        min_examples: int = 5,
        min_vocab_frequency: int = 5,
        random_state: int | None = None,
    ) -> None:
        self.n_trees = n_trees
        self.subsample = subsample
        self.max_depth = max_depth
        self.min_examples = min_examples
        self.min_vocab_frequency = min_vocab_frequency
        self.random_state = random_state

    # ------------------------------------------------------------------ data

    def _fit_encoding(self, X: pd.DataFrame) -> None:
        self.columns_ = list(X.columns)
        self.categorical_ = np.zeros(len(self.columns_), dtype=bool)
        self.fill_: list = []
        self.vocabulary_: list[dict[str, int] | None] = []
        for j, column in enumerate(self.columns_):
            values = X[column]
            if pd.api.types.is_numeric_dtype(values.dtype) or pd.api.types.is_bool_dtype(values.dtype):
                numbers = values.to_numpy(dtype=np.float64, na_value=np.nan)
                mean = np.nanmean(numbers) if np.isfinite(numbers).any() else 0.0
                self.fill_.append(float(mean))
                self.vocabulary_.append(None)
            else:
                self.categorical_[j] = True
                counts = values.dropna().map(str).value_counts()
                kept = sorted(str(v) for v, c in counts.items() if c >= self.min_vocab_frequency)
                self.vocabulary_.append({value: code for code, value in enumerate(kept, start=1)})
                self.fill_.append(str(counts.index[0]) if len(counts) else None)
        sizes = [len(v) + 1 for v in self.vocabulary_ if v is not None]
        self.n_codes_ = max(sizes) if sizes else 1

    def _encode(self, X: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.columns_ if c not in X.columns]
        if missing:
            raise ValueError(f"columns seen in fit are missing: {missing}")
        Z = np.empty((len(X), len(self.columns_)), dtype=np.float64)
        for j, column in enumerate(self.columns_):
            values = X[column]
            if not self.categorical_[j]:
                numbers = pd.to_numeric(values).to_numpy(dtype=np.float64, na_value=np.nan)
                Z[:, j] = np.where(np.isnan(numbers), self.fill_[j], numbers)
            else:
                vocabulary, fill = self.vocabulary_[j], self.fill_[j]
                assert vocabulary is not None  # fit builds one for every categorical column
                keys = [fill if pd.isna(v) else str(v) for v in values]
                Z[:, j] = [vocabulary.get(k, 0) for k in keys]
        return Z

    # -------------------------------------------------------------- training

    def fit(self, X: pd.DataFrame) -> IsolationForest:
        """Grow the forest on ``X``.

        Parameters
        ----------
        X : pandas.DataFrame
            Training data; non-numeric columns are treated as categorical.

        Returns
        -------
        IsolationForest
            The fitted forest.
        """
        X = pd.DataFrame(X)
        if len(X) == 0:
            raise ValueError("cannot fit an Isolation Forest on an empty dataset")
        rng = np.random.RandomState(self.random_state)
        self._fit_encoding(X)
        Z = self._encode(X)
        n = len(Z)
        self.subsample_size_ = min(self.subsample, n)
        samples = np.concatenate([rng.choice(n, self.subsample_size_, replace=False) for _ in range(self.n_trees)])
        self._grow(Z, samples, rng)
        self.denominator_ = float(average_path_length(self.subsample_size_))
        return self

    def _grow(self, Z: np.ndarray, samples: np.ndarray, rng: np.random.RandomState) -> None:
        feature: list[np.ndarray] = []
        threshold: list[np.ndarray] = []
        mask_row: list[np.ndarray] = []
        positive: list[np.ndarray] = []
        negative: list[np.ndarray] = []
        leaf_h: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        n_masks = 0

        # one element per (tree, sampled example); nodes of the current level
        node_of = np.repeat(np.arange(self.n_trees), self.subsample_size_)
        first_node, n_level, depth = 0, self.n_trees, 0
        while n_level:
            order = np.argsort(node_of, kind="stable")
            node_of, samples = node_of[order], samples[order]
            local = node_of - first_node
            counts = np.bincount(local, minlength=n_level)
            starts = np.concatenate(([0], np.cumsum(counts)[:-1]))

            level_feature = np.full(n_level, _LEAF)
            level_threshold = np.full(n_level, np.nan)
            level_mask = np.full(n_level, -1)
            level_leaf_h = depth + average_path_length(counts)

            splittable = (counts >= max(self.min_examples, 2)) & (counts > 1)
            if self.max_depth is not None and depth >= self.max_depth:
                splittable[:] = False
            candidates = np.flatnonzero(splittable)
            if len(candidates):
                # every node of a level holds at least one example, so the
                # starts are increasing and reduceat gives one row per node
                values = Z[samples]
                lows = np.minimum.reduceat(values, starts, axis=0)[candidates]
                highs = np.maximum.reduceat(values, starts, axis=0)[candidates]
                eligible = lows < highs
                has_split = eligible.any(axis=1)
                candidates, lows, highs, eligible = (
                    candidates[has_split],
                    lows[has_split],
                    highs[has_split],
                    eligible[has_split],
                )
                # draw one non-constant feature per node, uniformly
                n_eligible = eligible.sum(axis=1)
                pick = np.floor(rng.random_sample(len(candidates)) * n_eligible).astype(int)
                chosen = np.argmax(np.cumsum(eligible, axis=1) > pick[:, None], axis=1)
                level_feature[candidates] = chosen
                rows = np.arange(len(candidates))
                low, high = lows[rows, chosen], highs[rows, chosen]
                numerical = ~self.categorical_[chosen]
                start = np.nextafter(low, np.inf)
                draw = rng.random_sample(len(candidates))
                level_threshold[candidates[numerical]] = (start + draw * (high - start))[numerical]
                categorical = np.flatnonzero(~numerical)
                if len(categorical):
                    new_masks = self._categorical_masks(
                        Z, samples, local, candidates[categorical], chosen[categorical], rng
                    )
                    level_mask[candidates[categorical]] = n_masks + np.arange(len(categorical))
                    masks.append(new_masks)
                    n_masks += len(categorical)

            is_split = level_feature != _LEAF
            n_children = 2 * int(is_split.sum())
            next_first = first_node + n_level
            level_positive = np.full(n_level, -1)
            level_negative = np.full(n_level, -1)
            level_positive[is_split] = next_first + 2 * np.arange(n_children // 2)
            level_negative[is_split] = level_positive[is_split] + 1
            level_leaf_h[is_split] = np.nan

            feature.append(level_feature)
            threshold.append(level_threshold)
            mask_row.append(level_mask)
            positive.append(level_positive)
            negative.append(level_negative)
            leaf_h.append(level_leaf_h)

            # route the examples of split nodes to the children
            keep = is_split[local]
            node_of, samples, local = node_of[keep], samples[keep], local[keep]
            if len(samples):
                go = self._goes_positive(
                    Z[samples],
                    level_feature[local],
                    level_threshold[local],
                    level_mask[local],
                    np.concatenate(masks) if masks else np.zeros((0, self.n_codes_), dtype=bool),
                )
                node_of = np.where(go, level_positive[local], level_negative[local])
            first_node, n_level, depth = next_first, n_children, depth + 1

        self.feature_ = np.concatenate(feature)
        self.threshold_ = np.concatenate(threshold)
        self.mask_row_ = np.concatenate(mask_row)
        self.positive_ = np.concatenate(positive)
        self.negative_ = np.concatenate(negative)
        self.leaf_h_ = np.concatenate(leaf_h)
        self.masks_ = np.concatenate(masks) if masks else np.zeros((0, self.n_codes_), dtype=bool)

    def _categorical_masks(
        self,
        Z: np.ndarray,
        samples: np.ndarray,
        local: np.ndarray,
        nodes: np.ndarray,
        features: np.ndarray,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Random partition of the categories for each categorical split."""
        position = np.full(local.max() + 1, -1)
        position[nodes] = np.arange(len(nodes))
        member = position[local] >= 0
        owner = position[local[member]]
        codes = Z[samples[member], features[owner]].astype(int)
        pairs = np.unique(owner * self.n_codes_ + codes)
        pair_node, pair_code = pairs // self.n_codes_, pairs % self.n_codes_
        n_observed = np.bincount(pair_node, minlength=len(nodes))
        first = np.concatenate(([0], np.cumsum(n_observed)[:-1]))
        forced_in = np.floor(rng.random_sample(len(nodes)) * n_observed).astype(int)
        forced_out = np.floor(rng.random_sample(len(nodes)) * (n_observed - 1)).astype(int)
        forced_out += forced_out >= forced_in
        masks = rng.random_sample((len(nodes), self.n_codes_)) < 0.5
        rows = np.arange(len(nodes))
        masks[rows, pair_code[first + forced_in]] = True
        masks[rows, pair_code[first + forced_out]] = False
        return masks

    def _goes_positive(
        self,
        values: np.ndarray,
        features: np.ndarray,
        thresholds: np.ndarray,
        mask_rows: np.ndarray,
        masks: np.ndarray,
    ) -> np.ndarray:
        rows = np.arange(len(values))
        observed = values[rows, features]
        go = observed >= thresholds  # NaN thresholds (categorical splits) give False
        categorical = mask_rows >= 0
        if categorical.any():
            go[categorical] = masks[mask_rows[categorical], observed[categorical].astype(int)]
        return go

    # --------------------------------------------------------------- scoring

    def path_lengths(self, X: pd.DataFrame, chunk_size: int = 4096) -> np.ndarray:
        """Average path length ``E[h]`` of every row of ``X`` over the trees."""
        Z = self._encode(pd.DataFrame(X))
        out = np.empty(len(Z), dtype=np.float64)
        rows_per_chunk = max(1, chunk_size * 256 // self.n_trees)
        for begin in range(0, len(Z), rows_per_chunk):
            block = Z[begin : begin + rows_per_chunk]
            row = np.repeat(np.arange(len(block)), self.n_trees)
            node = np.tile(np.arange(self.n_trees), len(block))
            active = np.flatnonzero(self.feature_[node] != _LEAF)
            while len(active):
                at = node[active]
                go = self._goes_positive(
                    block[row[active]],
                    self.feature_[at],
                    self.threshold_[at],
                    self.mask_row_[at],
                    self.masks_,
                )
                node[active] = np.where(go, self.positive_[at], self.negative_[at])
                active = active[self.feature_[node[active]] != _LEAF]
            out[begin : begin + len(block)] = self.leaf_h_[node].reshape(len(block), -1).mean(axis=1)
        return out

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Anomaly score of every row of ``X``, between 0 and 1 (higher is more anomalous)."""
        if self.denominator_ == 0:
            return np.zeros(len(X))
        return 2.0 ** (-self.path_lengths(X) / self.denominator_)
