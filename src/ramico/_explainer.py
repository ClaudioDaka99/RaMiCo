"""The RaMiCo explainer."""

from __future__ import annotations

import copy
import logging
import numbers
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, TypeAlias

import numpy as np
import pandas as pd

from ramico._aggregation import aggregate
from ramico._distance import gower_matrix
from ramico._plausibility import ENGINES, fit_isolation_forest
from ramico._records import pair_records
from ramico._ruleset import RuleSet, empty_rules
from ramico._validation import (
    BlackBox,
    ModelView,
    RandomState,
    check_data,
    check_random_state,
    free_column_name,
    numerical_features,
)
from ramico.strategies import MiningStrategy, Pair, get_strategy

logger = logging.getLogger(__name__)

ClassNames: TypeAlias = Mapping[Any, str] | Sequence[str] | None


class RaMiCo:
    """Region-Aware Minimal Counterfactual Rules.

    For every class ``c``, a mining strategy pairs factual samples of ``c``
    with counterfactual samples of the other classes. Each feature in which a
    pair differs is switched, one at a time, from the factual to the
    counterfactual value; switches that change the model's prediction become
    elementary records, aggregated into rules per starting class, arrival
    class and feature: mean ± standard deviation of the values for
    numerical features, most frequent value for categorical ones.

    Parameters
    ----------
    model : estimator or callable
        The black box: an object with ``predict(X)`` (scikit-learn, XGBoost,
        pipelines with a ``ColumnTransformer``...) or a function returning one
        label per row. It receives DataFrames with the columns of ``X``.
    strategy : str or MiningStrategy, default="ensemble"
        Mining strategy, by name (``"boundary"``, ``"prototype"``,
        ``"random"``, ``"ensemble"`` or a registered one) or as an instance.
    n_factuals : int, default=100
        Factual samples per class (``s`` in the paper).
    n_counterfactuals : int, default=50
        Counterfactual samples per factual (``k`` in the paper).
    distance : "gower" or callable, default="gower"
        Distance between samples; a callable takes ``(x, y=None)``
        DataFrames and returns the distance matrix.
    random_state : int, RandomState, Generator or None, default=None
        Randomness of the strategies and of the native Isolation Forest.
        ``None`` uses numpy's global random state.
    isolation_forest : {"auto", "ydf", "native"}, default="auto"
        Engine of the Isolation Forest that scores implausibility: YDF when
        installed (``"auto"``), or the dependency-free reimplementation of the
        same algorithm (``"native"``).

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression    # doctest: +SKIP
    >>> model = LogisticRegression().fit(X, y)                 # doctest: +SKIP
    >>> rules = RaMiCo(model, random_state=0).fit(X, y)        # doctest: +SKIP
    >>> rules                                                  # doctest: +SKIP
    """

    def __init__(
        self,
        model: Any,
        strategy: str | MiningStrategy = "ensemble",
        n_factuals: int = 100,
        n_counterfactuals: int = 50,
        distance: str | Callable[..., np.ndarray] = "gower",
        random_state: RandomState = None,
        isolation_forest: str = "auto",
    ) -> None:
        self.model = model
        self.strategy = strategy
        self.n_factuals = n_factuals
        self.n_counterfactuals = n_counterfactuals
        self.distance = distance
        self.random_state = random_state
        self.isolation_forest = isolation_forest

    def _check_parameters(self) -> tuple[MiningStrategy, Callable[..., np.ndarray]]:
        for name in ("n_factuals", "n_counterfactuals"):
            value = getattr(self, name)
            if not isinstance(value, numbers.Integral) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.isolation_forest not in ENGINES:
            raise ValueError(f"isolation_forest must be one of {ENGINES}, got {self.isolation_forest!r}")
        if self.distance == "gower":
            distance = gower_matrix
        elif callable(self.distance):
            distance = self.distance
        else:
            raise ValueError(f'distance must be "gower" or a callable, got {self.distance!r}')
        return get_strategy(self.strategy), distance

    def _mine(
        self,
        X: pd.DataFrame,
        labels: np.ndarray,
        classes: list[Any],
        strategy: MiningStrategy,
        distance: Callable[..., np.ndarray],
        rng: np.random.RandomState | np.random.Generator,
    ) -> Iterator[tuple[str, Any, pd.DataFrame, pd.DataFrame, list[Pair]]]:
        """Pairs of every component and class: (label, class, X_c, X_not_c, pairs)."""
        # every component runs on all the classes before the next one, as in the
        # original implementation: this fixes the order of the random draws
        for label, component in strategy.components():
            for c in classes:
                class_space = X.iloc[np.where(labels == c)[0]]
                out_space = X.iloc[np.where(labels != c)[0]]
                logger.info("Mining class %s with strategy %s", c, label)
                pairs = component.select(class_space, out_space, self.n_factuals, self.n_counterfactuals, distance, rng)
                yield label, c, class_space, out_space, pairs

    def pairs(
        self,
        X: Any,
        y: Any,
        categorical_features: Sequence[str | int] | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """The factual/counterfactual pairs chosen by the strategy, without extracting rules.

        With an integer ``random_state`` these are the pairs :meth:`fit` uses:
        the strategy runs in the same order, with the same random draws. The
        model is not called.

        Parameters
        ----------
        X, y, categorical_features, feature_names
            As in :meth:`fit`.

        Returns
        -------
        pandas.DataFrame
            One row per pair: ``strategy`` (the component that chose it),
            ``class`` (class of the factual), ``factual`` and
            ``counterfactual`` (row labels of ``X``).

        Examples
        --------
        >>> RaMiCo(model, strategy="boundary", random_state=0).pairs(X, y)  # doctest: +SKIP
        """
        strategy, distance = self._check_parameters()
        X, labels = check_data(X, y, categorical_features, feature_names)
        rng = check_random_state(self.random_state)
        classes = [c.item() if isinstance(c, np.generic) else c for c in pd.unique(labels)]
        rows = [
            (label, c, class_space.index[pair.factual], out_space.index[position])
            for label, c, class_space, out_space, pairs in self._mine(X, labels, classes, strategy, distance, rng)
            for pair in pairs
            for position in pair.counterfactuals
        ]
        return pd.DataFrame(rows, columns=["strategy", "class", "factual", "counterfactual"])

    def fit(
        self,
        X: Any,
        y: Any,
        categorical_features: Sequence[str | int] | None = None,
        feature_names: Sequence[str] | None = None,
        class_names: ClassNames = None,
    ) -> RuleSet:
        """Extract the counterfactual rules of the model on ``X``.

        Parameters
        ----------
        X : pandas.DataFrame or array of shape (n_samples, n_features)
            Samples to explore, usually the training data. Columns with a
            non-numeric dtype are categorical.
        y : array-like of shape (n_samples,)
            True labels: they define the class regions explored by the
            strategies, as in the paper.
        categorical_features : sequence of str or int, optional
            Further columns to treat as categorical (names, or positions for
            arrays), e.g. integer-coded categories.
        feature_names : sequence of str, optional
            Column names when ``X`` is an array (default ``x0``, ``x1``...).
        class_names : mapping or sequence, optional
            Display names of the classes, by label or by position of the
            label in the sorted classes.

        Returns
        -------
        RuleSet
            The rules, their records and how they were obtained.

        Raises
        ------
        ramico.exceptions.DataError
            Invalid ``X``, ``y`` or ``categorical_features``.
        ramico.exceptions.ModelError
            ``model`` has no usable ``predict``.
        """
        strategy, distance = self._check_parameters()
        given = X
        X, labels = check_data(X, y, categorical_features, feature_names)
        model = BlackBox(self.model, ModelView.between(given, X))
        rng = check_random_state(self.random_state)
        integer_seed = isinstance(self.random_state, numbers.Integral)
        seed = int(self.random_state) if isinstance(self.random_state, numbers.Integral) else _forest_seed(rng)
        classes = [c.item() if isinstance(c, np.generic) else c for c in pd.unique(labels)]
        names = _class_names(class_names, classes)

        target = free_column_name(list(X.columns), "target")
        data = X.copy()
        data[target] = labels
        scorer = fit_isolation_forest(data, engine=self.isolation_forest, random_state=seed)
        training_scores = scorer.predict(data)
        score_min, score_max = training_scores.min(), training_scores.max()

        collected = []
        mined: list[Any] = []
        for label, c, class_space, out_space, pairs in self._mine(X, labels, classes, strategy, distance, rng):
            factuals = class_space.iloc[[pair.factual for pair in pairs]]
            counterfactuals = [out_space.iloc[pair.counterfactuals] for pair in pairs]
            mined += factuals.index.to_list()
            records = pair_records(model, factuals, counterfactuals, c, scorer, score_min, score_max, target)
            if not records.empty:
                records["method"] = label
                collected.append(records)

        metadata = {
            "strategy": strategy.name,
            "n_factuals": self.n_factuals,
            "n_counterfactuals": self.n_counterfactuals,
            "random_state": self.random_state if integer_seed or self.random_state is None else "generator",
            "isolation_forest": scorer.engine,
            "classes": classes,
            "class_names": names,
            "n_samples": len(X),
            "categorical_features": [c for c in X.columns if X[c].dtype == object],
            "numerical_features": numerical_features(X),
            "mined_indices": pd.Series(mined, dtype=object).unique().tolist(),
        }
        features = list(X.columns)
        if not collected:
            warnings.warn(
                "no single-feature switch changed the model's prediction: the RuleSet is empty",
                RuntimeWarning,
                stacklevel=2,
            )
            return empty_rules(features, metadata)
        records = pd.concat(collected, ignore_index=True)
        return RuleSet(aggregate(records, metadata["numerical_features"]), records, features, metadata)


def _forest_seed(rng: np.random.RandomState | np.random.Generator) -> int:
    """A seed for the native Isolation Forest, drawn from a copy of ``rng`` so that ``rng`` keeps its state.

    The mining strategies then draw the same numbers as without the forest,
    and the forest is reproducible whenever ``rng`` is.
    """
    twin = copy.deepcopy(rng)
    if isinstance(twin, np.random.Generator):
        return int(twin.integers(0, 2**31 - 1))
    return int(twin.randint(0, 2**31 - 1))


def _class_names(class_names: ClassNames, classes: list) -> dict | None:
    if class_names is None:
        return None
    if isinstance(class_names, Mapping):
        return dict(class_names)
    names = list(class_names)
    ordered = sorted(classes, key=lambda c: (str(type(c)), c))
    if len(names) != len(ordered):
        raise ValueError(f"{len(names)} class names for {len(ordered)} classes")
    return dict(zip(ordered, names, strict=True))
