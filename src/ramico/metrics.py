"""Quality metrics of counterfactual rules: fidelity, incomprehensibility, implausibility.

For every sample ``x`` of ``X``, the ``k`` records whose factual is closest to
``x`` (Gower distance), among the records that start from the model's
prediction for ``x``, form a *rule chain*: one condition per (feature, arrival
class), with the median (numerical) or most frequent value of those records.

``fidelity``
    Percentage of samples whose prediction changes when their chain is
    applied: all the conditions at once (``mode="and"``, the paper's setting)
    or at least one condition on its own reaching the rule's arrival class
    (``mode="or"``).
``incomprehensibility``
    Average number of conditions in the chains (lower is more readable).
``implausibility``
    Average over the rules of the normalised anomaly score of their records
    (lower is more realistic).

The metrics accept a :class:`ramico.RuleSet` or, for rules produced by other
explainers, a DataFrame of records with the columns of
:attr:`ramico.RuleSet.records`: ``feature``, ``to_value``, ``from_class``,
``to_class``, ``implausibility`` and the factual's values in
``factual.<feature>`` columns.

Examples
--------
>>> from ramico import metrics                        # doctest: +SKIP
>>> scores = metrics.evaluate(model, rules, X_train)  # doctest: +SKIP
>>> scores                                            # doctest: +SKIP
Scores(fidelity=93.7, incomprehensibility=3.2, implausibility=0.41)
>>> scores.fidelity                                   # doctest: +SKIP
93.7
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
import pandas as pd

from ramico._distance import gower_matrix, gower_rows
from ramico._ruleset import FACTUAL_PREFIX, RuleSet
from ramico._scores import Scores
from ramico._validation import BlackBox, ModelView, numerical_features
from ramico.exceptions import DataError

__all__ = ["Scores", "evaluate", "fidelity", "implausibility", "incomprehensibility"]

#: Default number of records per chain and combination mode used in the paper.
K = 7
MODES = ("and", "or")
_KEYS = ["feature", "original_prediction", "switched_prediction"]

Rules: TypeAlias = RuleSet | pd.DataFrame


@dataclass
class _Chains:
    """The rule chains of the samples of ``X``, one row of ``conditions`` per condition."""

    X: pd.DataFrame
    predictions: np.ndarray  # the model's prediction for each sample
    conditions: pd.DataFrame  # position (of the sample), feature, switched_value, switched_prediction

    @property
    def mean_length(self) -> float:
        """Average number of conditions per sample."""
        return len(self.conditions) / len(self.X) if len(self.X) else float("nan")


def _records(rules: Rules, X: pd.DataFrame) -> pd.DataFrame:
    """Records in the internal layout, with the factual values named as the features."""
    features = list(X.columns)
    if isinstance(rules, RuleSet):
        public = rules.records
    elif isinstance(rules, pd.DataFrame):
        public = rules
    else:
        raise TypeError(f"rules must be a RuleSet or a DataFrame of records, not {type(rules).__name__}")
    required = ["feature", "to_value", "from_class", "to_class"] + [FACTUAL_PREFIX + f for f in features]
    missing = [c for c in required if c not in public.columns]
    if missing:
        raise DataError(f"the records lack the columns {missing}")
    records = pd.DataFrame(
        {
            "feature": public["feature"],
            "switched_value": public["to_value"],
            "original_prediction": public["from_class"],
            "switched_prediction": public["to_class"],
        }
    )
    for feature in features:
        records[feature] = public[FACTUAL_PREFIX + feature]
    return records.reset_index(drop=True)


def _prepare(rules: Rules, X: Any) -> pd.DataFrame:
    """``X`` converted as :meth:`RaMiCo.fit` converted the data the rules come from."""
    if not isinstance(X, pd.DataFrame):
        features = rules._features if isinstance(rules, RuleSet) else None
        values = np.asarray(X)
        if features is None or values.ndim != 2 or values.shape[1] != len(features):
            raise DataError("X must be a DataFrame, or an array with the columns the rules were extracted from")
        X = pd.DataFrame(values, columns=features)
        for column in rules.metadata.get("numerical_features", []):  # an array of mixed data holds objects
            if column in X.columns and not pd.api.types.is_numeric_dtype(X[column]):
                X[column] = pd.to_numeric(X[column])
    X = X.copy()
    X.columns = [str(c) for c in X.columns]
    categorical = rules.metadata.get("categorical_features", []) if isinstance(rules, RuleSet) else []
    for column in X.columns:
        dtype = X[column].dtype
        textual = column in categorical or isinstance(dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(dtype)
        if textual and not pd.api.types.is_object_dtype(dtype):
            X[column] = X[column].astype(object)
    return X


def _first_k(distances: np.ndarray, k: int) -> np.ndarray:
    """``np.argsort(distances, kind="stable")[:k]``, without sorting all of them."""
    if len(distances) <= k:
        return np.argsort(distances, kind="stable")
    threshold = np.partition(distances, k - 1)[k - 1]
    if np.isnan(threshold):
        return np.argsort(distances, kind="stable")[:k]
    within = np.flatnonzero(distances <= threshold)  # ascending positions: ties stay in order
    return within[np.argsort(distances[within], kind="stable")][:k]


def _nearest_records(
    records: pd.DataFrame, X: pd.DataFrame, predictions: np.ndarray, k: int, max_cells: int = 4_000_000
) -> list[pd.Index | None]:
    """Labels of the ``k`` records nearest to each sample among those that start from its prediction.

    Each sample is compared with the candidate records on its own (Gower
    ranges over the sample and the candidates), ties broken by record order,
    as in the original implementation; the distances of many samples are
    computed at once when :func:`ramico._distance.gower_rows` can reproduce
    them exactly.
    """
    features = list(X.columns)
    nearest: list[pd.Index | None] = [None] * len(X)
    for prediction in pd.unique(predictions):
        positions = np.flatnonzero(predictions == prediction)
        candidates = records[records["original_prediction"] == prediction]
        if candidates.empty:
            continue
        space = candidates[features]
        block = max(1, max_cells // len(candidates))
        for start in range(0, len(positions), block):
            rows = positions[start : start + block]
            distances = gower_rows(X.iloc[rows], space)
            for i, position in enumerate(rows):
                if distances is not None:
                    row = distances[i]
                else:  # the one-row frame of the original, whose dtypes are inferred from the values
                    row = gower_matrix(pd.DataFrame([X.iloc[position]]), space)[0]
                nearest[position] = candidates.index[_first_k(row, k)]
    return nearest


def _combined_values(chosen: pd.DataFrame, numerical: Collection[str]) -> pd.DataFrame:
    """The value of each condition: median (numerical) or most frequent value of its records.

    ``chosen`` has one row per record of a chain (``position`` of the sample,
    the keys, ``switched_value``). Missing values are left out; among equally
    frequent values the smallest wins, as with ``Series.mode()[0]``.
    """
    group_keys = ["position", *_KEYS]
    valid = chosen[chosen["switched_value"].notna()]
    by_median = valid["feature"].isin(set(numerical)).to_numpy()
    parts = []

    medians = valid[by_median]
    if len(medians):
        values = pd.to_numeric(medians["switched_value"]).to_numpy(dtype=np.float64)
        group = medians.groupby(group_keys, sort=False).ngroup().to_numpy()  # numbered by first appearance
        order = np.lexsort((values, group))
        sizes = np.bincount(group)
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        ranked = values[order]
        low, high = ranked[starts + (sizes - 1) // 2], ranked[starts + sizes // 2]
        median = np.where(sizes % 2 == 1, low, (low + high) / 2)  # numpy's median of each group
        keys = medians.loc[~medians.duplicated(group_keys), group_keys].reset_index(drop=True)
        parts.append(keys.assign(value=pd.Series(list(median), dtype=object)))

    for _, part in valid[~by_median].groupby("feature", sort=False):  # one feature: values of one kind
        counts = part.groupby([*group_keys, "switched_value"], sort=True).size().rename("count").reset_index()
        best = counts.loc[counts.groupby(group_keys, sort=False)["count"].idxmax()]  # first: the smallest
        parts.append(best[group_keys].assign(value=best["switched_value"].astype(object)).reset_index(drop=True))

    if not parts:
        return pd.DataFrame(columns=[*group_keys, "value"])
    return pd.concat(parts, ignore_index=True)


def _chains(model: BlackBox, records: pd.DataFrame, X: pd.DataFrame, k: int, numerical: Collection[str]) -> _Chains:
    """Rule chain of every sample, as in the original implementation.

    For each sample, its ``k`` nearest records give one condition per (feature,
    arrival class), in the order the records first give it, with the combined
    value of those records (:func:`_combined_values`).
    """
    predictions = np.asarray(model.predict(X))
    nearest = _nearest_records(records, X, predictions, k)
    present = {position: labels for position, labels in enumerate(nearest) if labels is not None}
    columns = ["position", "feature", "switched_value", "switched_prediction"]
    if not present:
        return _Chains(X, predictions, pd.DataFrame(columns=columns))
    labels = np.concatenate([np.asarray(chosen_labels) for chosen_labels in present.values()])
    chosen = records.loc[labels, [*_KEYS, "switched_value"]].reset_index(drop=True)
    sizes = np.array([len(chosen_labels) for chosen_labels in present.values()])
    chosen.insert(0, "position", np.repeat(np.fromiter(present, dtype=np.int64), sizes))

    conditions = chosen[~chosen.duplicated(["position", *_KEYS])].reset_index(drop=True)
    conditions["switched_value"] = conditions["switched_value"].astype(object)
    values = _combined_values(chosen, numerical).assign(found=True)
    merged = conditions.merge(values, on=["position", *_KEYS], how="left", sort=False)
    found = merged["found"].eq(True).to_numpy()
    if found.any():
        conditions.loc[found, "switched_value"] = merged.loc[found, "value"].to_numpy(dtype=object)
    return _Chains(X, predictions, conditions[columns])


def _switched_frame(values: np.ndarray, columns: pd.Index) -> pd.DataFrame:
    """The switched samples as one frame, with the dtypes their values give.

    The same dtypes as concatenating one-row frames built from the same
    values: integers and floats give float64, text gives text.
    """
    return pd.DataFrame(values, columns=columns).infer_objects()


def _successes(model: BlackBox, chains: _Chains, mode: str) -> np.ndarray:
    """Whether each chain changes its sample's prediction (one batched predict).

    ``and``: all the conditions of a chain at once (a later condition on the
    same feature replaces an earlier one); success when the prediction moves
    away from the original. ``or``: each condition on its own; success when
    one of them reaches its arrival class.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    X, conditions = chains.X, chains.conditions
    base = np.asarray(X.values, dtype=object)  # the values of X.iterrows()
    column = {name: i for i, name in enumerate(X.columns)}
    success = np.zeros(len(X), dtype=bool)
    if mode == "and":
        switched = base.copy()
        last = conditions[~conditions.duplicated(["position", "feature"], keep="last")]
        if len(last):
            rows = last["position"].to_numpy(dtype=np.int64)
            cols = last["feature"].map(column).to_numpy(dtype=np.int64)
            switched[rows, cols] = last["switched_value"].to_numpy(dtype=object)
        predicted = np.asarray(model.predict(_switched_frame(switched, X.columns)))
        return np.asarray(predicted != chains.predictions, dtype=bool)  # moved away from the original
    if not len(conditions):
        return success
    rows = conditions["position"].to_numpy(dtype=np.int64)
    switched = base[rows].copy()
    switched[np.arange(len(rows)), conditions["feature"].map(column).to_numpy(dtype=np.int64)] = conditions[
        "switched_value"
    ].to_numpy(dtype=object)
    predicted = np.asarray(model.predict(_switched_frame(switched, X.columns)))
    reached = np.asarray(predicted == conditions["switched_prediction"].to_numpy(), dtype=bool)
    success[rows[reached]] = True  # one condition reached its arrival class
    return success


def _rule_chains(model: Any, rules: Rules, X: Any, k: int) -> tuple[BlackBox, pd.DataFrame, _Chains]:
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    prepared = _prepare(rules, X)
    view = ModelView.between(X, prepared)
    labels = rules.metadata.get("column_labels") if isinstance(rules, RuleSet) else None
    if labels is not None and not isinstance(X, pd.DataFrame):  # an array: the labels of the data of fit
        view = ModelView(tuple(labels), view.dtypes)
    black_box = BlackBox(model, view)
    X = prepared
    return black_box, X, _chains(black_box, _records(rules, X), X, int(k), set(numerical_features(X)))


def evaluate(model: Any, rules: Rules, X: Any, k: int = K, mode: str = "and") -> Scores:
    """Fidelity, incomprehensibility and implausibility of ``rules`` on ``X``.

    Parameters
    ----------
    model : estimator or callable
        The model the rules explain.
    rules : RuleSet or DataFrame
        Rules, or records with the columns of :attr:`RuleSet.records`.
    X : pandas.DataFrame
        Samples on which the rules are evaluated (the paper uses the
        training data the rules were extracted from).
    k : int, default=7
        Records per rule chain.
    mode : {"and", "or"}, default="and"
        How the conditions of a chain are applied for fidelity.

    Returns
    -------
    Scores
        ``fidelity`` (%), ``incomprehensibility`` (conditions per chain) and
        ``implausibility`` (normalised anomaly score), each read as an
        attribute (``scores.fidelity``) or as a key (``scores["fidelity"]``).
    """
    black_box, X, chains = _rule_chains(model, rules, X, k)
    success = _successes(black_box, chains, mode)
    return Scores(
        fidelity=float(100.0 * success.sum() / len(X)),
        incomprehensibility=float(chains.mean_length),
        implausibility=implausibility(rules),
    )


def fidelity(model: Any, rules: Rules, X: Any, k: int = K, mode: str = "and") -> float:
    """Percentage of samples of ``X`` whose prediction the rules flip (see :func:`evaluate`)."""
    black_box, X, chains = _rule_chains(model, rules, X, k)
    return float(100.0 * _successes(black_box, chains, mode).sum() / len(X))


def incomprehensibility(model: Any, rules: Rules, X: Any, k: int = K) -> float:
    """Average number of conditions in the rule chains of ``X`` (see :func:`evaluate`)."""
    _, _, chains = _rule_chains(model, rules, X, k)
    return float(chains.mean_length)


def implausibility(rules: Rules) -> float:
    """Average implausibility of the rules (mean of the per-rule means).

    For a DataFrame of records, rules are formed by feature, starting class,
    arrival class and, when present, strategy.
    """
    if isinstance(rules, RuleSet):
        values = rules.to_frame()["implausibility"]
    elif isinstance(rules, pd.DataFrame):
        if "implausibility" not in rules.columns:
            raise DataError("the records lack the 'implausibility' column")
        keys = [c for c in ("feature", "from_class", "to_class", "strategy") if c in rules.columns]
        values = rules.groupby(keys)["implausibility"].mean()
    else:
        raise TypeError(f"rules must be a RuleSet or a DataFrame of records, not {type(rules).__name__}")
    return float(values.mean()) if len(values) else float("nan")
