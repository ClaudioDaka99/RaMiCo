"""The metrics one sample at a time, as in ramico 0.1.0 before the batched implementation.

Kept as the reference of ``tests/test_metrics.py``: the batched code must give
the same fidelity and incomprehensibility on every input.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

import numpy as np
import pandas as pd

from ramico import metrics
from ramico._distance import gower_matrix
from ramico._validation import BlackBox, numerical_features

_KEYS = ["feature", "original_prediction", "switched_prediction"]


def _chains(
    model: BlackBox, records: pd.DataFrame, X: pd.DataFrame, k: int, numerical: Collection[str]
) -> list[tuple[pd.DataFrame, Any, pd.DataFrame]]:
    features = list(X.columns)
    predictions = model.predict(X)
    chains = []
    for position, (_, row) in enumerate(X.iterrows()):
        sample = pd.DataFrame([row])
        candidates = records[records["original_prediction"] == predictions[position]]
        if candidates.empty:
            empty = records.iloc[0:0][["feature", "switched_value", "switched_prediction"]]
            chains.append((sample, predictions[position], empty))
            continue
        distances = gower_matrix(sample, candidates[features])[0]
        nearest = np.argsort(distances, kind="stable")[: min(len(distances), k)]
        chosen = records.loc[candidates.iloc[nearest].index].copy()
        values = {}
        for key, group in chosen.groupby(_KEYS)["switched_value"]:
            group = group.dropna()
            if group.empty:
                continue
            median = key[0] in numerical
            group = pd.to_numeric(group) if median else group
            values[key] = group.median() if median else group.mode()[0]
        chosen = chosen.drop_duplicates(subset=_KEYS)
        chosen["switched_value"] = chosen["switched_value"].astype(object)
        for index, condition in chosen.iterrows():
            key = (condition["feature"], condition["original_prediction"], condition["switched_prediction"])
            if key in values:
                chosen.at[index, "switched_value"] = values[key]
        chains.append((sample, predictions[position], chosen[["feature", "switched_value", "switched_prediction"]]))
    return chains


def _successes(model: BlackBox, chains: list[tuple[pd.DataFrame, Any, pd.DataFrame]], mode: str) -> np.ndarray:
    switched, owner, expected = [], [], []
    for i, (sample, prediction, conditions) in enumerate(chains):
        if mode == "and":
            sample = sample.copy()
            for _, condition in conditions.iterrows():
                sample[condition["feature"]] = condition["switched_value"]
            switched.append(sample)
            owner.append(i)
            expected.append(prediction)
        else:
            for _, condition in conditions.iterrows():
                one = sample.copy()
                one[condition["feature"]] = condition["switched_value"]
                switched.append(one)
                owner.append(i)
                expected.append(condition["switched_prediction"])
    success = np.zeros(len(chains), dtype=bool)
    if not switched:
        return success
    predicted = model.predict(pd.concat(switched, ignore_index=True))
    for i, prediction, target in zip(owner, predicted, expected, strict=True):
        if mode == "and":
            success[i] = prediction != target
        else:
            success[i] |= prediction == target
    return success


def scores(model: Any, rules: Any, X: Any, k: int = 7, mode: str = "and") -> tuple[float, float]:
    """Fidelity and incomprehensibility, one sample at a time."""
    black_box = BlackBox(model)
    X = metrics._prepare(rules, X)
    chains = _chains(black_box, metrics._records(rules, X), X, k, set(numerical_features(X)))
    success = _successes(black_box, chains, mode)
    return 100.0 * success.sum() / len(X), float(np.mean([len(c[2]) for c in chains]))
