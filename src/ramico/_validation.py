"""Checks and conversions of the inputs of the public API."""

from __future__ import annotations

import numbers
from collections.abc import Callable, Sequence
from typing import Any, TypeAlias

import numpy as np
import pandas as pd

from ramico.exceptions import DataError, ModelError

RandomState: TypeAlias = int | np.random.RandomState | np.random.Generator | None


def check_random_state(random_state: RandomState) -> np.random.RandomState | np.random.Generator:
    """Turn ``random_state`` into a random generator.

    ``None`` returns numpy's global ``RandomState``, so that results depend on
    ``np.random.seed`` as in the original code; an integer seeds a new
    ``RandomState``, which draws the same sequence as ``np.random.seed``.
    """
    if random_state is None:
        return np.random.mtrand._rand
    if isinstance(random_state, (np.random.RandomState, np.random.Generator)):
        return random_state
    if isinstance(random_state, numbers.Integral) and not isinstance(random_state, bool):
        return np.random.RandomState(int(random_state))
    raise TypeError(
        "random_state must be None, an int, a numpy RandomState or a numpy Generator, "
        f"not {type(random_state).__name__}"
    )


class BlackBox:
    """Uniform ``predict`` around an estimator or a prediction function."""

    def __init__(self, model: Any) -> None:
        if hasattr(model, "predict") and callable(model.predict):
            self._predict: Callable[[pd.DataFrame], Any] = model.predict
        elif callable(model):
            self._predict = model
        else:
            raise ModelError(
                "model must have a predict(X) method or be a function returning one label "
                f"per row, got {type(model).__name__}"
            )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        labels = np.asarray(self._predict(X))
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels.ravel()
        if labels.ndim != 1 or len(labels) != len(X):
            raise ModelError(f"predict must return one label per row: got shape {labels.shape} for {len(X)} rows")
        return labels


def numerical_features(frame: pd.DataFrame) -> list[str]:
    """Columns summarised as numbers: numeric dtypes other than booleans.

    After :func:`check_data` the categorical features are object columns.
    """
    return [
        column
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column]) and not pd.api.types.is_bool_dtype(frame[column])
    ]


def free_column_name(columns: Sequence[str], preferred: str) -> str:
    """``preferred``, or a variant of it that is not among ``columns``."""
    name = preferred
    while name in columns:
        name = f"_{name}"
    return name


def check_data(
    X: Any,
    y: Any,
    categorical_features: Sequence[str | int] | None = None,
    feature_names: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Validate ``X`` and ``y`` and return a converted copy of them.

    Non-numeric columns and the declared categorical features become object
    columns, which the Gower distance treats as categorical; the caller's data
    is never modified.
    """
    if isinstance(X, pd.DataFrame):
        frame = X.copy()
        if feature_names is not None:
            raise DataError("feature_names is only used when X is an array")
    else:
        values = np.asarray(X)
        if values.ndim != 2:
            raise DataError(f"X must be a DataFrame or a 2-D array, got {values.ndim} dimensions")
        names = list(feature_names) if feature_names is not None else [f"x{j}" for j in range(values.shape[1])]
        if len(names) != values.shape[1]:
            raise DataError(f"{len(names)} feature names for {values.shape[1]} columns")
        frame = pd.DataFrame(values, columns=names)
    frame.columns = [str(c) for c in frame.columns]
    if frame.columns.duplicated().any():
        raise DataError(f"duplicated column names: {sorted(set(frame.columns[frame.columns.duplicated()]))}")
    if len(frame) < 2:
        raise DataError("X needs at least two rows")
    if frame.isna().to_numpy().any():
        missing = frame.columns[frame.isna().any()].tolist()
        raise DataError(f"X contains missing values (columns {missing}); impute or drop them first")

    labels = np.asarray(y)
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels.ravel()
    if labels.ndim != 1 or len(labels) != len(frame):
        raise DataError(f"y must have one label per row of X: got shape {labels.shape} for {len(frame)} rows")
    if pd.isna(labels).any():
        raise DataError("y contains missing labels")
    if len(pd.unique(labels)) < 2:
        raise DataError("y must contain at least two classes")

    declared: list[str] = []
    for feature in categorical_features or []:
        if isinstance(feature, numbers.Integral) and not isinstance(feature, bool):
            if not 0 <= feature < frame.shape[1]:
                raise DataError(f"categorical feature index {feature} is out of range")
            declared.append(frame.columns[feature])
        elif str(feature) in frame.columns:
            declared.append(str(feature))
        else:
            raise DataError(f"categorical feature {feature!r} is not a column of X")
    for column in frame.columns:
        dtype = frame[column].dtype
        textual = column in declared or isinstance(dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(dtype)
        if textual and not pd.api.types.is_object_dtype(dtype):
            frame[column] = frame[column].astype(object)
    return frame, labels
