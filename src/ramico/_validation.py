"""Checks and conversions of the inputs of the public API."""

from __future__ import annotations

import numbers
from collections.abc import Callable, Sequence
from dataclasses import dataclass
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


#: Columns of the internal records: a feature with one of these names would be confused with them.
RESERVED_NAMES = frozenset(
    {
        "feature",
        "original_value",
        "switched_value",
        "original_prediction",
        "switched_prediction",
        "anomaly_score",
        "expected_label",
        "sample_index",
        "method",
    }
)


@dataclass(frozen=True)
class ModelView:
    """The data as the model was trained on them: the caller's column labels and dtypes.

    :func:`check_data` names the columns by strings and turns the categorical
    ones into object columns, as the distances need; the model gets them back
    as they were, e.g. category columns for XGBoost's native categories or
    integer labels for a function that reads ``X[0]``.
    """

    labels: tuple[Any, ...] | None = None
    dtypes: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def between(cls, original: Any, frame: pd.DataFrame) -> ModelView:
        """What :func:`check_data` changed when it turned ``original`` into ``frame``."""
        if isinstance(original, pd.DataFrame):
            source, labels = original, list(original.columns)
        else:
            values = np.asarray(original)
            if values.dtype == object:  # mixed values: the converted frame is the most typed version of them
                return cls()
            source, labels = pd.DataFrame(values, columns=frame.columns), list(frame.columns)
        names = [str(label) for label in labels]
        changed = tuple(
            (name, source[label].dtype)
            for name, label in zip(names, labels, strict=True)
            if name in frame.columns and frame[name].dtype != source[label].dtype
        )
        return cls(tuple(labels) if labels != names else None, changed)

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        """``frame`` with the caller's dtypes and column labels."""
        if self.dtypes:
            present = {name: _covering(dtype, frame[name]) for name, dtype in self.dtypes if name in frame.columns}
            if present:
                frame = frame.astype(present)
        if self.labels is not None and [str(label) for label in self.labels] == list(frame.columns):
            frame = frame.set_axis(list(self.labels), axis=1)
        return frame


def _covering(dtype: Any, values: pd.Series) -> Any:
    """``dtype``, with the categories of a categorical dtype extended to every value in ``values``.

    The switched values come from the records: a value missing from the
    categories of the frame given to the metrics would otherwise become NaN.
    """
    if not isinstance(dtype, pd.CategoricalDtype):
        return dtype
    extra = [value for value in pd.unique(values.dropna()) if value not in dtype.categories]
    return pd.CategoricalDtype([*dtype.categories, *extra], ordered=dtype.ordered) if extra else dtype


class BlackBox:
    """Uniform ``predict`` around an estimator or a prediction function."""

    def __init__(self, model: Any, view: ModelView | None = None) -> None:
        if hasattr(model, "predict") and callable(model.predict):
            self._predict: Callable[[pd.DataFrame], Any] = model.predict
        elif callable(model):
            self._predict = model
        else:
            raise ModelError(
                "model must have a predict(X) method or be a function returning one label "
                f"per row, got {type(model).__name__}"
            )
        self.view = view or ModelView()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        labels = np.asarray(self._predict(self.view.apply(X)))
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
    labels_given = list(frame.columns)
    frame.columns = [str(c) for c in frame.columns]
    if frame.columns.duplicated().any():
        raise DataError(f"duplicated column names: {sorted(set(frame.columns[frame.columns.duplicated()]))}")
    reserved = sorted(RESERVED_NAMES.intersection(frame.columns))
    if reserved:
        raise DataError(f"the feature names {reserved} are used by RaMiCo's records: rename these columns")
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
        if isinstance(X, pd.DataFrame) and not isinstance(feature, bool) and feature in labels_given:
            declared.append(str(feature))  # a column label, even an integer one
        elif isinstance(feature, numbers.Integral) and not isinstance(feature, bool):
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
