"""Elementary counterfactual records: single-feature switches that flip the model.

The original implementation rebuilds a one-row frame for every factual,
counterfactual and switch, so that pandas infers the dtypes value by value;
:func:`switch_records` keeps those steps and the golden tests freeze them.
When every column holds values of a single kind, all those one-row frames
share the same dtypes, and :func:`switch_records_vectorized` builds the same
switches, in the same order and with the same dtypes, in one pass.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Columns of a record, followed by the values of its factual, its row label
#: (``sample_index``) and the label of the strategy that mined it (``method``).
RECORD_COLUMNS = [
    "feature",
    "original_value",
    "switched_value",
    "original_prediction",
    "switched_prediction",
    "anomaly_score",
    "expected_label",
]


def switch_records(
    model: Any,
    factual: pd.DataFrame,
    counterfactuals: pd.DataFrame,
    mined_class: Any,
    scorer: Any,
    score_min: float,
    score_max: float,
    target_column: str,
) -> pd.DataFrame | None:
    """Records obtained by switching, one at a time, the features of ``factual``.

    For every counterfactual and every feature where it differs from the
    factual, the factual takes the counterfactual's value; the switch is kept
    when the model's prediction changes. Its implausibility is the anomaly
    score of the switched sample, labelled with the new prediction, rescaled
    with the scores of the training data.
    """
    original_prediction = model.predict(factual).item()

    switched_rows = []
    switches = []
    for position in range(counterfactuals.shape[0]):
        counterfactual = pd.DataFrame([counterfactuals.iloc[position]], index=[0])
        for feature in factual.columns:
            if factual[feature].values != counterfactual[feature].values:
                switched = factual.copy()
                switched[feature] = counterfactual[feature]
                switched_rows.append(switched)
                switches.append(
                    (feature, factual[feature].values[0], counterfactual[feature].values[0], original_prediction)
                )

    if not switched_rows:
        return None
    switched = pd.concat(switched_rows, ignore_index=True)
    predictions = model.predict(switched)
    switched[target_column] = predictions
    return _records(
        switched, switches, predictions, original_prediction, factual, mined_class, scorer, score_min, score_max
    )


def _records(
    switched: pd.DataFrame,
    switches: list[tuple[Any, ...]],
    predictions: np.ndarray,
    original_prediction: Any,
    factual: pd.DataFrame,
    mined_class: Any,
    scorer: Any,
    score_min: float,
    score_max: float,
) -> pd.DataFrame | None:
    """Keep the switches that flip the model and score their implausibility."""
    flipped = np.where(predictions != original_prediction)[0]
    if len(flipped) == 0:
        return None
    scores = scorer.predict(switched.iloc[flipped])
    records = pd.DataFrame([switches[i] + (predictions[i],) for i in flipped], columns=RECORD_COLUMNS[:5])
    records["anomaly_score"] = (scores - score_min) / (score_max - score_min)
    records["expected_label"] = mined_class
    return records.merge(factual, how="cross")


def switch_records_vectorized(
    model: Any,
    factual: pd.DataFrame,
    counterfactuals: pd.DataFrame,
    mined_class: Any,
    scorer: Any,
    score_min: float,
    score_max: float,
    target_column: str,
) -> pd.DataFrame | None:
    """Same result as :func:`switch_records` when :func:`rows_share_dtypes` holds.

    ``factual`` is the one-row frame rebuilt from the factual's row: its
    dtypes are those of every frame that :func:`switch_records` would rebuild,
    so the counterfactuals are cast to them once and compared column by
    column instead of cell by cell.
    """
    original_prediction = model.predict(factual).item()
    dtypes = factual.dtypes.to_dict()
    candidates = counterfactuals.astype(dtypes)
    columns = list(factual.columns)
    factual_values = [factual[c].values for c in columns]
    candidate_values = [candidates[c].values for c in columns]
    differs = np.column_stack(
        [np.asarray(values != f[0], dtype=bool) for values, f in zip(candidate_values, factual_values, strict=True)]
    )
    # row-major order: counterfactual by counterfactual, features in column order
    rows, features = np.nonzero(differs)
    if len(rows) == 0:
        return None

    data = {}
    for j, column in enumerate(columns):
        values = np.repeat(np.asarray(factual_values[j], dtype=object), len(rows))
        switched_here = features == j
        values[switched_here] = np.asarray(candidate_values[j], dtype=object)[rows[switched_here]]
        data[column] = pd.Series(values, dtype=dtypes[column])
    switched = pd.DataFrame(data)
    switches = [
        (columns[j], factual_values[j][0], candidate_values[j][i], original_prediction)
        for i, j in zip(rows, features, strict=True)
    ]
    predictions = model.predict(switched)
    switched[target_column] = predictions
    return _records(
        switched, switches, predictions, original_prediction, factual, mined_class, scorer, score_min, score_max
    )


_UNIFORM_KINDS = ("string", "integer", "floating", "boolean")


def rows_share_dtypes(frame: pd.DataFrame) -> bool:
    """Whether every one-row frame rebuilt from a row of ``frame`` gets the same dtypes.

    It holds for numeric columns and for object columns whose values are all
    strings, all integers, all floats or all booleans; mixed object columns
    (numbers and strings, integers and floats) and extension dtypes are left
    to the row-by-row path.
    """
    for column in frame.columns:
        dtype = frame[column].dtype
        if isinstance(dtype, np.dtype) and dtype.kind in "biuf":
            continue
        if not pd.api.types.is_object_dtype(dtype):
            return False
        if pd.api.types.infer_dtype(frame[column], skipna=False) not in _UNIFORM_KINDS:
            return False
    return True


def pair_records(
    model: Any,
    factuals: pd.DataFrame,
    counterfactuals: list[pd.DataFrame],
    mined_class: Any,
    scorer: Any,
    score_min: float,
    score_max: float,
    target_column: str,
) -> pd.DataFrame:
    """Records of every factual of ``factuals`` against its counterfactuals."""
    # the factuals and the counterfactuals are rows of the same data: the
    # vectorised switches need the dtypes of the rebuilt rows to agree across all of them
    together = pd.concat([factuals, *counterfactuals]) if counterfactuals else factuals
    switch = switch_records_vectorized if rows_share_dtypes(together) else switch_records
    collected = []
    for position in range(factuals.shape[0]):
        row = factuals.iloc[position]
        factual = pd.DataFrame([row], index=[0])
        records = switch(
            model,
            factual,
            counterfactuals[position],
            mined_class,
            scorer,
            score_min,
            score_max,
            target_column,
        )
        if records is not None and records.shape[0] > 0:
            records["sample_index"] = row.name
            collected.append(records)
    if not collected:
        logger.debug("No record flips the model for class %s", mined_class)
        return pd.DataFrame()
    return pd.concat(collected)
