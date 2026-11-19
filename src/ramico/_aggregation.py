"""Aggregation of the elementary records into rules.

Records are grouped by feature, starting class, arrival class and strategy.
Numerical features are summarised by mean and standard deviation, the others
by mode and by the share of records holding the mode ("confidence"), with
the rounding of the original implementation.

Whether a feature is numerical depends on its column in ``X``, not on the
values of each group: the original implementation decided from the group,
which on data with categorical features summarised the numerical features
by their mode too.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

import numpy as np
import pandas as pd

GROUP_KEYS = ["feature", "original_prediction", "switched_prediction", "method"]
RULE_COLUMNS = [
    "feature",
    "original_prediction",
    "switched_prediction",
    "original_value",
    "original_value_std/confidence",
    "switched_value",
    "switched_value_std/confidence",
    "anomaly_score_mean",
    "anomaly_score_std",
    "switch_count_per_feat",
    "support",
    "expected_label",
    "method",
    "aggregation",
]


def _summarise(group: pd.DataFrame, numerical: bool) -> dict[str, Any]:
    original, switched = group["original_value"], group["switched_value"]
    size = original.size
    if numerical:
        original, switched = pd.to_numeric(original), pd.to_numeric(switched)
        aggregation = "mean"
        summary: dict[str, Any] = {
            "original_value": np.round(original.mean(), 2),
            "original_value_std/confidence": round(original.std(), 2) if size > 1 else 0,
            "switched_value": np.round(switched.mean(), 2),
            "switched_value_std/confidence": round(switched.std(), 2) if size > 1 else 0,
        }
    else:
        aggregation = "mode"
        original_mode = original.mode()[0]
        switched_mode = switched.mode()[0]
        summary = {
            "original_value": original_mode,
            "original_value_std/confidence": np.round(original[original == original_mode].size / size * 100, 2),
            "switched_value": switched_mode,
            "switched_value_std/confidence": np.round(switched[switched == switched_mode].size / size * 100, 2),
        }
    scores = group["anomaly_score"]
    summary["switch_count_per_feat"] = int(size)
    summary["anomaly_score_mean"] = round(scores.mean(), 3)
    summary["anomaly_score_std"] = round(scores.std(), 3) if scores.size > 1 else 0
    labels = group["expected_label"].mode()
    summary["expected_label"] = labels.iloc[0] if not labels.empty else None
    # The original returned pd.Series(summary): when every value is a number
    # pandas casts them all to float (a mode of 40 becomes 40.0). Going through
    # the same constructor keeps the values identical.
    return {**pd.Series(summary).to_dict(), "aggregation": aggregation}


def aggregate(records: pd.DataFrame, numerical_features: Collection[str]) -> pd.DataFrame:
    """Rules of ``records``, one row per (feature, from, to, strategy).

    ``numerical_features`` names the features summarised by mean and
    standard deviation; the others are summarised by their mode.
    """
    if records.empty:
        return pd.DataFrame(columns=RULE_COLUMNS)
    rows = []
    for keys, group in records.groupby(GROUP_KEYS, sort=True):
        rows.append(dict(zip(GROUP_KEYS, keys, strict=True), **_summarise(group, keys[0] in numerical_features)))
    rules = pd.DataFrame(rows)
    rules["support"] = np.round(rules["switch_count_per_feat"] / len(records) * 100, 2)
    return rules[RULE_COLUMNS]
