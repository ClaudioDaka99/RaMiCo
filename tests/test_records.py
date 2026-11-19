"""The vectorised switches reproduce the row-by-row ones exactly.

Beyond the records, the frames handed to the model and to the Isolation
Forest must be identical too, dtypes included: real models (pipelines,
YDF) depend on them, while the golden tests use a model that does not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent / "golden"))
import _harness as h

from ramico._records import pair_records, rows_share_dtypes, switch_records, switch_records_vectorized
from ramico._validation import check_data


class Spy:
    """Black box and Isolation Forest that record every frame they receive."""

    def __init__(self, n_classes: int = 2) -> None:
        self.model = h.ToyModel(n_classes)
        self.calls: list[pd.DataFrame] = []

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.calls.append(X.copy())
        if "target" in X.columns:  # scorer: a deterministic score per row
            return np.array([len(str(tuple(row))) for row in X.itertuples(index=False)], dtype=float)
        return self.model.predict(X)


def datasets() -> dict[str, pd.DataFrame]:
    mixed, _ = h.make_dataset(h.Case("records", "mixed", 2, "random", n_samples=60))
    numeric, _ = h.make_dataset(h.Case("records", "numeric", 2, "random", n_samples=60))
    checked, _ = check_data(  # categories and pandas strings become object columns, as in fit
        mixed.astype({"color": "category", "segment": "string"}), np.arange(60) % 2
    )
    flags = mixed.assign(flag=mixed["flag"].astype(int), member=mixed["flag"].astype(bool))
    floats = mixed.assign(flag=mixed["flag"].astype(float).astype(object))  # object column of floats
    return {"mixed": mixed, "numeric": numeric, "checked": checked, "bools": flags, "object floats": floats}


def run(switch, X: pd.DataFrame, factual_rows: list[int], counterfactual_rows: list[int]):
    spy = Spy()
    counterfactuals = X.iloc[counterfactual_rows]
    results = []
    for position in factual_rows:
        factual = pd.DataFrame([X.iloc[position]], index=[0])
        results.append(switch(spy, factual, counterfactuals, 1, spy, 0.0, 100.0, "target"))
    return results, spy.calls


@pytest.mark.parametrize("name", list(datasets()))
def test_vectorised_switches_are_identical(name):
    X = datasets()[name]
    assert rows_share_dtypes(X)
    factuals, counterfactuals = list(range(12)), list(range(20, 45))
    expected, expected_calls = run(switch_records, X, factuals, counterfactuals)
    actual, actual_calls = run(switch_records_vectorized, X, factuals, counterfactuals)
    assert any(r is not None for r in expected)
    for a, e in zip(actual, expected, strict=True):
        if e is None:
            assert a is None
        else:
            pd.testing.assert_frame_equal(a, e, check_exact=True)
            for column in ["original_value", "switched_value"]:  # same Python and numpy types inside
                assert [type(v) for v in a[column]] == [type(v) for v in e[column]]
    assert len(actual_calls) == len(expected_calls)
    for a, e in zip(actual_calls, expected_calls, strict=True):
        pd.testing.assert_frame_equal(a, e, check_exact=True)


def test_mixed_object_columns_use_the_row_by_row_path(monkeypatch):
    X, _ = h.make_dataset(h.Case("records", "mixed", 2, "random", n_samples=40))
    X["flag"] = pd.Series([1 if i % 2 else "one" for i in range(40)], dtype=object)  # numbers and strings
    assert not rows_share_dtypes(X)
    assert not rows_share_dtypes(X.astype({"age": "Int64"}).drop(columns="flag"))  # extension dtype

    used = []
    monkeypatch.setattr("ramico._records.switch_records", lambda *a: used.append("rows"))
    monkeypatch.setattr("ramico._records.switch_records_vectorized", lambda *a: used.append("vectorised"))
    spy = Spy()
    pair_records(spy, X.iloc[:2], [X.iloc[10:15]] * 2, 0, spy, 0.0, 1.0, "target")
    pair_records(spy, X.drop(columns="flag").iloc[:2], [X.drop(columns="flag").iloc[10:15]] * 2, 0, spy, 0.0, 1.0, "t")
    assert used == ["rows", "rows", "vectorised", "vectorised"]
