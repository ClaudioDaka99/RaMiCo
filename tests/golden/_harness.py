"""Shared helpers for the golden (characterisation) tests.

The golden tests freeze the behaviour of the original implementation on
small synthetic problems. The fixtures in ``tests/fixtures/golden`` were
produced by ``generate_fixtures.py`` in the reference environment pinned
in ``requirements-reference.txt``. Every later refactoring must reproduce
them: only :func:`run_case` (the adapter to the code under test) may
change, never the expected values.

Three deliberate deviations from a plain run of the original code:

* the YDF Isolation Forest is replaced by :class:`StubIsolationForestLearner`,
  a deterministic scorer, because YDF has no Windows wheels and its scores
  depend on the YDF version;
* ties in nearest-neighbour selection are broken by position (stable
  sort). The original ``np.argsort`` default is not stable, so with ties
  its choice depends on the numpy version and on the CPU;
* the k-medoids "heuristic" initialisation lists the initial medoids in
  sorted order. The original takes them from ``np.argpartition``, whose
  order depends on the numpy version.

One deliberate change of the method: on data with
categorical features the original summarised the numerical features by their
most frequent value, RaMiCo by mean ± standard deviation (median in the rule
chains). On those data the tests compare the rules without the summaries of
the numerical features (:func:`without_numerical_summaries`), and not the
fidelity and incomprehensibility.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import types
import warnings
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar
from unittest import mock

import numpy as np
import pandas as pd

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "golden"

# Explicit numerical tolerances for every float comparison.
RTOL = 1e-6
ATOL = 1e-9

CATEGORICAL = ("color", "segment", "flag")
FIDELITY_KS = (3, 7)
FIDELITY_MODES = ("and", "or")


@dataclass(frozen=True)
class Case:
    """One golden scenario: a dataset, a mining strategy and its parameters."""

    name: str
    dataset: str  # "mixed" (numeric + categorical features) or "numeric"
    n_classes: int
    strategy: str  # value of the original ``mining`` argument
    n_factuals: int = 6
    n_counterfactuals: int = 5
    seed: int = 7
    n_samples: int = 160


CASES = (
    Case("mixed_binary_random", "mixed", 2, "random"),
    Case("mixed_binary_boundary", "mixed", 2, "margin"),
    Case("mixed_binary_prototype", "mixed", 2, "multicenter"),
    Case("mixed_binary_ensemble", "mixed", 2, "all_groups"),
    Case("numeric_binary_ensemble", "numeric", 2, "all_groups", seed=3, n_samples=140),
    Case("mixed_multiclass_random", "mixed", 3, "random", seed=5),
    Case("mixed_multiclass_ensemble", "mixed", 3, "all_groups", seed=5),
)


# --------------------------------------------------------------------------
# Synthetic data and black-box model
# --------------------------------------------------------------------------


class ToyModel:
    """Deterministic black box used in place of a trained classifier.

    It accepts DataFrames whatever the column dtypes (the original code
    often passes object columns) and returns integer labels.
    """

    def __init__(self, n_classes: int) -> None:
        self.n_classes = n_classes

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(X)

        def num(column: str) -> np.ndarray:
            return pd.to_numeric(X[column]).to_numpy(dtype=float)

        z = (
            0.05 * (num("age") - 45.0)
            + 0.9 * num("score")
            + 0.03 * (num("hours") - 30.0)
            + 0.3 * (np.log(num("income")) - 10.0)
        )
        if "segment" in X.columns:
            z = z + 0.6 * X["segment"].astype(str).isin(["A", "B"]).to_numpy(dtype=float)
        if "color" in X.columns:
            z = z - 0.7 * (X["color"].astype(str) == "red").to_numpy(dtype=float)
        if "flag" in X.columns:
            z = z + 0.4 * num("flag")
        return z

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        z = self.decision_function(X)
        if self.n_classes == 2:
            return (z > 0.0).astype(np.int64)
        return np.digitize(z, [-0.6, 0.6]).astype(np.int64)


def make_dataset(case: Case) -> tuple[pd.DataFrame, np.ndarray]:
    """Return the features and the (noisy) true labels of ``case``."""
    rs = np.random.RandomState(1000 + case.seed)
    n = case.n_samples
    X = pd.DataFrame(
        {
            "age": rs.randint(18, 80, size=n).astype(np.int64),
            "income": np.round(rs.lognormal(mean=10.0, sigma=0.5, size=n), 2),
            "score": np.round(rs.normal(size=n), 3),
            "hours": rs.randint(1, 60, size=n).astype(np.int64),
        }
    )
    if case.dataset == "mixed":
        X["color"] = pd.Series(rs.choice(["red", "green", "blue"], size=n).tolist(), dtype=object)
        X["segment"] = pd.Series(rs.choice(["A", "B", "C", "D"], size=n).tolist(), dtype=object)
        # numeric-coded categorical stored as object, as the experiments do
        X["flag"] = pd.Series(rs.choice([0, 1], size=n).tolist(), dtype=object)
    y = ToyModel(case.n_classes).predict(X)
    # label noise, so that some factuals are misclassified by the model
    flip = rs.rand(n) < 0.12
    shift = 1 + rs.randint(0, case.n_classes - 1, size=n)
    y = np.where(flip, (y + shift) % case.n_classes, y).astype(np.int64)
    return X, y


def categorical_features(case: Case) -> list[str]:
    return list(CATEGORICAL) if case.dataset == "mixed" else []


def dataset_digest(X: pd.DataFrame, y: np.ndarray) -> str:
    """Fingerprint of the generated data, to detect generator drift."""
    payload = json.dumps(
        {
            "columns": list(map(str, X.columns)),
            "rows": [[_to_py(v) for v in row] for row in X.itertuples(index=False, name=None)],
            "y": [int(v) for v in y],
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Isolation Forest stub
# --------------------------------------------------------------------------


class StubIsolationForestModel:
    """Anomaly scorer with the same interface as a trained YDF model.

    Numerical columns contribute their clipped absolute z-score, the other
    columns the rarity of their value in the training data. The total is
    squashed to [0, 1) and returned as float32, like YDF predictions.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        data = pd.DataFrame(data)
        self.columns = list(data.columns)
        self.numeric: dict[str, tuple[float, float]] = {}
        self.frequencies: dict[str, dict[str, float]] = {}
        for column in self.columns:
            values = data[column]
            if pd.api.types.is_numeric_dtype(values):
                as_float = values.astype(float)
                std = float(as_float.std(ddof=0))
                self.numeric[column] = (float(as_float.mean()), std if std > 0 else 1.0)
            else:
                keys = values.map(str)
                self.frequencies[column] = {str(k): float(v) for k, v in keys.value_counts(normalize=True).items()}

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        data = pd.DataFrame(data)
        total = np.zeros(len(data), dtype=float)
        for column, (mean, std) in self.numeric.items():
            values = pd.to_numeric(data[column]).to_numpy(dtype=float)
            total += np.minimum(np.abs((values - mean) / std), 6.0)
        for column, frequencies in self.frequencies.items():
            seen = np.array([frequencies.get(str(v), 0.0) for v in data[column]], dtype=float)
            total += 1.0 - seen
        return (total / (total + len(self.columns))).astype(np.float32)


class StubIsolationForestLearner:
    """Drop-in replacement for ``ydf.IsolationForestLearner``."""

    calls: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, **params: Any) -> None:
        self.params = params
        StubIsolationForestLearner.calls.append(dict(params))

    def train(self, data: pd.DataFrame, **kwargs: Any) -> StubIsolationForestModel:
        return StubIsolationForestModel(data)


def ydf_stub_module() -> types.ModuleType:
    module = types.ModuleType("ydf")
    module.IsolationForestLearner = StubIsolationForestLearner  # type: ignore[attr-defined]
    return module


# --------------------------------------------------------------------------
# Adapter to the implementation under test
# --------------------------------------------------------------------------


# The cases use the strategy names of the original implementation; the
# refactored code uses the names of the paper.
STRATEGY_NAMES = {
    "random": "random",
    "margin": "boundary",
    "multicenter": "prototype",
    "all_groups": "ensemble",
}
# labels of the rules and records, mapped back to the original ones
ORIGINAL_LABELS = {
    "random": "random",
    "boundary": "margin",
    "prototype": "multicenter",
    "nearest-boundary": "similarity",
    "random-pairs": "random random",
}
RULE_KEYS = ["feature", "original_prediction", "switched_prediction", "method"]


def _legacy_records(rules: Any, features: list[str]) -> pd.DataFrame:
    """Public records of a RuleSet in the layout of the original records table."""
    public = rules.records
    legacy = pd.DataFrame(
        {
            "feature": public["feature"],
            "original_value": public["from_value"],
            "switched_value": public["to_value"],
            "original_prediction": public["from_class"],
            "switched_prediction": public["to_class"],
            "anomaly_score": public["implausibility"],
            "expected_label": public["mined_class"],
        }
    )
    for feature in features:
        legacy[feature] = public[f"factual.{feature}"]
    legacy["sample_index"] = public["factual_index"]
    legacy["method"] = public["strategy"].map(ORIGINAL_LABELS).astype(object)
    return legacy


def _legacy_rules(rules: Any) -> pd.DataFrame:
    """Public rules of a RuleSet in the layout and order of the original rules table."""
    public = rules.to_frame()
    is_mean = public["aggregation"] == "mean"
    legacy = pd.DataFrame(
        {
            "feature": public["feature"],
            "original_prediction": public["from_class"],
            "switched_prediction": public["to_class"],
            "original_value": public["from_value"],
            "original_value_std/confidence": public["from_std"].where(is_mean, public["from_confidence"]),
            "switched_value": public["to_value"],
            "switched_value_std/confidence": public["to_std"].where(is_mean, public["to_confidence"]),
            "anomaly_score_mean": public["implausibility"],
            "anomaly_score_std": public["implausibility_std"],
            "switch_count_per_feat": public["n_records"],
            "support": public["support"],
            "expected_label": public["mined_class"],
            "method": public["strategy"].map(ORIGINAL_LABELS).astype(object),
        }
    )
    # the original rules come out of a groupby, sorted by these keys
    return legacy.sort_values(RULE_KEYS, kind="stable").reset_index(drop=True)


@contextmanager
def _implementation() -> Iterator[types.SimpleNamespace]:
    import ramico
    import ramico._plausibility as plausibility
    from ramico import metrics

    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(plausibility, "_import_ydf", ydf_stub_module))
        stack.enter_context(warnings.catch_warnings())
        # the cases are smaller than YDF's sample; the stub has no such problem
        warnings.filterwarnings("ignore", message="YDF's Isolation Forest draws")
        yield types.SimpleNamespace(RaMiCo=ramico.RaMiCo, metrics=metrics)


@functools.cache
def run_case(case: Case) -> dict[str, Any]:
    """Run the implementation under test on ``case`` and collect its outputs."""
    X, y = make_dataset(case)
    model = ToyModel(case.n_classes)
    StubIsolationForestLearner.calls.clear()

    with _implementation() as impl:
        ruleset = impl.RaMiCo(
            model,
            strategy=STRATEGY_NAMES[case.strategy],
            n_factuals=case.n_factuals,
            n_counterfactuals=case.n_counterfactuals,
            random_state=case.seed,
            isolation_forest="ydf",
        ).fit(X, y, categorical_features=categorical_features(case))
        mined = ruleset.mined_indices
        records = _legacy_records(ruleset, list(X.columns))
        rules = _legacy_rules(ruleset)
        fidelity: dict[str, dict[str, dict[str, float]]] = {}
        for mode in FIDELITY_MODES:
            fidelity[mode] = {}
            for k in FIDELITY_KS:
                percentage = impl.metrics.fidelity(model, ruleset, X, k=k, mode=mode)
                per_sample = impl.metrics.incomprehensibility(model, ruleset, X, k=k)
                successes = round(percentage * len(X) / 100)
                fidelity[mode][str(k)] = {
                    # the original rounded the fidelity percentage to two decimals
                    "fidelity": round(successes / len(X) * 100, 2),
                    "tot_nearest_rules": per_sample * len(X),
                    "rules_per_sample": per_sample,
                }

    return {
        "dataset": case.dataset,
        "numerical_features": list(ruleset.metadata["numerical_features"]),
        "data_digest": dataset_digest(X, y),
        "mined_indices": [_to_py(i) for i in mined],
        "records": records,
        "rules": rules,
        "metrics": {
            # run-level implausibility as computed by the experiments
            "implausibility": float(np.round(rules["anomaly_score_mean"].mean(), 3)),
            "implausibility_std": float(np.round(rules["anomaly_score_std"].mean(), 3)),
            "fidelity": fidelity,
        },
        "isolation_forest_params": [dict(c) for c in StubIsolationForestLearner.calls],
    }


# --------------------------------------------------------------------------
# Serialisation and comparison
# --------------------------------------------------------------------------


def _to_py(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def frame_to_jsonable(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "columns": [str(c) for c in frame.columns],
        "dtypes": [str(t) for t in frame.dtypes],
        "index": [_to_py(i) for i in frame.index],
        "rows": [[_to_py(v) for v in row] for row in frame.itertuples(index=False, name=None)],
    }


def outputs_to_jsonable(outputs: dict[str, Any]) -> dict[str, Any]:
    converted = dict(outputs)
    converted["records"] = frame_to_jsonable(outputs["records"])
    converted["rules"] = frame_to_jsonable(outputs["rules"])
    return converted


def fixture_path(case: Case) -> Path:
    return FIXTURES_DIR / f"{case.name}.json"


def write_fixture(case: Case, outputs: dict[str, Any]) -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"case": asdict(case), "outputs": outputs_to_jsonable(outputs)}
    fixture_path(case).write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def load_fixture(case: Case) -> dict[str, Any]:
    payload = json.loads(fixture_path(case).read_text(encoding="utf-8"))
    assert payload["case"] == asdict(case), f"fixture {case.name} was generated for another case"
    return payload["outputs"]


_NUMERIC_DTYPES = ("int", "uint", "float", "bool")


def _is_numeric_dtype_name(name: str) -> bool:
    return name.startswith(_NUMERIC_DTYPES)


def _values_match(actual: Any, expected: Any) -> bool:
    actual, expected = _to_py(actual), _to_py(expected)
    if isinstance(expected, float) and math.isnan(expected):
        return isinstance(actual, float) and math.isnan(actual)
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        # object columns: int and float are kept distinct on purpose
        return type(actual) is type(expected) and math.isclose(actual, expected, rel_tol=RTOL, abs_tol=ATOL)
    return type(actual) is type(expected) and actual == expected


def assert_frame_matches(actual: pd.DataFrame, expected: dict[str, Any], what: str) -> None:
    """Compare a DataFrame with its serialised golden version."""
    assert [str(c) for c in actual.columns] == expected["columns"], f"{what}: columns differ"
    assert len(actual) == len(expected["rows"]), f"{what}: {len(actual)} rows, expected {len(expected['rows'])}"
    assert [_to_py(i) for i in actual.index] == expected["index"], f"{what}: index differs"
    for j, (column, dtype) in enumerate(zip(expected["columns"], expected["dtypes"], strict=True)):
        expected_values = [row[j] for row in expected["rows"]]
        values = actual.iloc[:, j]
        if _is_numeric_dtype_name(dtype):
            assert pd.api.types.is_numeric_dtype(values), (
                f"{what}.{column}: dtype {values.dtype}, expected numeric ({dtype})"
            )
            np.testing.assert_allclose(
                values.to_numpy(dtype=float),
                np.asarray(expected_values, dtype=float),
                rtol=RTOL,
                atol=ATOL,
                err_msg=f"{what}.{column}",
            )
        else:
            assert not pd.api.types.is_numeric_dtype(values), (
                f"{what}.{column}: dtype {values.dtype}, expected non-numeric ({dtype})"
            )
            for i, (a, e) in enumerate(zip(values.tolist(), expected_values, strict=True)):
                assert _values_match(a, e), f"{what}.{column}[{i}]: {a!r} != {e!r}"


#: The columns of the rules that summarise the values of a feature.
SUMMARY_COLUMNS = ("original_value", "original_value_std/confidence", "switched_value", "switched_value_std/confidence")


def without_numerical_summaries(
    actual: pd.DataFrame, expected: dict[str, Any], numerical: list[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Both tables of rules with the summaries of the numerical features left blank.

    On data with categorical features the original implementation summarised
    the numerical features by their most frequent value, RaMiCo by mean ±
    standard deviation: these are the only cells that
    differ. The other columns of those rules, and every column of the rules
    on categorical features, are still compared.
    """
    actual = actual.copy()
    rows = actual["feature"].isin(numerical).to_numpy()
    for column in SUMMARY_COLUMNS:
        actual.loc[rows, column] = np.nan if pd.api.types.is_numeric_dtype(actual[column]) else None
    feature = expected["columns"].index("feature")
    positions = [expected["columns"].index(column) for column in SUMMARY_COLUMNS]
    blanked = []
    for row in expected["rows"]:
        row = list(row)
        if row[feature] in numerical:
            for j in positions:
                row[j] = None  # NaN in a numeric column, None in an object one
        blanked.append(row)
    return actual, {**expected, "rows": blanked}


def assert_metrics_match(actual: dict[str, Any], expected: dict[str, Any], what: str) -> None:
    """Recursively compare nested dicts of floats with explicit tolerances."""
    assert set(actual) == set(expected), f"{what}: keys {sorted(actual)} != {sorted(expected)}"
    for key, value in expected.items():
        if isinstance(value, dict):
            assert_metrics_match(actual[key], value, f"{what}.{key}")
        else:
            np.testing.assert_allclose(actual[key], value, rtol=RTOL, atol=ATOL, err_msg=f"{what}.{key}")
