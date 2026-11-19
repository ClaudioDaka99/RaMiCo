"""Tests of the input checks of the public API."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent / "golden"))
import _harness as h

from ramico import RaMiCo
from ramico._distance import gower_matrix, gower_rows
from ramico._validation import BlackBox, check_data, check_random_state, free_column_name
from ramico.exceptions import DataError, ModelError, RaMiCoError

LABELS = np.array([0, 1, 0, 1, 0, 1])
CATEGORICAL = list(h.CATEGORICAL)


def test_gower_between_frames_of_equal_length():
    # gower 0.1.2 mirrors the upper triangle when the frames have as many rows
    x, y = pd.DataFrame({"age": [0.0, 10.0]}), pd.DataFrame({"age": [1.0, 2.0]})
    np.testing.assert_allclose(gower_matrix(x, y), [[0.1, 0.2], [0.9, 0.8]], rtol=1e-6)
    rng = np.random.default_rng(0)
    a = pd.DataFrame({"n": rng.normal(size=30), "c": rng.choice(list("pqr"), 30)})
    b = pd.DataFrame({"n": rng.normal(size=30), "c": rng.choice(list("pqr"), 30)})
    span = np.ptp(pd.concat([a, b])["n"].to_numpy())  # Series.agg(np.ptp) is elementwise in pandas 2
    expected = ((a["n"].to_numpy()[:, None] - b["n"].to_numpy()[None]).__abs__() / span
                + (a["c"].to_numpy()[:, None] != b["c"].to_numpy()[None])) / 2  # fmt: skip
    np.testing.assert_allclose(gower_matrix(a, b), expected, atol=1e-6)
    np.testing.assert_allclose(gower_matrix(a, b), gower_matrix(b, a).T, atol=1e-6)


def test_gower_is_unchanged_for_frames_of_different_length():
    import gower

    rng = np.random.default_rng(1)
    a = pd.DataFrame({"n": rng.normal(size=30), "c": rng.choice(list("pqr"), 30)}).astype({"c": object})
    b = a.iloc[:17]
    np.testing.assert_array_equal(gower_matrix(a, b), gower.gower_matrix(a, b))
    np.testing.assert_array_equal(gower_matrix(a), gower.gower_matrix(a))


def one_row_at_a_time(x: pd.DataFrame, y: pd.DataFrame) -> np.ndarray:
    """What the metrics computed before gower_rows: one-row frames, dtypes inferred from the values."""
    return np.vstack([gower_matrix(pd.DataFrame([row]), y)[0] for _, row in x.iterrows()])


def mixed_frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "income": rng.lognormal(10, 1, n),
            "count": rng.integers(0, 6, n),
            "score": -rng.random(n),
            "colour": pd.Series(rng.choice(["red", "green", "blue"], n), dtype=object),
            "flag": pd.Series(rng.choice([0, 1], n).tolist(), dtype=object),  # numbers in an object column
            "segment": pd.Categorical(rng.choice(list("ABCD"), n)),
        }
    )
    frame.loc[rng.random(n) < 0.1, "income"] = np.nan
    return frame


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gower_rows_equal_one_row_at_a_time(seed):
    frame = mixed_frame(260, seed)
    x, y = frame.iloc[:60], frame.iloc[60:]
    fast = gower_rows(x, y, max_cells=1000)  # several blocks
    assert fast is not None
    np.testing.assert_array_equal(fast.view(np.uint32), one_row_at_a_time(x, y).view(np.uint32))


def test_gower_rows_with_a_constant_column():
    frame = mixed_frame(80, 3).assign(count=4)
    x, y = frame.iloc[:20], frame.iloc[20:]
    np.testing.assert_array_equal(gower_rows(x, y), one_row_at_a_time(x, y))


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10)}),  # no categorical column
        pd.DataFrame({"a": np.arange(10.0), "c": pd.Series(["x", None] * 5, dtype=object)}),  # missing category
        pd.DataFrame({"a": np.arange(10.0), "m": pd.Series([1, "x"] * 5, dtype=object)}),  # numbers and text
    ],
)
def test_gower_rows_leaves_the_other_cases_to_gower(frame):
    assert gower_rows(frame.iloc[:3], frame.iloc[3:]) is None


def small_frame(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame({"a": np.arange(n, dtype=float), "b": list("xyzxyz")[:n]})


@pytest.fixture(scope="module")
def data():
    return h.make_dataset(h.Case("validation", "mixed", 2, "random", n_samples=120))


def fit(data, class_names=None, **kwargs):
    X, y = data
    params = dict(strategy="random", n_factuals=3, n_counterfactuals=2, random_state=0, isolation_forest="native")
    params.update(kwargs)
    model = params.pop("model", h.ToyModel(2))
    return RaMiCo(model, **params).fit(X, y, categorical_features=CATEGORICAL, class_names=class_names)


def test_errors_share_a_base_class_and_the_builtin_ones():
    assert issubclass(DataError, RaMiCoError) and issubclass(DataError, ValueError)
    assert issubclass(ModelError, RaMiCoError) and issubclass(ModelError, TypeError)


class TestRandomState:
    def test_none_is_the_global_state(self):
        assert check_random_state(None) is np.random.mtrand._rand

    def test_an_integer_seeds_a_new_random_state(self):
        np.testing.assert_array_equal(check_random_state(3).rand(4), np.random.RandomState(3).rand(4))
        assert isinstance(check_random_state(np.int64(3)), np.random.RandomState)

    @pytest.mark.parametrize("generator", [np.random.RandomState(0), np.random.default_rng(0)])
    def test_generators_are_used_as_they_are(self, generator):
        assert check_random_state(generator) is generator

    @pytest.mark.parametrize("value", [True, 1.5, "0"])
    def test_other_values_are_rejected(self, value):
        with pytest.raises(TypeError, match="random_state must be"):
            check_random_state(value)


class TestBlackBox:
    def test_estimator_with_predict(self):
        class Estimator:
            def predict(self, X):
                return np.zeros(len(X), dtype=int)

        assert BlackBox(Estimator()).predict(small_frame()).shape == (6,)

    def test_function_returning_a_column_vector(self):
        assert BlackBox(lambda X: np.zeros((len(X), 1))).predict(small_frame()).shape == (6,)

    def test_object_without_predict(self):
        with pytest.raises(ModelError, match=r"predict\(X\) method"):
            BlackBox(object())

    @pytest.mark.parametrize("output", [np.zeros((6, 2)), np.zeros(5)])
    def test_wrong_number_of_labels(self, output):
        with pytest.raises(ModelError, match="one label per row"):
            BlackBox(lambda X: output).predict(small_frame())


def test_free_column_name():
    assert free_column_name(["a", "b"], "target") == "target"
    assert free_column_name(["target", "_target"], "target") == "__target"


class TestCheckData:
    def test_textual_columns_become_object_on_a_copy(self):
        X = small_frame().assign(
            c=pd.Series(list("pqpqpq"), dtype="category"), s=pd.Series(list("uvuvuv"), dtype="string")
        )
        checked, labels = check_data(X, LABELS)
        assert checked is not X
        assert checked["c"].dtype == object and checked["s"].dtype == object
        assert checked["a"].dtype == float
        assert isinstance(X["c"].dtype, pd.CategoricalDtype)  # the caller's data is untouched
        assert labels.tolist() == LABELS.tolist()

    def test_declared_categorical_features_by_name_or_position(self):
        X = small_frame().assign(code=[1, 2, 1, 2, 1, 2])
        by_name, _ = check_data(X, LABELS, categorical_features=["code"])
        by_position, _ = check_data(X, LABELS, categorical_features=[2])
        assert by_name["code"].dtype == object and by_position["code"].dtype == object

    def test_arrays_get_feature_names(self):
        values = np.arange(12.0).reshape(6, 2)
        assert list(check_data(values, LABELS)[0].columns) == ["x0", "x1"]
        assert list(check_data(values, LABELS, feature_names=["p", "q"])[0].columns) == ["p", "q"]

    def test_column_names_become_strings(self):
        checked, _ = check_data(pd.DataFrame({0: [1.0, 2.0], 1: [3.0, 4.0]}), [0, 1])
        assert list(checked.columns) == ["0", "1"]

    def test_labels_as_a_column_vector(self):
        assert check_data(small_frame(), LABELS.reshape(-1, 1))[1].ndim == 1

    @pytest.mark.parametrize(
        ("X", "y", "kwargs", "message"),
        [
            (small_frame(), LABELS, {"feature_names": ["a", "b"]}, "only used when X is an array"),
            (np.arange(6.0), LABELS, {}, "2-D array"),
            (np.zeros((6, 2)), LABELS, {"feature_names": ["p"]}, "1 feature names for 2 columns"),
            (pd.DataFrame([[1, 2]] * 6, columns=[1, "1"]), LABELS, {}, "duplicated column names"),
            (small_frame(1), LABELS[:1], {}, "at least two rows"),
            (small_frame().assign(a=[1.0, np.nan, 3, 4, 5, 6]), LABELS, {}, r"missing values \(columns \['a'\]\)"),
            (small_frame(), LABELS[:5], {}, "one label per row"),
            (small_frame(), np.array([0, 1, None, 1, 0, 1], dtype=object), {}, "missing labels"),
            (small_frame(), np.zeros(6), {}, "at least two classes"),
            (small_frame(), LABELS, {"categorical_features": [5]}, "index 5 is out of range"),
            (small_frame(), LABELS, {"categorical_features": ["c"]}, "'c' is not a column"),
            (small_frame(), LABELS, {"categorical_features": [True]}, "True is not a column"),
        ],
    )
    def test_invalid_inputs(self, X, y, kwargs, message):
        with pytest.raises(DataError, match=message):
            check_data(X, y, **kwargs)


class TestExplainerParameters:
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"n_factuals": 0}, "n_factuals must be a positive integer"),
            ({"n_factuals": True}, "n_factuals must be a positive integer"),
            ({"n_counterfactuals": 2.5}, "n_counterfactuals must be a positive integer"),
            ({"isolation_forest": "sklearn"}, "isolation_forest must be one of"),
            ({"distance": "euclidean"}, 'distance must be "gower" or a callable'),
            ({"strategy": "margin"}, "unknown strategy 'margin'; available: boundary, ensemble"),
        ],
    )
    def test_invalid_parameters(self, data, kwargs, message):
        with pytest.raises(ValueError, match=message):
            fit(data, **kwargs)

    def test_invalid_strategy_type(self, data):
        with pytest.raises(TypeError, match="strategy must be a name or a MiningStrategy"):
            fit(data, strategy=3)

    def test_custom_distance(self, data):
        calls = []

        def distance(x, y=None):
            calls.append(len(x))
            return gower_matrix(x, y)

        custom = fit(data, distance=distance)
        assert calls
        pd.testing.assert_frame_equal(custom.to_frame(), fit(data).to_frame())

    def test_class_names(self, data):
        named = fit(data, class_names=["no", "yes"])  # by position in the sorted labels
        assert named.metadata["class_names"] == {0: "no", 1: "yes"}
        assert named.class_name(1) == "yes" and fit(data).class_name(1) == "1"
        assert fit(data, class_names={1: "yes"}).class_name(0) == "0"
        with pytest.raises(ValueError, match="3 class names for 2 classes"):
            fit(data, class_names=["a", "b", "c"])

    def test_random_state_in_the_metadata(self, data):
        assert fit(data, random_state=4).metadata["random_state"] == 4
        assert fit(data, random_state=np.random.default_rng(4)).metadata["random_state"] == "generator"

    def test_model_that_never_changes_its_prediction(self, data):
        with pytest.warns(RuntimeWarning, match="the RuleSet is empty"):
            empty = fit(data, model=lambda X: np.zeros(len(X), dtype=int))
        assert len(empty) == 0 and empty.records.empty
        assert empty.metadata["mined_indices"]  # factuals were mined, none flipped the model
