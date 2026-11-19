"""Tests of the choice of the Isolation Forest engine."""

from __future__ import annotations

import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent / "golden"))
import _harness as h

from ramico import _plausibility as plausibility
from ramico.exceptions import MissingDependencyError


@pytest.fixture(scope="module")
def data():
    X, y = h.make_dataset(h.Case("plausibility", "mixed", 2, "random", n_samples=60))
    return X.assign(target=y)


def test_unknown_engine(data):
    with pytest.raises(ValueError, match="isolation_forest must be one of"):
        plausibility.fit_isolation_forest(data, engine="sklearn")


def test_without_ydf(monkeypatch, data):
    monkeypatch.setitem(sys.modules, "ydf", None)  # makes `import ydf` fail
    assert not plausibility.ydf_available()
    with pytest.raises(MissingDependencyError, match=r'pip install "ramico\[plausibility\]"'):
        plausibility.fit_isolation_forest(data, engine="ydf")
    assert plausibility.fit_isolation_forest(data, engine="auto", random_state=0).engine == "native"


def test_ydf_engine(monkeypatch, data):
    stub = h.ydf_stub_module()
    stub.__version__ = "9.9.9"
    monkeypatch.setitem(sys.modules, "ydf", stub)
    assert plausibility.ydf_available()
    with pytest.warns(RuntimeWarning, match="draws 256 examples per tree but the data have 60"):
        scorer = plausibility.fit_isolation_forest(data)  # "auto" prefers YDF
    assert scorer.engine == "ydf 9.9.9" and scorer.predict(data).shape == (len(data),)
    assert h.StubIsolationForestLearner.calls[-1] == {"num_trees": 1000, "max_depth": -1}  # the paper's forest
    del stub.__version__
    large = pd.concat([data] * 5, ignore_index=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # 300 rows: no warning
        assert plausibility.fit_isolation_forest(large, engine="ydf").engine == "ydf"


def test_native_engine_is_deterministic(data):
    first = plausibility.fit_isolation_forest(data, engine="native", random_state=3)
    again = plausibility.fit_isolation_forest(data, engine="native", random_state=3)
    assert first.engine == "native"
    np.testing.assert_array_equal(first.predict(data), again.predict(data))


def test_native_output_is_silenced_and_restored(capfd):
    import os

    with plausibility.native_output_silenced():
        os.write(1, b"from C++ to stdout\n")
        os.write(2, b"from C++ to stderr\n")
    os.write(1, b"after\n")
    out, err = capfd.readouterr()
    assert out == "after\n" and err == ""


def test_real_ydf_trains_without_printing(capfd):
    pytest.importorskip("ydf")
    rng = np.random.default_rng(0)
    data = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    plausibility.fit_isolation_forest(data, engine="ydf")
    out, err = capfd.readouterr()
    assert out == "" and err == ""


def test_ydf_gets_the_categorical_features_as_text(monkeypatch):
    # YDF refuses an object column of floats, e.g. a categorical feature coded 0.0, 1.0...
    seen = {}

    class Model:
        def predict(self, data):
            seen["predict"] = data
            return np.zeros(len(data))

    class Learner:
        def __init__(self, **params):
            pass

        def train(self, data, **kwargs):
            seen["train"] = data
            return Model()

    module = types.ModuleType("ydf")
    module.IsolationForestLearner = Learner
    monkeypatch.setitem(sys.modules, "ydf", module)
    data = pd.DataFrame({"code": pd.Series([0.0, 1.0, 2.0] * 100, dtype=object), "x": np.arange(300.0)})
    plausibility.fit_isolation_forest(data, engine="ydf").predict(data)
    for frame in seen.values():
        assert frame["code"].map(type).eq(str).all() and frame["x"].dtype == np.float64
    assert data["code"].map(type).eq(float).all()  # the caller's data are left as they were


def test_real_ydf_accepts_categorical_features_coded_by_floats(capfd):  # capfd: YDF's output is redirected
    pytest.importorskip("ydf")
    rng = np.random.default_rng(0)
    data = pd.DataFrame({"code": pd.Series(rng.choice([0.0, 1.0, 2.0], 300), dtype=object), "x": rng.normal(size=300)})
    scorer = plausibility.fit_isolation_forest(data, engine="ydf")
    assert np.isfinite(scorer.predict(data)).all()
