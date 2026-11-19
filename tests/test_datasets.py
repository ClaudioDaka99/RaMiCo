"""Tests for ramico.datasets (no network: downloads are simulated)."""

from __future__ import annotations

import hashlib
import io

import pytest

from ramico import datasets
from ramico.exceptions import DataError

HEART_ROWS = (
    "63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0,0\n"
    "67.0,1.0,4.0,160.0,286.0,0.0,2.0,108.0,1.0,1.5,2.0,3.0,3.0,2\n"
    "67.0,1.0,4.0,120.0,229.0,0.0,2.0,129.0,1.0,2.6,2.0,2.0,7.0,1\n"
    "37.0,1.0,3.0,130.0,250.0,0.0,0.0,187.0,0.0,3.5,3.0,?,3.0,0\n"
)
ADULT_ROWS = (
    "39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, "
    "United-States, <=50K\n"
    "52, Self-emp-inc, 287927, HS-grad, 9, Married-civ-spouse, Exec-managerial, Wife, White, Female, 15024, 0, 40, "
    "United-States, >50K\n"
    "54, ?, 180211, Some-college, 10, Married-civ-spouse, ?, Husband, Asian-Pac-Islander, Male, 0, 0, 60, South, >50K\n"
)


def test_parse_heart_disease(tmp_path):
    path = tmp_path / "processed.cleveland.data"
    path.write_text(HEART_ROWS)
    data = datasets.parse_heart_disease(path)
    assert data.X.shape == (3, 13)  # the row with a missing value is dropped
    assert data.y.tolist() == [0, 1, 1]
    assert data.X["cp"].tolist() == ["typical angina", "asymptomatic", "asymptomatic"]
    assert data.X["thal"].tolist() == ["fixed defect", "normal", "reversible defect"]
    assert str(data.X["sex"].dtype) == "category"
    X, y = data
    assert X is data.X and y is data.y
    raw = datasets.parse_heart_disease(path, readable=False)
    assert raw.X["cp"].tolist() == [1, 4, 4]
    assert raw.categorical_features == ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]


def test_parse_adult(tmp_path):
    path = tmp_path / "adult.data"
    path.write_text(ADULT_ROWS)
    data = datasets.parse_adult(path)
    assert data.X.shape == (2, 14)
    assert data.y.tolist() == [0, 1]
    assert data.target_names == {0: "<=50K", 1: ">50K"}
    assert str(data.X["workclass"].dtype) == "category"
    assert data.X["age"].dtype.kind == "i"


@pytest.fixture
def fake_download(monkeypatch, tmp_path):
    content = HEART_ROWS.encode()
    calls = []
    monkeypatch.setitem(
        datasets._FILES,
        "heart_disease",
        ("https://example.org/processed.cleveland.data", hashlib.sha256(content).hexdigest()),
    )

    def urlopen(url, timeout):
        calls.append(url)
        return io.BytesIO(content)

    monkeypatch.setattr(datasets.urllib.request, "urlopen", urlopen)
    return calls


def test_download_is_cached_and_checked(fake_download, tmp_path):
    first = datasets.load_heart_disease(data_home=tmp_path)
    second = datasets.load_heart_disease(data_home=tmp_path)
    assert len(fake_download) == 1  # the second load reads the cache
    assert first.X.equals(second.X)
    assert (tmp_path / "processed.cleveland.data").exists()


def test_corrupted_download_is_rejected(fake_download, tmp_path, monkeypatch):
    monkeypatch.setitem(datasets._FILES, "heart_disease", ("https://example.org/x", "0" * 64))
    with pytest.raises(DataError, match="checksum"):
        datasets.load_heart_disease(data_home=tmp_path)
    assert not list(tmp_path.iterdir())  # nothing half-written is left behind


def test_missing_file_without_download(tmp_path):
    with pytest.raises(DataError, match="download=False"):
        datasets.load_adult(data_home=tmp_path, download=False)


def test_data_home_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("RAMICO_DATA", str(tmp_path / "cache"))
    assert datasets.get_data_home() == tmp_path / "cache"
    assert (tmp_path / "cache").is_dir()
