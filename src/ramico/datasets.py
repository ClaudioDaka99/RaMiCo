"""Public datasets of the paper: Adult and Heart Disease (UCI Machine Learning Repository).

Nothing is shipped with the package: the files are downloaded on first use,
checked against a SHA-256 checksum and kept in a local cache, by default
``~/.cache/ramico`` (override with ``data_home`` or the ``RAMICO_DATA``
environment variable). Rows with missing values are dropped, as in the
experiments of the paper.

Examples
--------
>>> from ramico.datasets import load_heart_disease   # doctest: +SKIP
>>> data = load_heart_disease()                      # doctest: +SKIP
>>> X, y = data                                      # doctest: +SKIP
>>> data.target_names                                # doctest: +SKIP
{0: 'no disease', 1: 'disease'}
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ramico.exceptions import DataError

__all__ = ["Dataset", "get_data_home", "load_adult", "load_heart_disease"]

_UCI = "https://archive.ics.uci.edu/ml/machine-learning-databases"
_FILES = {
    "heart_disease": (
        f"{_UCI}/heart-disease/processed.cleveland.data",
        "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8",
    ),
    "adult": (
        f"{_UCI}/adult/adult.data",
        "5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d",
    ),
}


@dataclass(frozen=True)
class Dataset:
    """A dataset ready for RaMiCo: features, labels and their description.

    Unpacks as ``X, y = dataset``.
    """

    name: str
    X: pd.DataFrame
    y: pd.Series
    target_names: dict[int, str]
    categorical_features: list[str]
    description: str
    url: str
    frame: pd.DataFrame = field(repr=False)

    def __iter__(self) -> Iterator[pd.DataFrame | pd.Series]:
        yield self.X
        yield self.y


def get_data_home(data_home: str | Path | None = None) -> Path:
    """Folder of the downloaded datasets (created if missing)."""
    path = Path(data_home or os.environ.get("RAMICO_DATA") or Path.home() / ".cache" / "ramico")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch(name: str, data_home: str | Path | None, download: bool) -> Path:
    url, checksum = _FILES[name]
    target = get_data_home(data_home) / Path(url).name
    if target.exists() and _sha256(target) == checksum:
        return target
    if not download:
        raise DataError(f"{target} is missing and download=False")
    with tempfile.NamedTemporaryFile(delete=False, dir=target.parent, suffix=".part") as partial:
        temporary = Path(partial.name)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(temporary, "wb") as stream:
            shutil.copyfileobj(response, stream)
        if _sha256(temporary) != checksum:
            raise DataError(f"the file downloaded from {url} does not match its expected checksum")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


# ---------------------------------------------------------------- Heart Disease

_HEART_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]
_HEART_CATEGORIES = {
    "sex": {0: "female", 1: "male"},
    "cp": {1: "typical angina", 2: "atypical angina", 3: "non-anginal pain", 4: "asymptomatic"},
    "fbs": {0: "no", 1: "yes"},
    "restecg": {0: "normal", 1: "ST-T abnormality", 2: "LV hypertrophy"},
    "exang": {0: "no", 1: "yes"},
    "slope": {1: "upsloping", 2: "flat", 3: "downsloping"},
    "thal": {3: "normal", 6: "fixed defect", 7: "reversible defect"},
}
_HEART_DESCRIPTION = (
    "Heart Disease (Cleveland), UCI. 297 patients after dropping the rows with missing "
    "values; 13 clinical features; target: any degree of heart disease (num > 0). "
    "Categorical: sex, cp (chest pain), fbs (fasting blood sugar > 120 mg/dl), restecg, "
    "exang (exercise-induced angina), slope, thal."
)


def parse_heart_disease(path: str | Path, readable: bool = True) -> Dataset:
    """Read ``processed.cleveland.data`` (see :func:`load_heart_disease`)."""
    frame = pd.read_csv(path, header=None, names=_HEART_COLUMNS, na_values="?").dropna()
    frame = frame.reset_index(drop=True)
    for column in ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "slope", "ca", "thal"]:
        frame[column] = frame[column].astype(int)
    for column, labels in _HEART_CATEGORIES.items():
        if readable:
            frame[column] = pd.Categorical(frame[column].map(labels), categories=list(labels.values()))
    y = (frame.pop("num") > 0).astype(int).rename("disease")
    return Dataset(
        name="heart_disease",
        X=frame,
        y=y,
        target_names={0: "no disease", 1: "disease"},
        categorical_features=list(_HEART_CATEGORIES),
        description=_HEART_DESCRIPTION,
        url=_FILES["heart_disease"][0],
        frame=frame.assign(disease=y),
    )


def load_heart_disease(*, readable: bool = True, data_home: str | Path | None = None, download: bool = True) -> Dataset:
    """Heart Disease (Cleveland) dataset of the UCI repository.

    Parameters
    ----------
    readable : bool, default=True
        Categorical features as pandas categories with descriptive labels
        (e.g. ``cp == "asymptomatic"``); with False they keep the integer
        codes of the UCI file and must be declared as categorical.
    data_home : str or Path, optional
        Cache folder (see :func:`get_data_home`).
    download : bool, default=True
        Download the file when it is not cached.

    Returns
    -------
    Dataset
        297 samples, 13 features, binary target (1 = heart disease).
    """
    return parse_heart_disease(_fetch("heart_disease", data_home, download), readable=readable)


# ------------------------------------------------------------------------ Adult

_ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]
_ADULT_CATEGORICAL = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
_ADULT_DESCRIPTION = (
    "Adult (Census Income), UCI, training file. 30,162 people after dropping the rows with "
    "missing values; 14 features (6 numerical, 8 categorical); target: income > 50K."
)


def parse_adult(path: str | Path) -> Dataset:
    """Read ``adult.data`` (see :func:`load_adult`)."""
    frame = pd.read_csv(path, header=None, names=_ADULT_COLUMNS, na_values="?", skipinitialspace=True)
    frame = frame.dropna().reset_index(drop=True)
    for column in _ADULT_CATEGORICAL:
        frame[column] = frame[column].astype("category")
    y = (frame.pop("income") == ">50K").astype(int).rename("income_above_50k")
    return Dataset(
        name="adult",
        X=frame,
        y=y,
        target_names={0: "<=50K", 1: ">50K"},
        categorical_features=list(_ADULT_CATEGORICAL),
        description=_ADULT_DESCRIPTION,
        url=_FILES["adult"][0],
        frame=frame.assign(income_above_50k=y),
    )


def load_adult(*, data_home: str | Path | None = None, download: bool = True) -> Dataset:
    """Adult (Census Income) dataset of the UCI repository (training file).

    Parameters
    ----------
    data_home : str or Path, optional
        Cache folder (see :func:`get_data_home`).
    download : bool, default=True
        Download the file when it is not cached.

    Returns
    -------
    Dataset
        30,162 samples, 14 features with categorical columns as pandas
        categories, binary target (1 = income above 50K).
    """
    return parse_adult(_fetch("adult", data_home, download))
