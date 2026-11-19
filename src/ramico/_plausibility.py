"""Isolation Forest that scores how implausible a counterfactual record is.

Two engines implement the same algorithm:

``"ydf"``
    YDF's ``IsolationForestLearner(num_trees=1000, max_depth=-1)``, the
    configuration of the paper. Optional dependency (``ramico[plausibility]``),
    not deterministic across trainings.
``"native"``
    :class:`ramico._isolation_forest.IsolationForest`, a numpy reimplementation
    of the same algorithm, always available and deterministic for a given
    random state.

``"auto"`` uses YDF when it is installed and the native engine otherwise.
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import os
import sys
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

from ramico._isolation_forest import IsolationForest
from ramico.exceptions import MissingDependencyError

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def native_output_silenced() -> Iterator[None]:
    """Send to the null device what native code writes on the standard output and error.

    Python-level output (``print``, warnings, logging, a notebook's streams)
    is not affected: it does not go through the file descriptors 1 and 2 in
    a notebook, and is flushed before they are redirected. Other threads
    writing to those descriptors meanwhile are silenced too.
    """
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.flush()
    saved: list[tuple[int, int]] = []
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        for fd in (1, 2):
            with contextlib.suppress(OSError):  # a descriptor may be closed, as under some services
                copy = os.dup(fd)
                os.dup2(null, fd)
                saved.append((fd, copy))
        yield
    finally:
        for fd, copy in saved:
            os.dup2(copy, fd)
            os.close(copy)
        os.close(null)


# Configuration of the paper: YDF Isolation Forest, unconstrained tree depth.
YDF_PARAMS = {"num_trees": 1000, "max_depth": -1}
#: Examples drawn for each tree, by YDF and by the native engine.
SUBSAMPLE = 256
ENGINES = ("auto", "ydf", "native")


def _import_ydf() -> ModuleType:
    try:
        return importlib.import_module("ydf")
    except ImportError as exc:
        raise MissingDependencyError(
            "The YDF Isolation Forest needs YDF (Yggdrasil Decision Forests). "
            'Install it with: pip install "ramico[plausibility]", '
            'or use isolation_forest="native".'
        ) from exc


def ydf_available() -> bool:
    """Whether YDF can be imported."""
    try:
        _import_ydf()
    except ImportError:
        return False
    return True


@dataclass(frozen=True)
class AnomalyScorer:
    """A trained Isolation Forest and the name of the engine that trained it."""

    model: Any
    engine: str

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Anomaly score of every row of ``data`` (higher is more anomalous)."""
        return np.asarray(self.model.predict(data))


def fit_isolation_forest(data: pd.DataFrame, engine: str = "auto", random_state: int | None = None) -> AnomalyScorer:
    """Train the Isolation Forest of the paper on ``data``.

    Parameters
    ----------
    data : pandas.DataFrame
        Training data, labels included as in the original implementation.
    engine : {"auto", "ydf", "native"}, default="auto"
        Implementation to use; "auto" prefers YDF when it is installed.
    random_state : int, optional
        Seed of the native engine (YDF uses its own fixed seed).

    Returns
    -------
    AnomalyScorer
        The trained forest; ``engine`` names the implementation and, for YDF,
        its version.
    """
    if engine not in ENGINES:
        raise ValueError(f"isolation_forest must be one of {ENGINES}, got {engine!r}")
    if engine == "auto":
        engine = "ydf" if ydf_available() else "native"
    if engine == "ydf":
        ydf = _import_ydf()
        if len(data) < SUBSAMPLE:
            warnings.warn(
                f"YDF's Isolation Forest draws {SUBSAMPLE} examples per tree but the data have {len(data)}: "
                "YDF 0.8.0 at least fills the sample with copies of the first row, which biases the "
                'implausibility; isolation_forest="native" handles small data.',
                RuntimeWarning,
                stacklevel=3,
            )
        with native_output_silenced():  # YDF 0.8.0 logs from C++ even with verbose=0
            model = ydf.IsolationForestLearner(**YDF_PARAMS).train(data, verbose=0)
        version = getattr(ydf, "__version__", None)
        return AnomalyScorer(model, f"ydf {version}" if version else "ydf")
    logger.info("Training the native Isolation Forest (1000 trees)")
    forest = IsolationForest(
        n_trees=YDF_PARAMS["num_trees"],
        subsample=SUBSAMPLE,
        max_depth=None,
        min_examples=5,
        random_state=random_state,
    )
    return AnomalyScorer(forest.fit(data), "native")
