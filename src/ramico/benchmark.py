"""Repeated runs of the mining strategies and their statistical comparison.

The protocol follows the paper: every seed splits the data, trains the model
on the training part, extracts the rules of each strategy on the same
training data and evaluates fidelity, incomprehensibility and implausibility
there. :func:`compare_strategies` then tests every strategy against a
reference (the ensemble) with a two-sided Mann-Whitney U test per metric and
Benjamini-Hochberg correction within each metric.

Examples
--------
>>> from sklearn.linear_model import LogisticRegression          # doctest: +SKIP
>>> from ramico.benchmark import compare_strategies, paper_seeds, run_benchmark
>>> results = run_benchmark(lambda seed: LogisticRegression(), X, y,   # doctest: +SKIP
...                         seeds=paper_seeds(5), cache_dir="cache")
>>> compare_strategies(results)                                 # doctest: +SKIP
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import warnings
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ramico._explainer import RaMiCo
from ramico.exceptions import DataError, MissingDependencyError
from ramico.metrics import K, evaluate

__all__ = [
    "METRICS",
    "PAPER_STRATEGIES",
    "balanced_split",
    "benjamini_hochberg",
    "compare_strategies",
    "paper_seeds",
    "run_benchmark",
    "stratified_split",
]

PAPER_STRATEGIES = ("boundary", "prototype", "random", "ensemble")
#: Metrics and whether higher values are better.
METRICS = {"fidelity": "higher", "incomprehensibility": "lower", "implausibility": "lower"}
FIXED_SEEDS = (42, 123, 456, 789, 101112, 131415)

Splitter = Callable[[pd.DataFrame, np.ndarray, int], tuple[np.ndarray, np.ndarray]]


def paper_seeds(n: int = 30, generator_seed: int = 999) -> list[int]:
    """Seeds of the paper's protocol: six fixed seeds, then distinct draws in [1000, 10^6).

    ``paper_seeds()`` returns the 30 seeds of the paper's experiments; smaller
    ``n`` returns their prefix.
    """
    seeds = list(FIXED_SEEDS)
    rng = np.random.default_rng(seed=generator_seed)
    while len(seeds) < n:
        candidate = int(rng.integers(1000, 1000000))
        if candidate not in seeds:
            seeds.append(candidate)
    return seeds[:n]


def stratified_split(test_size: float = 0.2) -> Splitter:
    """Stratified train/test split (scikit-learn's ``train_test_split``), as in the paper."""

    def split(X: pd.DataFrame, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
        try:
            from sklearn.model_selection import train_test_split
        except ImportError as exc:
            raise MissingDependencyError(
                'stratified_split needs scikit-learn: pip install "ramico[tutorials]"'
            ) from exc
        train, test = train_test_split(np.arange(len(y)), test_size=test_size, random_state=seed, stratify=y)
        return np.asarray(train), np.asarray(test)

    return split


def balanced_split(fraction: float = 0.1) -> Splitter:
    """Test set with ``int(fraction * n)`` samples of each class, as for Adult in the paper."""

    def split(X: pd.DataFrame, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
        rs = np.random.RandomState(seed)
        per_class = int(len(y) * fraction)
        test = np.concatenate([rs.choice(np.where(y == c)[0], per_class, replace=False) for c in sorted(pd.unique(y))])
        return np.setdiff1d(np.arange(len(y)), test), test

    return split


def _progress(items: Sequence[Any], enabled: bool, description: str) -> Iterable[Any]:
    if not enabled:
        return items
    try:
        with warnings.catch_warnings():
            # without ipywidgets tqdm.auto falls back to a text bar, with a warning
            warnings.simplefilter("ignore")
            from tqdm.auto import tqdm
    except ImportError:
        return items
    # stdout: notebooks show stderr in red, as an error
    return tqdm(items, desc=description, file=sys.stdout)


def run_benchmark(
    make_model: Callable[[int], Any],
    X: pd.DataFrame,
    y: Any,
    *,
    strategies: Sequence[str] = PAPER_STRATEGIES,
    seeds: Sequence[int] | None = None,
    n_factuals: int = 100,
    n_counterfactuals: int = 50,
    split: Splitter | None = None,
    k: int = K,
    mode: str = "and",
    isolation_forest: str = "auto",
    categorical_features: Sequence[str] | None = None,
    cache_dir: str | Path | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Extract and evaluate the rules of every strategy over several seeds.

    Parameters
    ----------
    make_model : callable
        ``make_model(seed)`` returns an unfitted estimator; it is fitted on the
        training split of each seed.
    X, y : DataFrame and array-like
        The whole dataset.
    strategies : sequence of str, default=("boundary", "prototype", "random", "ensemble")
        Strategies to compare.
    seeds : sequence of int, optional
        One run per seed; the 30 seeds of the paper by default.
    n_factuals, n_counterfactuals : int, default=100, 50
        ``s`` and ``k`` of the paper.
    split : callable, optional
        ``split(X, y, seed) -> (train_indices, test_indices)``; a stratified
        80/20 split by default (:func:`stratified_split`).
    k, mode : int, str
        Settings of the fidelity and incomprehensibility (see
        :func:`ramico.metrics.evaluate`).
    isolation_forest : {"auto", "ydf", "native"}
        Engine of the implausibility scores.
    categorical_features : sequence of str, optional
        Passed to :meth:`ramico.RaMiCo.fit`.
    cache_dir : str or Path, optional
        Folder where each (seed, strategy) result is saved as it completes;
        a rerun reuses the results computed with the same settings, data,
        training rows and model parameters (``get_params``, or else the
        attributes of the unfitted model).
    progress : bool, default=True
        Show a progress bar (needs tqdm).

    Returns
    -------
    pandas.DataFrame
        One row per (seed, strategy): metrics, numbers of rules and records,
        seconds spent extracting and evaluating, Isolation Forest engine,
        ``n_factuals`` and ``n_counterfactuals``.
    """
    seeds = list(seeds) if seeds is not None else paper_seeds()
    split = split or stratified_split()
    labels = np.asarray(y)
    if len(labels) != len(X):
        raise DataError(f"X has {len(X)} rows but y has {len(labels)} labels")
    settings = {
        "n_factuals": n_factuals,
        "n_counterfactuals": n_counterfactuals,
        "k": k,
        "mode": mode,
        "isolation_forest": isolation_forest,
        "n_samples": len(X),
        "data": _digest(pd.util.hash_pandas_object(X, index=True).to_numpy(), _label_hashes(labels)),
        "categorical_features": None if categorical_features is None else [str(c) for c in categorical_features],
    }
    cache = Path(cache_dir) if cache_dir is not None else None
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in _progress(seeds, progress, "seeds"):
        train, _ = split(X, labels, seed)
        X_train, y_train = X.iloc[train], labels[train]
        model, fitted = make_model(seed), False
        # a cached result is reused only for the same settings, data, training rows and model parameters
        run_settings = {**settings, "train": _digest(np.asarray(train, dtype=np.int64)), "model": _model_key(model)}
        for strategy in strategies:
            path = cache / f"{strategy}_seed{seed}.json" if cache is not None else None
            if path is not None and path.exists():
                saved = json.loads(path.read_text(encoding="utf-8"))
                if saved.get("settings") == run_settings:
                    rows.append(saved["result"])
                    continue
            if not fitted:
                model.fit(X_train, y_train)
                fitted = True
            start = time.perf_counter()
            rules = RaMiCo(
                model,
                strategy=strategy,
                n_factuals=n_factuals,
                n_counterfactuals=n_counterfactuals,
                random_state=seed,
                isolation_forest=isolation_forest,
            ).fit(X_train, y_train, categorical_features=categorical_features)
            extracted = time.perf_counter()
            scores = evaluate(model, rules, X_train, k=k, mode=mode)
            result = {
                "seed": int(seed),
                "strategy": strategy,
                **{name: float(value) for name, value in scores.items()},
                "n_rules": len(rules),
                "n_records": len(rules.records),
                "extraction_seconds": extracted - start,
                "evaluation_seconds": time.perf_counter() - extracted,
                "isolation_forest": rules.metadata["isolation_forest"],
            }
            rows.append(result)
            if path is not None:
                path.write_text(json.dumps({"settings": run_settings, "result": result}), encoding="utf-8")
    # the settings of the run, so that runs with different s and k can be concatenated
    return pd.DataFrame(rows).assign(n_factuals=n_factuals, n_counterfactuals=n_counterfactuals)


def _digest(*arrays: np.ndarray) -> str:
    """Short SHA-256 of the bytes of ``arrays``."""
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()[:16]


def _label_hashes(labels: np.ndarray) -> np.ndarray:
    return pd.util.hash_pandas_object(pd.Series(labels, dtype=object), index=False).to_numpy()


def _model_key(model: Any) -> str:
    """The class and settings of an unfitted model, as a short digest.

    Estimators with ``get_params`` (scikit-learn, XGBoost...) are described by
    their parameters, other objects by their attributes. A value whose
    ``repr`` holds a memory address changes the digest at every run, so that
    such results are recomputed rather than reused by mistake.
    """
    params: Any = None
    get_params = getattr(model, "get_params", None)
    if callable(get_params):
        try:
            params = get_params(deep=True)
        except Exception:  # any failure: describe the model by its attributes
            params = None
    if not isinstance(params, dict):
        params = vars(model) if hasattr(model, "__dict__") else {"repr": model}
    described = {
        str(name): type(value).__qualname__ if hasattr(value, "get_params") else repr(value)
        for name, value in params.items()
    }
    text = f"{type(model).__module__}.{type(model).__qualname__} {json.dumps(described, sort_keys=True)}"
    return _digest(np.frombuffer(text.encode("utf-8"), dtype=np.uint8))


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (same as statsmodels' ``fdr_bh``)."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adjusted = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1], 1.0)
    out = np.empty(n)
    out[order] = adjusted
    return out


def compare_strategies(
    results: pd.DataFrame,
    reference: str = "ensemble",
    metrics: Sequence[str] | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Test every strategy against ``reference`` on each metric.

    Two-sided Mann-Whitney U test on the per-seed values (unpaired, as in the
    paper), Benjamini-Hochberg correction within each metric.

    Returns
    -------
    pandas.DataFrame
        One row per (metric, strategy): means and standard deviations,
        raw and adjusted p-values, ``significant`` (adjusted p < alpha) and
        ``better`` (the strategy's mean beats the reference in the metric's
        direction: higher fidelity, lower incomprehensibility and
        implausibility).
    """
    try:
        from scipy.stats import mannwhitneyu
    except ImportError as exc:
        raise MissingDependencyError('compare_strategies needs scipy: pip install "ramico[benchmark]"') from exc
    metrics = list(metrics) if metrics is not None else list(METRICS)
    if reference not in set(results["strategy"]):
        raise DataError(f"no results for the reference strategy {reference!r}")
    rows = []
    for metric in metrics:
        base = results.loc[results["strategy"] == reference, metric].dropna()
        family = []
        for strategy in [s for s in pd.unique(results["strategy"]) if s != reference]:
            values = results.loc[results["strategy"] == strategy, metric].dropna()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)  # empty samples: handled below
                    p_value = float(mannwhitneyu(base, values, alternative="two-sided").pvalue)
            except ValueError:
                p_value = float("nan")
            if not np.isfinite(p_value):
                p_value = 1.0  # not enough data, or all the values tied: no evidence of a difference
            higher_is_better = METRICS.get(metric, "higher") == "higher"
            difference = values.mean() - base.mean()
            family.append(
                {
                    "metric": metric,
                    "strategy": strategy,
                    "mean": values.mean(),
                    "std": values.std(),
                    "reference_mean": base.mean(),
                    "reference_std": base.std(),
                    "p_value": p_value,
                    "better": bool(difference > 0 if higher_is_better else difference < 0),
                }
            )
        adjusted = benjamini_hochberg([row["p_value"] for row in family])
        for row, p_adjusted in zip(family, adjusted, strict=True):
            row["p_adjusted"] = float(p_adjusted)
            row["significant"] = bool(p_adjusted < alpha)
        rows.extend(family)
    columns = [
        "metric",
        "strategy",
        "mean",
        "std",
        "reference_mean",
        "reference_std",
        "p_value",
        "p_adjusted",
        "significant",
        "better",
    ]
    return pd.DataFrame(rows, columns=columns)
