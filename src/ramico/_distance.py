"""Gower distance between the rows of mixed-type DataFrames."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def _with_numpy_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace pandas extension dtypes, which the gower package cannot read.

    Numeric extension columns (nullable integers and floats) become float64;
    the others (``str``, ``category``, nullable booleans) become object, the
    dtype pandas 2 gives to text and that gower treats as categorical.
    """
    converted = {}
    for column in frame.columns:
        dtype = frame[column].dtype
        if isinstance(dtype, np.dtype):
            continue
        if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
            converted[column] = frame[column].to_numpy(dtype=np.float64, na_value=np.nan)
        else:
            converted[column] = frame[column].astype(object)
    return frame.assign(**converted) if converted else frame


def gower_matrix(x: pd.DataFrame, y: pd.DataFrame | None = None) -> np.ndarray:
    """Gower distances between the rows of ``x`` and ``y`` (``x`` itself by default).

    The distances of :func:`gower.gower_matrix`: numerical columns are
    range-normalised over the rows of ``x`` and ``y`` together, the other
    columns are compared by equality. Returns float32, like gower. On mixed
    data the matrix is computed here, with gower's arithmetic but in blocks
    of rows instead of one row at a time: the result is the same bit for bit,
    in a fraction of the time (:func:`_mixed_distances`).

    gower 0.1.2 takes two frames with as many rows for one frame against
    itself: it computes the upper triangle and mirrors it, so half of the
    distances are wrong. For frames of equal length the
    distances are computed on their union, whose matrix is symmetric indeed,
    and its block of ``x`` against ``y`` returned: the ranges are the same,
    and the result is the one gower gives when the lengths differ.
    """
    x = _with_numpy_dtypes(x)
    y = x if y is None else _with_numpy_dtypes(y)
    fast = _gower_fast(x, y)
    if fast is not None:
        return fast

    import gower  # imported on first use: it pulls in scipy

    if y is x:
        return gower.gower_matrix(x)
    if len(x) == len(y):
        both = pd.concat([x, y], ignore_index=True)
        return gower.gower_matrix(both)[: len(x), len(x) :]
    return gower.gower_matrix(x, y)


#: dtype of ``1 - np.float32(x)``: float32 under NEP 50 (numpy >= 2), float64 before
_ONE_MINUS_FLOAT32 = np.result_type(1 - np.float32(0.5))


def _row_kind(column: pd.Series) -> str | None:
    """How gower sees ``column`` in the one-row frame of one of its values.

    ``pd.DataFrame([row])`` infers the dtype of each value: numbers stored in
    an object column become numerical. None when the rows would disagree.
    """
    if is_numpy_number(column.dtype):
        return "numerical"
    if column.dtype != object:
        return "categorical"  # booleans and the like
    inferred = pd.api.types.infer_dtype(column, skipna=False)
    if inferred in ("integer", "floating", "mixed-integer-float"):
        return "numerical"
    if inferred in ("string", "boolean"):
        return "categorical"
    return None


def _codes(
    x: pd.DataFrame, y: pd.DataFrame, categorical: list[str]
) -> tuple[list[np.ndarray], list[np.ndarray]] | None:
    """Integer codes of the categorical values, equal where the values are (None with missing values)."""
    codes_x, codes_y = [], []
    for column in categorical:
        values = np.concatenate([x[column].to_numpy(object), y[column].to_numpy(object)])
        codes, _ = pd.factorize(values, use_na_sentinel=True)
        if (codes < 0).any():
            return None  # Python's == on missing values is not an equality of codes
        codes_x.append(codes[: len(x)])
        codes_y.append(codes[len(x) :])
    return codes_x, codes_y


def _scales(low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """gower's maxima and ranges from the float32 minima and maxima of the columns."""
    high = np.where(np.isnan(high), np.float32(0), high)
    low = np.where(np.isnan(low), np.float32(0), low)
    with np.errstate(invalid="ignore", divide="ignore"):
        # gower computes 1 - min / max on float32 scalars: float32 with numpy 2, float64 before (NEP 50)
        ratio = (low / high).astype(_ONE_MINUS_FLOAT32)
        ranges = np.where(high != 0, np.abs(np.asarray(1, dtype=_ONE_MINUS_FLOAT32) - ratio), 0)
    return high.astype(np.float64), ranges.astype(np.float64)


def _mixed_distances(
    x_values: np.ndarray,
    codes_x: list[np.ndarray],
    y_values: np.ndarray,
    codes_y: list[np.ndarray],
    maxima: np.ndarray,
    ranges: np.ndarray,
    n_features: int,
    max_cells: int,
) -> np.ndarray:
    """gower's distances when its arrays have the object dtype, in blocks of rows.

    ``maxima`` and ``ranges`` have one row per row of ``x``, or one row for
    all. Python floats are IEEE doubles, so the same operations in the same
    order give the same bits: scaled values and their differences in
    float64, the numerical terms added one column after the other to the
    number of categorical mismatches, divided by the number of features and
    rounded to float32.
    """
    n_x, n_y = len(x_values), len(y_values)
    shared = maxima.shape[0] == 1  # one scale for all the rows: scale y once
    maxima = np.broadcast_to(maxima, (n_x, maxima.shape[1]))
    ranges = np.broadcast_to(ranges, (n_x, ranges.shape[1]))
    scaled = maxima != 0
    with np.errstate(invalid="ignore", divide="ignore"):
        x_scaled = np.where(scaled, x_values / np.where(scaled, maxima, 1.0), 0.0)
        if shared:
            y_scaled_all = np.where(scaled[:1], y_values / np.where(scaled[:1], maxima[:1], 1.0), 0.0)
    out = np.empty((n_x, n_y), dtype=np.float32)
    block = max(1, max_cells // max(n_y, 1))
    for start in range(0, n_x, block):
        rows = slice(start, min(start + block, n_x))
        mismatches = np.zeros((rows.stop - rows.start, n_y), dtype=np.float64)
        for codes_xi, codes_yi in zip(codes_x, codes_y, strict=True):
            mismatches += codes_xi[rows, None] != codes_yi[None, :]
        total = None
        for f in range(x_values.shape[1]):
            with np.errstate(invalid="ignore", divide="ignore"):
                if shared:
                    y_scaled = y_scaled_all[None, :, f]
                else:
                    on = scaled[rows, f][:, None]
                    y_scaled = np.where(on, y_values[None, :, f] / np.where(on, maxima[rows, f][:, None], 1.0), 0.0)
                term = np.abs(x_scaled[rows, f][:, None] - y_scaled)
                if shared:
                    span = ranges[0, f]
                    term = term / span if span != 0 else np.zeros_like(term)
                else:
                    span = ranges[rows, f][:, None]
                    term = np.where(span != 0, term / np.where(span != 0, span, 1.0), 0.0)
            total = term if total is None else total + term
        sums = mismatches if total is None else mismatches + total
        out[rows] = sums / float(n_features)
    return out


def _gower_fast(x: pd.DataFrame, y: pd.DataFrame, max_cells: int = 2_000_000) -> np.ndarray | None:
    """``gower.gower_matrix(x, y)`` on mixed data, computed in blocks; None in the other cases."""
    if list(x.columns) != list(y.columns) or len(x) == 0 or len(y) == 0:
        return None
    categorical = [c for c in x.columns if not is_numpy_number(x[c].dtype)]  # gower reads the dtypes of x
    numerical = [c for c in x.columns if c not in categorical]
    if np.asarray(x.iloc[:1]).dtype != object and np.asarray(y.iloc[:1]).dtype != object:
        return None  # numerical arrays: gower sums the terms pairwise, not one after the other
    codes = _codes(x, y, categorical)
    if codes is None:
        return None
    try:
        x_values = x[numerical].to_numpy(dtype=np.float64)
        y_values = y[numerical].to_numpy(dtype=np.float64)
    except (TypeError, ValueError):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
        both = np.concatenate([x_values, y_values]).astype(np.float32)
        low, high = np.nanmin(both, axis=0, keepdims=True), np.nanmax(both, axis=0, keepdims=True)
    maxima, ranges = _scales(low, high)
    return _mixed_distances(x_values, codes[0], y_values, codes[1], maxima, ranges, len(x.columns), max_cells)


def gower_rows(samples: pd.DataFrame, candidates: pd.DataFrame, max_cells: int = 2_000_000) -> np.ndarray | None:
    """Distances of each sample, on its own, to all the candidates.

    Row ``i`` equals ``gower_matrix(pd.DataFrame([samples.iloc[i]]), candidates)[0]``
    bit for bit: each sample is range-normalised together with the candidates
    only, as when the samples are compared one at a time, but the rows are
    computed in blocks of about ``max_cells`` distances. The kind of each
    column is the one gower sees in a one-row frame, whose dtypes pandas
    infers from the values. Returns None when gower's arrays would not have
    the object dtype or the values do not allow it (missing categories,
    columns mixing numbers and text): the caller then compares one row at a
    time.
    """
    samples = _with_numpy_dtypes(samples)
    candidates = _with_numpy_dtypes(candidates[list(samples.columns)])
    kinds = {column: _row_kind(samples[column]) for column in samples.columns}
    if None in kinds.values() or len(candidates) <= 1 or len(samples) == 0:
        return None
    numerical = [c for c in samples.columns if kinds[c] == "numerical"]
    categorical = [c for c in samples.columns if kinds[c] == "categorical"]
    if not categorical and np.asarray(candidates.iloc[:1]).dtype != object:
        return None  # numerical arrays: gower sums the terms pairwise, not one after the other
    codes = _codes(samples, candidates, categorical)
    if codes is None:
        return None
    try:
        x_values = samples[numerical].to_numpy(dtype=np.float64)
        c_values = candidates[numerical].to_numpy(dtype=np.float64)
    except (TypeError, ValueError):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
        c32 = c_values.astype(np.float32)
        c_low, c_high = np.nanmin(c32, axis=0), np.nanmax(c32, axis=0)
    x32 = x_values.astype(np.float32)
    maxima, ranges = _scales(np.fmin(x32, c_low), np.fmax(x32, c_high))  # nanmin and nanmax with the sample
    return _mixed_distances(x_values, codes[0], c_values, codes[1], maxima, ranges, len(samples.columns), max_cells)


def is_numpy_number(dtype: object) -> bool:
    """``np.issubdtype(dtype, np.number)``, False for pandas extension dtypes."""
    return isinstance(dtype, np.dtype) and np.issubdtype(dtype, np.number)
