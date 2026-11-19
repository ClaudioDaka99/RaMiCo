"""JSON files of RuleSets: tables and values that come back with their types.

Plain JSON keeps strings, integers, floats, booleans and null apart; what it
cannot write is tagged and restored on reading:

- tables are stored by column, with the pandas dtype of each column;
- in float columns, NaN is written as null and infinities as strings;
- elsewhere, non-finite floats, timestamps (row labels of a DatetimeIndex),
  tuples and mappings whose keys are not strings (class names keyed by
  integer labels) become small tagged objects.

The files are standard JSON (no ``NaN`` literals), compressed with gzip when
their name ends in ``.gz``.
"""

from __future__ import annotations

import contextlib
import datetime
import gzip
import json
import math
import os
import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import IO, Any

import numpy as np
import pandas as pd

_FLOAT = "__float__"
_TUPLE = "__tuple__"
_MAPPING = "__mapping__"
_TIMESTAMP = "__timestamp__"


def encode_value(value: Any) -> Any:
    """``value`` as JSON-ready Python objects; :func:`decode_value` inverts it."""
    if isinstance(value, np.datetime64):
        value = pd.Timestamp(value)  # .item() would give nanoseconds as an int
    elif isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if value is pd.NaT:
        return None
    if isinstance(value, float):
        return value if math.isfinite(value) else {_FLOAT: repr(value)}
    if isinstance(value, (datetime.date, pd.Timestamp)):
        return {_TIMESTAMP: pd.Timestamp(value).isoformat()}
    if isinstance(value, tuple):
        return {_TUPLE: [encode_value(v) for v in value]}
    if isinstance(value, list):
        return [encode_value(v) for v in value]
    if isinstance(value, Mapping):
        if all(isinstance(k, str) and not k.startswith("__") for k in value):
            return {k: encode_value(v) for k, v in value.items()}
        return {_MAPPING: [[encode_value(k), encode_value(v)] for k, v in value.items()]}
    raise TypeError(f"cannot write a value of type {type(value).__name__} to JSON: {value!r}")


def decode_value(value: Any) -> Any:
    """Inverse of :func:`encode_value`."""
    if isinstance(value, list):
        return [decode_value(v) for v in value]
    if isinstance(value, dict):
        if len(value) == 1:
            tag, content = next(iter(value.items()))
            if tag == _FLOAT:
                return float(content)
            if tag == _TIMESTAMP:
                return pd.Timestamp(content)
            if tag == _TUPLE:
                return tuple(decode_value(v) for v in content)
            if tag == _MAPPING:
                return {_hashable(decode_value(k)): decode_value(v) for k, v in content}
        return {k: decode_value(v) for k, v in value.items()}
    return value


def _hashable(key: Any) -> Any:
    return tuple(_hashable(k) for k in key) if isinstance(key, (list, tuple)) else key


def _encode_column(column: pd.Series) -> list:
    kind = column.dtype.kind
    if kind in "biu":
        values = column.tolist()
        return [None if v is pd.NA else v for v in values] if column.hasnans else values  # nullable dtypes
    if kind == "f":
        return [v if math.isfinite(v) else (None if math.isnan(v) else repr(v)) for v in column.tolist()]
    # object, strings, categories: one value at a time; missing values become null
    return [None if _is_nan(v) else encode_value(v) for v in column.astype(object).tolist()]


def _is_nan(value: Any) -> bool:
    return isinstance(value, (float, np.floating)) and math.isnan(value)


def _decode_column(values: list, dtype: str) -> pd.Series:
    if dtype in ("bool", "boolean") or dtype.startswith(("int", "uint", "Int", "UInt")):
        return pd.Series(values, dtype=dtype)
    if dtype.startswith(("float", "Float")):
        return pd.Series([math.nan if v is None else float(v) for v in values], dtype=dtype)
    column = pd.Series([math.nan if v is None else decode_value(v) for v in values], dtype=object)
    if dtype != "object":
        with contextlib.suppress(TypeError, ValueError):  # a dtype this pandas does not know: keep the objects
            column = column.astype(dtype)
    return column


def encode_table(frame: pd.DataFrame) -> dict[str, Any]:
    """A DataFrame as ``{"columns", "dtypes", "data"}``, ``data`` holding one list per column."""
    columns = [str(c) for c in frame.columns]
    if len(set(columns)) != len(columns):
        raise ValueError(f"cannot write a table with repeated column names: {columns}")
    return {
        "columns": columns,
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "data": {name: _encode_column(frame.iloc[:, i]) for i, name in enumerate(columns)},
    }


def decode_table(table: Mapping[str, Any]) -> pd.DataFrame:
    """Inverse of :func:`encode_table`."""
    columns, dtypes, data = table["columns"], table["dtypes"], table["data"]
    if len(columns) != len(dtypes):
        raise ValueError(f"{len(columns)} columns but {len(dtypes)} dtypes")
    return pd.DataFrame({name: _decode_column(data[name], dtype) for name, dtype in zip(columns, dtypes, strict=True)})


def _open(path: Path, mode: str) -> IO[str]:
    encoding = "utf-8-sig" if mode == "r" else "utf-8"  # reading also accepts a file re-saved with a BOM
    if path.suffix.lower() == ".gz":
        return gzip.open(path, mode + "t", encoding=encoding)  # type: ignore[return-value]  # text mode
    return open(path, mode, encoding=encoding)


def check_path(path: str | os.PathLike[str]) -> Path:
    """``path`` as a Path, if its name ends in ``.json`` or ``.json.gz``."""
    path = Path(path)
    if not path.name.lower().endswith((".json", ".json.gz")):
        raise ValueError(
            f"RuleSet files are JSON: name the file *.json, or *.json.gz to compress it (got {path.name!r}). "
            "For CSV tables, use rules.to_frame().to_csv(...) and rules.records.to_csv(...)"
        )
    return path


#: Errors of a file that is not JSON: wrong text, truncated or damaged gzip.
UNREADABLE = (ValueError, EOFError, gzip.BadGzipFile, zlib.error)


def write_json(payload: Mapping[str, Any], path: Path) -> None:
    with _open(path, "w") as stream:
        json.dump(payload, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def read_json(path: Path) -> Any:
    with _open(path, "r") as stream:
        return json.load(stream)
