"""Errors raised by RaMiCo."""

from __future__ import annotations

__all__ = ["DataError", "MissingDependencyError", "ModelError", "RaMiCoError"]


class RaMiCoError(Exception):
    """Base class of the errors raised by RaMiCo."""


class DataError(RaMiCoError, ValueError):
    """The data passed to RaMiCo cannot be used (shape, labels, missing values...)."""


class ModelError(RaMiCoError, TypeError):
    """The model cannot be used as a black box: no ``predict`` or wrong output."""


class MissingDependencyError(RaMiCoError, ImportError):
    """An optional dependency is needed; the message names the extra to install."""
