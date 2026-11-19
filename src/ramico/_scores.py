"""Metrics of a set of rules, read by name: ``scores.fidelity`` or ``scores["fidelity"]``."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any


class Scores(Mapping[str, Any]):
    """Metrics of a set of rules, each read as an attribute or as a key.

    :func:`ramico.metrics.evaluate` returns the three metrics of the paper,
    and :meth:`ramico.RuleSet.load` the metrics saved with the rules. Scores
    are read-only and behave as a dict: ``dict(scores)``, ``scores.items()``,
    ``pandas.Series(scores)``, ``scores == {...}``.

    Examples
    --------
    >>> scores = Scores(fidelity=93.7, incomprehensibility=3.2, implausibility=0.41)
    >>> scores.fidelity
    93.7
    >>> scores["implausibility"]
    0.41
    >>> scores
    Scores(fidelity=93.7, incomprehensibility=3.2, implausibility=0.41)
    """

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, Any] | None = None, /, **metrics: Any) -> None:
        object.__setattr__(self, "_values", {**(values or {}), **metrics})

    def __getitem__(self, name: str) -> Any:
        return self._values[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):  # also the probes of copy and pickle, before _values exists
            raise AttributeError(name)
        try:
            return self._values[name]
        except KeyError:
            available = ", ".join(map(str, self._values)) or "none"
            raise AttributeError(f"no metric named {name!r} (available: {available})") from None

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Scores are read-only")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Scores are read-only")

    def __reduce__(self) -> tuple:
        return (Scores, (dict(self._values),))

    def __dir__(self) -> list[str]:
        # the metrics show up in the completions of notebooks and editors
        return sorted({*super().__dir__(), *(name for name in self._values if str(name).isidentifier())})

    def __repr__(self) -> str:
        if all(str(name).isidentifier() for name in self._values):
            return "Scores(" + ", ".join(f"{name}={value!r}" for name, value in self._values.items()) + ")"
        return f"Scores({self._values!r})"
