"""Mining strategies: how RaMiCo picks factual and counterfactual samples.

For a class ``c``, a strategy receives the samples of that class (``X_c``)
and all the others (``X_not_c``) and returns *pairs*: each pair is one
factual of ``X_c`` with the counterfactuals of ``X_not_c`` it will be
compared with. RaMiCo then switches, one at a time, the features in which a
factual and a counterfactual differ.

The strategies of the paper are registered under these names:

``"boundary"``
    Decision-Boundary mining (:class:`BoundaryStrategy`).
``"prototype"``
    Prototype mining (:class:`PrototypeStrategy`).
``"random"``
    Random mining (:class:`RandomStrategy`).
``"ensemble"``
    The reference configuration of the original RaMiCo
    (:class:`EnsembleStrategy`).

New strategies subclass :class:`MiningStrategy` and can be registered with
:func:`register_strategy`.

Examples
--------
>>> from ramico.strategies import available_strategies, get_strategy
>>> available_strategies()
['boundary', 'ensemble', 'prototype', 'random']
>>> get_strategy("prototype")
PrototypeStrategy()
"""

from __future__ import annotations

import abc
import logging
from collections.abc import Callable, Sequence
from typing import NamedTuple, TypeAlias

import numpy as np
import pandas as pd

from ramico._kmedoids import kmedoids

__all__ = [
    "BoundaryStrategy",
    "EnsembleStrategy",
    "MiningStrategy",
    "NearestBoundaryStrategy",
    "Pair",
    "PrototypeStrategy",
    "RandomPairsStrategy",
    "RandomStrategy",
    "available_strategies",
    "get_strategy",
    "register_strategy",
]

logger = logging.getLogger(__name__)

#: ``distance(x, y=None)`` returns the matrix of distances between the rows
#: of ``x`` and the rows of ``y`` (of ``x`` itself when ``y`` is None).
Distance = Callable[..., np.ndarray]
Generator: TypeAlias = np.random.RandomState | np.random.Generator


class Pair(NamedTuple):
    """A factual and the counterfactuals it is compared with.

    Attributes
    ----------
    factual : int
        Row position of the factual in ``X_c``.
    counterfactuals : numpy.ndarray
        Row positions of the counterfactuals in ``X_not_c``.
    """

    factual: int
    counterfactuals: np.ndarray


class MiningStrategy(abc.ABC):
    """Base class of the mining strategies.

    Subclasses implement :meth:`select`. Composite strategies, such as the
    ensemble, also override :meth:`components`.
    """

    #: Name used in the ``strategy`` column of the rules.
    name: str = "custom"

    @abc.abstractmethod
    def select(
        self,
        X_c: pd.DataFrame,
        X_not_c: pd.DataFrame,
        s: int,
        k: int,
        distance: Distance,
        rng: Generator,
    ) -> list[Pair]:
        """Choose the factual/counterfactual pairs for one class.

        Parameters
        ----------
        X_c : pandas.DataFrame
            Samples of the class being explained.
        X_not_c : pandas.DataFrame
            Samples of all the other classes.
        s : int
            Number of factuals to select.
        k : int
            Number of counterfactuals per factual.
        distance : callable
            ``distance(x, y=None)``, the distance matrix between the rows of
            ``x`` and ``y`` (Gower by default).
        rng : numpy.random.RandomState or numpy.random.Generator
            Source of randomness; deterministic strategies ignore it.

        Returns
        -------
        list of Pair
            One pair per selected factual, in selection order.
        """

    def components(self) -> list[tuple[str, MiningStrategy]]:
        """Strategies to run, each with the label of the rules it produces."""
        return [(self.name, self)]

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


def _nearest(distances: np.ndarray, n: int) -> np.ndarray:
    """Positions of the ``n`` smallest distances, ties broken by position."""
    return np.argsort(distances, kind="stable")[:n]


def _nearest_counterfactuals(
    X_not_c: pd.DataFrame,
    factuals: pd.DataFrame,
    positions: np.ndarray,
    k: int,
    distance: Distance,
) -> list[Pair]:
    distances = distance(X_not_c, factuals)
    return [Pair(int(position), _nearest(distances[:, i], k)) for i, position in enumerate(positions)]


class RandomStrategy(MiningStrategy):
    """Random mining.

    Factuals are drawn uniformly at random from ``X_c``; their
    counterfactuals are their ``k`` nearest neighbours in ``X_not_c``.
    """

    name = "random"

    def select(
        self, X_c: pd.DataFrame, X_not_c: pd.DataFrame, s: int, k: int, distance: Distance, rng: Generator
    ) -> list[Pair]:
        s = min(s, len(X_c))
        positions = rng.choice(len(X_c), s, replace=False)
        return _nearest_counterfactuals(X_not_c, X_c.iloc[positions], positions, k, distance)


class BoundaryStrategy(MiningStrategy):
    """Decision-Boundary mining.

    Takes the ``n_candidates`` samples of ``X_c`` closest to ``X_not_c``
    (smallest distance to their nearest sample of another class), draws
    ``s`` factuals at random among them and pairs each one with its ``k``
    nearest neighbours in ``X_not_c``.

    Parameters
    ----------
    n_candidates : int, optional
        Size of the candidate pool; by default ``int(1.5 * s)``, as in the
        original implementation.
    """

    name = "boundary"

    def __init__(self, n_candidates: int | None = None) -> None:
        self.n_candidates = n_candidates

    def select(
        self, X_c: pd.DataFrame, X_not_c: pd.DataFrame, s: int, k: int, distance: Distance, rng: Generator
    ) -> list[Pair]:
        n_candidates = self.n_candidates if self.n_candidates is not None else int(s * 1.5)
        closest = _nearest(distance(X_c, X_not_c).min(axis=1), n_candidates)
        if len(closest) < s:
            logger.warning("Only %d candidate factuals: using all of them", len(closest))
            positions = closest
        else:
            positions = rng.choice(closest, s, replace=False)
        return _nearest_counterfactuals(X_not_c, X_c.iloc[positions], positions, k, distance)

    def __repr__(self) -> str:
        if self.n_candidates is None:
            return "BoundaryStrategy()"
        return f"BoundaryStrategy(n_candidates={self.n_candidates})"


class NearestBoundaryStrategy(MiningStrategy):
    """Deterministic variant of Decision-Boundary mining.

    Factuals are the ``s`` samples of ``X_c`` closest to ``X_not_c``;
    counterfactuals are their ``k`` nearest neighbours in ``X_not_c``.
    It is one of the components of :class:`EnsembleStrategy`.
    """

    name = "nearest-boundary"

    def select(
        self, X_c: pd.DataFrame, X_not_c: pd.DataFrame, s: int, k: int, distance: Distance, rng: Generator
    ) -> list[Pair]:
        positions = _nearest(distance(X_c, X_not_c).min(axis=1), s)
        return _nearest_counterfactuals(X_not_c, X_c.iloc[positions], positions, k, distance)


class PrototypeStrategy(MiningStrategy):
    """Prototype mining.

    Factuals are ``s`` medoids of ``X_c`` and counterfactuals ``k`` medoids
    of ``X_not_c``; every factual is paired with the same counterfactual
    medoids.
    """

    name = "prototype"

    def select(
        self, X_c: pd.DataFrame, X_not_c: pd.DataFrame, s: int, k: int, distance: Distance, rng: Generator
    ) -> list[Pair]:
        factuals = kmedoids(distance(X_c), min(s, len(X_c)))
        counterfactuals = kmedoids(distance(X_not_c), min(k, len(X_not_c)))
        return [Pair(int(position), counterfactuals) for position in factuals]


class RandomPairsStrategy(MiningStrategy):
    """Random factuals paired with random counterfactuals.

    Each of the ``s`` random factuals of ``X_c`` gets ``k`` counterfactuals
    drawn at random from ``X_not_c``. It is one of the components of
    :class:`EnsembleStrategy`.
    """

    name = "random-pairs"

    def select(
        self, X_c: pd.DataFrame, X_not_c: pd.DataFrame, s: int, k: int, distance: Distance, rng: Generator
    ) -> list[Pair]:
        s = min(s, len(X_c))
        positions = rng.choice(len(X_c), s, replace=False)
        k = min(k, len(X_not_c))
        return [Pair(int(position), rng.choice(len(X_not_c), k, replace=False)) for position in positions]


class EnsembleStrategy(MiningStrategy):
    """Ensemble of heterogeneous strategies, the reference RaMiCo configuration.

    Each component runs on its own and keeps its label: rules are aggregated
    per component, not merged across components. The default components are
    those of the original implementation, in this order: Prototype,
    nearest-boundary and random-pairs mining.

    Parameters
    ----------
    strategies : sequence of MiningStrategy, optional
        Components to combine, run in the given order.
    """

    name = "ensemble"

    def __init__(self, strategies: Sequence[MiningStrategy] | None = None) -> None:
        if strategies is None:
            strategies = (PrototypeStrategy(), NearestBoundaryStrategy(), RandomPairsStrategy())
        self.strategies = tuple(strategies)

    def select(
        self, X_c: pd.DataFrame, X_not_c: pd.DataFrame, s: int, k: int, distance: Distance, rng: Generator
    ) -> list[Pair]:
        pairs: list[Pair] = []
        for _, strategy in self.components():
            pairs.extend(strategy.select(X_c, X_not_c, s, k, distance, rng))
        return pairs

    def components(self) -> list[tuple[str, MiningStrategy]]:
        return [component for strategy in self.strategies for component in strategy.components()]

    def __repr__(self) -> str:
        return f"EnsembleStrategy(strategies={list(self.strategies)!r})"


_REGISTRY: dict[str, Callable[[], MiningStrategy]] = {}


def register_strategy(name: str, factory: Callable[[], MiningStrategy], *, overwrite: bool = False) -> None:
    """Make a strategy available by name.

    Parameters
    ----------
    name : str
        Name to use, for example in ``RaMiCo(model, strategy=name)``.
    factory : callable
        Called without arguments, returns a :class:`MiningStrategy`; usually
        the strategy class itself.
    overwrite : bool, default=False
        Replace a strategy already registered under ``name``.

    Raises
    ------
    ValueError
        If ``name`` is taken and ``overwrite`` is False.

    Examples
    --------
    >>> import numpy as np
    >>> from ramico.strategies import MiningStrategy, Pair, register_strategy
    >>> class FirstSamples(MiningStrategy):
    ...     name = "first"
    ...     def select(self, X_c, X_not_c, s, k, distance, rng):
    ...         counterfactuals = np.arange(min(k, len(X_not_c)))
    ...         return [Pair(i, counterfactuals) for i in range(min(s, len(X_c)))]
    >>> register_strategy("first", FirstSamples)
    """
    if name in _REGISTRY and not overwrite:
        raise ValueError(f"a strategy named {name!r} is already registered; pass overwrite=True")
    _REGISTRY[name] = factory


def available_strategies() -> list[str]:
    """Names of the registered strategies, sorted."""
    return sorted(_REGISTRY)


def get_strategy(strategy: str | MiningStrategy) -> MiningStrategy:
    """Return a strategy instance from a name or an instance.

    Raises
    ------
    ValueError
        If no strategy is registered under the given name.
    TypeError
        If ``strategy`` is neither a string nor a :class:`MiningStrategy`.
    """
    if isinstance(strategy, MiningStrategy):
        return strategy
    if isinstance(strategy, str):
        try:
            factory = _REGISTRY[strategy]
        except KeyError:
            raise ValueError(f"unknown strategy {strategy!r}; available: {', '.join(available_strategies())}") from None
        return factory()
    raise TypeError(f"strategy must be a name or a MiningStrategy, not {type(strategy).__name__}")


register_strategy("boundary", BoundaryStrategy)
register_strategy("prototype", PrototypeStrategy)
register_strategy("random", RandomStrategy)
register_strategy("ensemble", EnsembleStrategy)
