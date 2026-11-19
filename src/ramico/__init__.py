"""RaMiCo: Region-Aware Minimal Counterfactual Rules.

Model-agnostic extraction of global counterfactual rules from black-box
classifiers::

    from ramico import RaMiCo
    rules = RaMiCo(model).fit(X, y)

Submodules, loaded on first use: :mod:`ramico.strategies`,
:mod:`ramico.metrics`, :mod:`ramico.datasets`, :mod:`ramico.benchmark`,
:mod:`ramico.viz` and :mod:`ramico.exceptions`.
"""

from __future__ import annotations

import importlib
from typing import Any

from ramico._explainer import RaMiCo
from ramico._ruleset import Rule, RuleSet

__version__ = "0.1.0"

__all__ = ["RaMiCo", "Rule", "RuleSet"]

_SUBMODULES = {"benchmark", "datasets", "exceptions", "metrics", "strategies", "viz"}


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        return importlib.import_module(f"ramico.{name}")
    raise AttributeError(f"module 'ramico' has no attribute {name!r}")
