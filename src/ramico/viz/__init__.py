"""Visual helpers of RaMiCo: one theme, tables and plots.

The functions need the ``[viz]`` extra (plotly, matplotlib, great_tables,
kaleido), imported only when a table or a figure is built.

>>> from ramico import viz                    # doctest: +SKIP
>>> viz.theme.apply()                         # doctest: +SKIP
>>> viz.rules_table(rules)                    # doctest: +SKIP
>>> viz.show(viz.plot_pareto(results))       # doctest: +SKIP
"""

from __future__ import annotations

from ramico.viz import theme
from ramico.viz.plots import plot_pareto, plot_rule, plot_rule_graph, plot_sensitivity, show
from ramico.viz.tables import comparison_table, rules_table, sensitivity_table

__all__ = [
    "comparison_table",
    "plot_pareto",
    "plot_rule",
    "plot_rule_graph",
    "plot_sensitivity",
    "rules_table",
    "sensitivity_table",
    "show",
    "theme",
]
