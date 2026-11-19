"""Publication-style tables (great_tables) for rules and benchmark results."""

from __future__ import annotations

import html
from typing import Any

import pandas as pd

from ramico._ruleset import Rule, RuleSet
from ramico.exceptions import MissingDependencyError
from ramico.viz.theme import SEQUENTIAL, strategy_colors, strategy_sort_key

__all__ = ["comparison_table", "rules_table", "sensitivity_table"]

_METRIC_LABELS = {
    "fidelity": "Fidelity (%) ↑",
    "incomprehensibility": "Incomprehensibility ↓",
    "implausibility": "Implausibility ↓",
}


def _great_tables() -> Any:
    try:
        import great_tables
    except ImportError as exc:
        raise MissingDependencyError('tables need great_tables: pip install "ramico[viz]"') from exc
    return great_tables


def rules_table(rules: RuleSet, max_rules: int = 12, title: str | None = None) -> Any:
    """The rules as a great_tables table, grouped by class transition.

    Parameters
    ----------
    rules : RuleSet
        Rules to show.
    max_rules : int, default=12
        Rules shown per transition, by decreasing support.
    title : str, optional
        Table title.

    Returns
    -------
    great_tables.GT
    """
    gt = _great_tables()
    frame = rules.to_frame()
    shown = frame.groupby(["from_class", "to_class"], sort=False).head(max_rules)
    several = shown["strategy"].nunique() > 1
    entries = [rules._rule(row) for _, row in shown.iterrows()]

    def side(rule: Rule, before: bool) -> str:
        if before:
            return rule._side(rule.from_value, rule.from_std, rule.from_confidence)
        return rule._side(rule.to_value, rule.to_std, rule.to_confidence)

    table = pd.DataFrame(
        {
            "transition": [f"{r.from_name or r.from_class} → {r.to_name or r.to_class}" for r in entries],
            "feature": [r.feature for r in entries],
            "factual": [side(r, True) for r in entries],
            "counterfactual": [side(r, False) for r in entries],
            "support": [float(r.support) / 100 for r in entries],
            "implausibility": [float(r.implausibility) for r in entries],
            "records": [int(r.n_records) for r in entries],
        }
    )
    if several:
        table["strategy"] = [r.strategy for r in entries]
    meta = rules.metadata
    subtitle = (
        f"{rules.strategy} strategy · {meta.get('n_factuals')} factuals × "
        f"{meta.get('n_counterfactuals')} counterfactuals per class · "
        f"{len(frame):,} rules from {len(rules.records):,} records"
    )
    styled = (
        gt.GT(table, groupname_col="transition")
        .tab_header(title=title or "Counterfactual rules", subtitle=subtitle)
        .tab_spanner(label="Switch", columns=["factual", "counterfactual"])
        .tab_spanner(label="Quality", columns=["support", "implausibility", "records"])
        .cols_label(
            feature="Feature",
            factual="Factual value",
            counterfactual="Counterfactual value",
            support="Support",
            implausibility="Implausibility",
            records="Records",
            **({"strategy": "Strategy"} if several else {}),
        )
        .fmt_percent(columns="support", decimals=1)
        .fmt_number(columns="implausibility", decimals=3)
        .data_color(columns="implausibility", palette=list(SEQUENTIAL[:7]), domain=_domain(table["implausibility"]))
        .tab_source_note(
            "Numerical rules: mean ± standard deviation of the values before and after the switch; "
            "categorical rules: most frequent value (share of records). Support: share of all records. "
            f"Implausibility: normalised Isolation Forest anomaly score ({meta.get('isolation_forest')}), "
            "0 = typical of the training data, 1 = most anomalous; shaded from the lowest to the highest value shown."
        )
    )
    if len(shown) < len(frame):
        styled = styled.tab_source_note(
            f"Showing up to {max_rules} rules per transition, by decreasing support; "
            "RuleSet.to_frame() lists all of them."
        )
    if several:
        names = sorted(set(table["strategy"]), key=strategy_sort_key)
        colours = strategy_colors(names)
        styled = styled.fmt(
            lambda name: f'<span style="color:{colours[name]}">&#9679;</span>&nbsp;{html.escape(name)}',
            columns="strategy",
        )
    return styled


def comparison_table(
    results: pd.DataFrame,
    reference: str = "ensemble",
    comparison: pd.DataFrame | None = None,
    alpha: float = 0.05,
    title: str | None = None,
    by: str | None = None,
) -> Any:
    """Mean ± standard deviation of each metric per strategy, best value in bold.

    Parameters
    ----------
    results : pandas.DataFrame
        Output of :func:`ramico.benchmark.run_benchmark`.
    reference : str, default="ensemble"
        Strategy the others are tested against.
    comparison : pandas.DataFrame, optional
        Output of :func:`ramico.benchmark.compare_strategies`; computed when
        omitted. Significant differences are marked with an asterisk. With
        ``by``, the tests are computed within each group.
    alpha : float, default=0.05
        Significance level, used when the tests are computed here.
    title : str, optional
        Table title.
    by : str, optional
        Column of ``results`` (a model, a dataset...) whose values become
        groups of columns, each with its own tests and best values.

    Returns
    -------
    great_tables.GT
    """
    from ramico.benchmark import METRICS, compare_strategies

    gt = _great_tables()
    metrics = [m for m in METRICS if m in results.columns]
    strategies = sorted(pd.unique(results["strategy"]), key=lambda s: (s != reference, *strategy_sort_key(s)))
    groups = [("", results)] if by is None else [(str(g), part) for g, part in results.groupby(by, sort=False)]
    table = pd.DataFrame({"strategy": strategies})
    best: dict[str, str] = {}
    for group, part in groups:
        if comparison is None or by is not None:
            tests = compare_strategies(part, reference=reference, metrics=metrics, alpha=alpha)
        else:
            tests = comparison
        significant = {(row.metric, row.strategy) for row in tests.itertuples() if bool(row.significant)}
        summary = part.groupby("strategy")[metrics].agg(["mean", "std"])
        if by is None:
            table["runs"] = [int((part["strategy"] == s).sum()) for s in strategies]
        for metric in metrics:
            column = f"{group}|{metric}" if by is not None else metric
            digits = 1 if metric == "fidelity" else 2 if metric == "incomprehensibility" else 3
            table[column] = [
                f"{summary.loc[s, (metric, 'mean')]:.{digits}f} ± {summary.loc[s, (metric, 'std')]:.{digits}f}"
                + (" *" if (metric, s) in significant else "")
                if s in summary.index
                else ""
                for s in strategies
            ]
            means = summary[(metric, "mean")]
            best[column] = means.idxmax() if METRICS[metric] == "higher" else means.idxmin()
    runs = results.groupby(["strategy"] if by is None else [by, "strategy"]).size()
    styled = (
        gt.GT(table, rowname_col="strategy")
        .tab_header(
            title=title or "Mining strategies compared",
            subtitle=f"mean ± standard deviation over the seeds; reference: {reference}"
            + ("" if by is None else f"; {runs.min()} runs per strategy and {by}"),
        )
        .tab_source_note(
            f"* p < {alpha} against {reference}: two-sided Mann-Whitney U test, Benjamini-Hochberg correction "
            "within each metric" + (f" and {by}" if by else "") + ". Bold: best mean of the column."
        )
    )
    if by is None:
        styled = styled.cols_label(runs="Runs", **{m: _METRIC_LABELS.get(m, m) for m in metrics})
    else:
        for group, _ in groups:
            columns = [f"{group}|{m}" for m in metrics]
            styled = styled.tab_spanner(group, columns).cols_label(
                **{f"{group}|{m}": _METRIC_LABELS.get(m, m) for m in metrics}
            )
    for column, strategy in best.items():
        styled = styled.tab_style(
            style=gt.style.text(weight="bold"),
            locations=gt.loc.body(columns=column, rows=[strategies.index(strategy)]),
        )
    return styled


def sensitivity_table(results: pd.DataFrame, title: str | None = None) -> Any:
    """Mean ± standard deviation of each metric for every (s, k), shaded by the mean.

    Parameters
    ----------
    results : pandas.DataFrame
        Outputs of :func:`ramico.benchmark.run_benchmark` for several values of
        ``n_factuals`` (s) and ``n_counterfactuals`` (k), concatenated.
    title : str, optional
        Table title.

    Returns
    -------
    great_tables.GT
    """
    from ramico.benchmark import METRICS

    gt = _great_tables()
    metrics = [m for m in METRICS if m in results.columns]
    ks = sorted(results["n_counterfactuals"].unique())
    table = pd.DataFrame({"s": sorted(results["n_factuals"].unique())})
    shades = {}
    for metric in metrics:
        grouped = results.groupby(["n_factuals", "n_counterfactuals"])[metric]
        mean, std = grouped.mean().unstack(), grouped.std().unstack()
        digits = 1 if metric == "fidelity" else 2 if metric == "incomprehensibility" else 3
        low, high = float(mean.min().min()), float(mean.max().max())
        for k in ks:
            column = f"{metric}_{k}"
            table[column] = [_mean_std(mean.loc[s, k], std.loc[s, k], digits) for s in table["s"]]
            shades[column] = [
                None  # no run with this s and k
                if pd.isna(mean.loc[s, k])
                else SEQUENTIAL[round(6 * (mean.loc[s, k] - low) / (high - low))]
                if high > low
                else SEQUENTIAL[0]
                for s in table["s"]
            ]
    runs = results.groupby(["n_factuals", "n_counterfactuals"]).size()
    styled = (
        gt.GT(table, rowname_col="s")
        .tab_header(
            title=title or "Sensitivity to s and k",
            subtitle=f"mean ± standard deviation over {_span(runs)} runs per cell; "
            "rows: s factuals per class, columns: k counterfactuals per factual",
        )
        .tab_stubhead("s")
        .tab_source_note("Shading: the mean within each metric, from lightest (lowest) to darkest (highest).")
    )
    for metric in metrics:
        columns = [f"{metric}_{k}" for k in ks]
        styled = styled.tab_spanner(f"{_METRIC_LABELS.get(metric, metric)} · k =", columns).cols_label(
            **{column: str(k) for column, k in zip(columns, ks, strict=True)}
        )
        for column in columns:
            for row, colour in enumerate(shades[column]):
                if colour is not None:
                    styled = styled.tab_style(gt.style.fill(color=colour), gt.loc.body(columns=column, rows=[row]))
    return styled.cols_align("center")


def _mean_std(mean: float, std: float, digits: int) -> str:
    """ "mean ± std", the mean alone for a single run, nothing for a cell without runs."""
    if pd.isna(mean):
        return ""
    return f"{mean:.{digits}f}" if pd.isna(std) else f"{mean:.{digits}f} ± {std:.{digits}f}"


def _span(counts: pd.Series) -> str:
    """ "n" or "n-m": the range of the numbers of runs."""
    low, high = int(counts.min()), int(counts.max())
    return str(low) if low == high else f"{low}-{high}"


def _domain(values: pd.Series) -> list[float]:
    """Colour domain spanning ``values``: on [0, 1] close implausibilities would all get the same shade."""
    low, high = float(values.min()), float(values.max())
    return [low, high] if high - low >= 1e-3 else [low - 5e-4, high + 5e-4]
