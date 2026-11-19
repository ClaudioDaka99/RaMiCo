"""Result of :meth:`ramico.RaMiCo.fit`: the rules and the records behind them."""

from __future__ import annotations

import html
import math
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from ramico import _io
from ramico._scores import Scores
from ramico.exceptions import DataError

# public names of the rule columns, in display order
RULE_COLUMNS = [
    "strategy",
    "from_class",
    "to_class",
    "feature",
    "aggregation",
    "from_value",
    "from_std",
    "from_confidence",
    "to_value",
    "to_std",
    "to_confidence",
    "n_records",
    "support",
    "implausibility",
    "implausibility_std",
    "mined_class",
]
RECORD_COLUMNS = [
    "strategy",
    "factual_index",
    "feature",
    "from_value",
    "to_value",
    "from_class",
    "to_class",
    "mined_class",
    "implausibility",
]
FACTUAL_PREFIX = "factual."
# identification of the files written by RuleSet.save; the version changes when their layout does
FILE_FORMAT = "ramico.RuleSet"
FILE_VERSION = 1


def _theme() -> Any:
    # imported when rendering: ramico.viz itself imports this module
    from ramico.viz import theme

    return theme


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def format_value(value: Any) -> str:
    """Compact text of a rule value: numbers with two decimals, the rest as is."""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):,.2f}"
    return str(value)


def format_data_value(value: Any) -> str:
    """A value as found in the data: whole numbers without decimals, others as written."""
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return f"{int(number):,}" if number.is_integer() else f"{number:,.6g}"
    return format_value(value)


@dataclass(frozen=True)
class Rule:
    """One counterfactual rule: switching ``feature`` moves the model between two classes.

    Numerical rules (``aggregation == "mean"``) give the mean and the standard
    deviation of the values before and after the switch: the rule holds for
    values in ``mean ± std``. Categorical rules (``aggregation == "mode"``)
    give the most frequent value and the share of records holding it.
    """

    strategy: str
    from_class: Any
    to_class: Any
    feature: str
    aggregation: str
    from_value: Any
    from_std: float
    from_confidence: float
    to_value: Any
    to_std: float
    to_confidence: float
    n_records: int
    support: float
    implausibility: float
    implausibility_std: float
    from_name: str = ""
    to_name: str = ""

    def _side(self, value: Any, std: float, confidence: float) -> str:
        if self.aggregation == "mean":
            return f"{format_value(value)} ± {format_value(std)}"
        return f"{format_data_value(value)} ({float(confidence):.3g}%)"

    def _change(self, arrow: str) -> str:
        before = self._side(self.from_value, self.from_std, self.from_confidence)
        after = self._side(self.to_value, self.to_std, self.to_confidence)
        return f"{before} {arrow} {after}"

    @property
    def change(self) -> str:
        """The switch in words, e.g. ``"45.20 ± 3.10 -> 61.00 ± 5.40"``."""
        return self._change("->")

    def __str__(self) -> str:
        # ASCII arrows: the text must print on any console encoding
        return (
            f"{self.feature}: {self.change}  |  class {self.from_name or self.from_class} -> "
            f"{self.to_name or self.to_class}  |  support {self.support:.2f}% "
            f"({self.n_records} records)  |  implausibility {self.implausibility:.3f}"
        )

    def _repr_html_(self) -> str:
        def cell(label: str, value: str) -> str:
            return (
                f'<div style="padding:6px 10px"><div style="font-size:11px;opacity:.65;'
                f'text-transform:uppercase;letter-spacing:.04em">{html.escape(label)}</div>'
                f'<div style="font-size:15px;font-weight:600">{value}</div></div>'
            )

        unit = "mean ± std" if self.aggregation == "mean" else "mode (share of records)"
        rows = [
            cell("feature switched", f"<code>{html.escape(self.feature)}</code>"),
            cell(
                f"factual value · {unit}", html.escape(self._side(self.from_value, self.from_std, self.from_confidence))
            ),
            cell(
                f"counterfactual value · {unit}",
                html.escape(self._side(self.to_value, self.to_std, self.to_confidence)),
            ),
            cell(
                "prediction",
                f"{html.escape(str(self.from_name or self.from_class))} → "
                f"{html.escape(str(self.to_name or self.to_class))}",
            ),
            cell("support", f"{self.support:.2f}% · {self.n_records} records"),
            cell("implausibility", f"{_chip(self.implausibility)} ± {self.implausibility_std:.3f}"),
        ]
        return (
            '<div style="display:inline-grid;grid-template-columns:repeat(3,auto);gap:2px 18px;'
            "border:1px solid rgba(127,127,127,.35);border-radius:10px;padding:8px 6px;"
            'font-family:system-ui,-apple-system,Segoe UI,sans-serif">' + "".join(rows) + "</div>"
        )


def _chip(value: float) -> str:
    """Implausibility as a chip whose blue deepens with the value (0 lightest).

    Only the light half of the sequential ramp is used, so the text keeps the
    primary ink with a contrast of at least 5:1 on every shade.
    """
    if _is_missing(value):
        return '<span style="opacity:.6">n/a</span>'
    theme = _theme()
    shades = theme.SEQUENTIAL[:7]
    t = min(max(float(value), 0.0), 1.0)
    return (
        f'<span style="background:{shades[round(t * (len(shades) - 1))]};color:{theme.INK["light"]["primary"]};'
        f'padding:1px 8px;border-radius:9px;font-variant-numeric:tabular-nums">{value:.3f}</span>'
    )


def _bar(support: float) -> str:
    width = min(max(support, 0.0), 100.0)
    return (
        '<div style="display:flex;align-items:center;gap:6px">'
        '<div style="width:90px;height:7px;border-radius:4px;background:rgba(127,127,127,.25)">'
        f'<div style="width:{width:.1f}%;height:7px;border-radius:4px;'
        f'background:{_theme().CATEGORICAL["light"][0]}"></div></div>'
        f'<span style="font-variant-numeric:tabular-nums">{support:.2f}%</span></div>'
    )


def _strategy_tag(strategy: str, colour: str) -> str:
    # the dot carries the colour; the name keeps the text colour
    return (
        '<span style="display:inline-flex;align-items:center;gap:5px;white-space:nowrap">'
        f'<span style="width:9px;height:9px;border-radius:50%;background:{colour}"></span>'
        f"{html.escape(strategy)}</span>"
    )


class RuleSet:
    """Counterfactual rules extracted by :class:`ramico.RaMiCo`, with their records.

    In Jupyter a RuleSet displays as a table grouped by class transition.

    Attributes
    ----------
    metadata : Mapping
        How the rules were obtained: strategy, ``n_factuals``,
        ``n_counterfactuals``, ``random_state``, Isolation Forest engine,
        classes, class names, number of samples and the indices of the mined
        factuals (``mined_indices``).
    scores : Scores
        Metrics saved with the rules (see :meth:`save`), empty otherwise.

    Examples
    --------
    >>> rules = RaMiCo(model).fit(X, y)           # doctest: +SKIP
    >>> rules.to_frame().head()                   # doctest: +SKIP
    >>> rules.filter(to_class=1, feature="chol")  # doctest: +SKIP
    >>> print(rules[0])                           # doctest: +SKIP
    >>> rules.save("rules.json")                  # doctest: +SKIP
    >>> rules = RuleSet.load("rules.json")        # doctest: +SKIP
    """

    def __init__(
        self,
        rules: pd.DataFrame,
        records: pd.DataFrame,
        features: list[str],
        metadata: Mapping[str, Any],
        scores: Mapping[str, Any] | None = None,
    ) -> None:
        self._rules = rules.reset_index(drop=True)
        self._records = records.reset_index(drop=True)
        self._features = list(features)
        self.metadata = MappingProxyType(dict(metadata))
        self._scores = Scores(scores)

    def __reduce__(self) -> tuple:
        # a read-only mapping cannot be pickled: rebuild from plain dicts, so that rules can be saved with
        # pickle or joblib and sent to worker processes
        return (RuleSet, (self._rules, self._records, self._features, dict(self.metadata), dict(self._scores)))

    # ------------------------------------------------------------- files

    def save(self, path: str | os.PathLike[str], scores: Mapping[str, Any] | None = None) -> None:
        """Write the rules, their records and how they were obtained to a JSON file.

        :meth:`load` reads the file back: the rules, the records and the
        metrics are then available without fitting or evaluating again.
        Values keep their types (numbers, category names, class labels), and
        the file is plain JSON that other tools can read: ``rules`` and
        ``records`` hold the columns of :meth:`to_frame` and :attr:`records`.

        Parameters
        ----------
        path : str or path-like
            File name ending in ``.json``, or ``.json.gz`` to compress the file
            with gzip: about 4 MB and 0.1 MB for the 20,000 records of the
            ensemble strategy on Adult with the paper's settings.
        scores : mapping, optional
            Metrics to keep with the rules, e.g. the result of
            :func:`ramico.metrics.evaluate`. By default the metrics the
            RuleSet already holds (:attr:`scores`).

        Raises
        ------
        ValueError
            ``path`` does not end in ``.json`` or ``.json.gz``.

        Examples
        --------
        >>> scores = metrics.evaluate(model, rules, X_train)     # doctest: +SKIP
        >>> rules.save("heart_rules.json", scores=scores)        # doctest: +SKIP
        >>> rules = RuleSet.load("heart_rules.json")             # doctest: +SKIP
        >>> rules.scores.fidelity                                # doctest: +SKIP
        """
        from ramico import __version__  # at call time: ramico imports this module

        path = _io.check_path(path)
        payload = {
            "format": FILE_FORMAT,
            "version": FILE_VERSION,
            "ramico_version": __version__,
            "features": self._features,
            "metadata": _io.encode_value(dict(self.metadata)),
            "scores": _io.encode_value(dict(self._scores if scores is None else scores)),
            "rules": _io.encode_table(self.to_frame()),
            "records": _io.encode_table(self.records),
        }
        _io.write_json(payload, path)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> RuleSet:
        """Read a RuleSet written by :meth:`save`, with its metrics in :attr:`scores`.

        Raises
        ------
        ramico.exceptions.DataError
            The file was not written by :meth:`save`, or by a newer RaMiCo.
        """
        path = _io.check_path(path)
        try:
            payload = _io.read_json(path)
        except _io.UNREADABLE as error:
            raise DataError(f"{path.name} is not a RuleSet file: {error}") from error
        if not isinstance(payload, dict) or payload.get("format") != FILE_FORMAT:
            raise DataError(f"{path.name} is not a RuleSet file written by RuleSet.save")
        version = payload.get("version")
        if not isinstance(version, int) or version > FILE_VERSION:
            raise DataError(
                f"{path.name} was written by RaMiCo {payload.get('ramico_version')} in a newer format "
                f"(version {version}); update RaMiCo to read it"
            )
        try:
            return cls._from_frames(
                _io.decode_table(payload["rules"]),
                _io.decode_table(payload["records"]),
                list(payload["features"]),
                _io.decode_value(payload["metadata"]),
                _io.decode_value(payload.get("scores") or {}),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise DataError(f"{path.name} is an incomplete or damaged RuleSet file: {error!r}") from error

    @classmethod
    def _from_frames(
        cls,
        rules: pd.DataFrame,
        records: pd.DataFrame,
        features: list[str],
        metadata: Mapping[str, Any],
        scores: Mapping[str, Any],
    ) -> RuleSet:
        """Inverse of :meth:`to_frame` and :attr:`records`: the RuleSet these views show."""
        from ramico._aggregation import RULE_COLUMNS as INTERNAL_RULE_COLUMNS

        is_mean = rules["aggregation"] == "mean"
        internal_rules = pd.DataFrame(
            {
                "feature": rules["feature"],
                "original_prediction": rules["from_class"],
                "switched_prediction": rules["to_class"],
                "original_value": rules["from_value"],
                "original_value_std/confidence": rules["from_std"].where(is_mean, rules["from_confidence"]),
                "switched_value": rules["to_value"],
                "switched_value_std/confidence": rules["to_std"].where(is_mean, rules["to_confidence"]),
                "anomaly_score_mean": rules["implausibility"],
                "anomaly_score_std": rules["implausibility_std"],
                "switch_count_per_feat": rules["n_records"],
                "support": rules["support"],
                "expected_label": rules["mined_class"],
                "method": rules["strategy"],
                "aggregation": rules["aggregation"],
            },
            columns=INTERNAL_RULE_COLUMNS,
        )
        internal_records = pd.DataFrame(
            {
                "feature": records["feature"],
                "original_value": records["from_value"],
                "switched_value": records["to_value"],
                "original_prediction": records["from_class"],
                "switched_prediction": records["to_class"],
                "anomaly_score": records["implausibility"],
                "expected_label": records["mined_class"],
                **{feature: records[FACTUAL_PREFIX + feature] for feature in features},
                "sample_index": records["factual_index"],
                "method": records["strategy"],
            }
        )
        return cls(internal_rules, internal_records, features, metadata, scores)

    # ------------------------------------------------------------ metadata

    @property
    def strategy(self) -> str:
        """Name of the mining strategy."""
        return str(self.metadata.get("strategy"))

    @property
    def classes(self) -> list:
        """Classes, in order of first appearance in ``y``."""
        return list(self.metadata.get("classes", []))

    @property
    def mined_indices(self) -> list:
        """Row labels of the factuals selected by the strategy."""
        return list(self.metadata.get("mined_indices", []))

    @property
    def scores(self) -> Scores:
        """Metrics saved with the rules by :meth:`save` (``rules.scores.fidelity``); empty when there are none.

        The metrics describe the whole set: :meth:`filter` leaves them out.
        """
        return self._scores

    def class_name(self, label: Any) -> str:
        """Display name of a class label."""
        names = self.metadata.get("class_names") or {}
        return str(names.get(label, label))

    # --------------------------------------------------------------- views

    def to_frame(self) -> pd.DataFrame:
        """The rules as a DataFrame, sorted by transition and decreasing support.

        Columns: ``strategy``, ``from_class``, ``to_class``, ``feature``,
        ``aggregation`` ("mean" or "mode"), ``from_value``, ``from_std``,
        ``from_confidence``, ``to_value``, ``to_std``, ``to_confidence``,
        ``n_records``, ``support`` (% of the records), ``implausibility`` and
        ``implausibility_std`` (normalised anomaly score of the records) and
        ``mined_class`` (class whose factuals produced the rule).
        """
        r = self._rules
        is_mean = r["aggregation"] == "mean"
        # counts and labels were cast to float by the original aggregation
        labels = {label: label for label in self.classes}
        mined = r["expected_label"].map(lambda v: next((c for c in labels if c == v), v))
        frame = pd.DataFrame(
            {
                "strategy": r["method"],
                "from_class": r["original_prediction"],
                "to_class": r["switched_prediction"],
                "feature": r["feature"],
                "aggregation": r["aggregation"],
                "from_value": r["original_value"],
                "from_std": r["original_value_std/confidence"].where(is_mean),
                "from_confidence": r["original_value_std/confidence"].where(~is_mean),
                "to_value": r["switched_value"],
                "to_std": r["switched_value_std/confidence"].where(is_mean),
                "to_confidence": r["switched_value_std/confidence"].where(~is_mean),
                "n_records": r["switch_count_per_feat"].astype(int),
                "support": r["support"],
                "implausibility": r["anomaly_score_mean"],
                "implausibility_std": r["anomaly_score_std"],
                "mined_class": mined,
            },
            columns=RULE_COLUMNS,
        )
        order = {label: i for i, label in enumerate(self.classes)}
        keys = pd.DataFrame(
            {
                "f": frame["from_class"].map(order),
                "t": frame["to_class"].map(order),
                "s": -frame["support"].astype(float),
            }
        )
        index = keys.sort_values(["f", "t", "s"], kind="stable").index
        return frame.loc[index].reset_index(drop=True)

    @property
    def records(self) -> pd.DataFrame:
        """The elementary records: one row per single-feature switch that flips the model.

        The factual's values follow, in columns prefixed with ``factual.``.
        """
        r = self._records
        frame = pd.DataFrame(
            {
                "strategy": r["method"],
                "factual_index": r["sample_index"],
                "feature": r["feature"],
                "from_value": r["original_value"],
                "to_value": r["switched_value"],
                "from_class": r["original_prediction"],
                "to_class": r["switched_prediction"],
                "mined_class": r["expected_label"],
                "implausibility": r["anomaly_score"],
            },
            columns=RECORD_COLUMNS,
        )
        factuals = r[self._features].add_prefix(FACTUAL_PREFIX)
        return pd.concat([frame, factuals], axis=1)

    def records_of(self, rule: Rule | int) -> pd.DataFrame:
        """The records aggregated into ``rule`` (a Rule or its position), as in :attr:`records`."""
        if not isinstance(rule, Rule):
            rule = self[rule]
        records = self.records
        keep = (
            (records["feature"] == rule.feature)
            & (records["from_class"] == rule.from_class)
            & (records["to_class"] == rule.to_class)
            & (records["strategy"] == rule.strategy)
        )
        return records[keep].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self._rules)

    def _rule(self, row: pd.Series) -> Rule:
        # plain Python scalars, whichever way the row was read from the frame
        return Rule(
            **{name: _scalar(row[name]) for name in RULE_COLUMNS if name != "mined_class"},
            from_name=self.class_name(row["from_class"]),
            to_name=self.class_name(row["to_class"]),
        )

    def __iter__(self) -> Iterator[Rule]:
        for _, row in self.to_frame().iterrows():
            yield self._rule(row)

    def __getitem__(self, position: int) -> Rule:
        return self._rule(self.to_frame().iloc[position])

    def filter(
        self,
        *,
        from_class: Any = None,
        to_class: Any = None,
        feature: str | list[str] | None = None,
        strategy: str | list[str] | None = None,
    ) -> RuleSet:
        """Rules (and records) matching every given condition.

        Each argument takes one value or a list of values. Supports are not
        recomputed: they stay relative to all the records. The metrics in
        :attr:`scores`, which describe all the rules, are not kept.

        Examples
        --------
        >>> rules.filter(from_class=0, to_class=1)            # doctest: +SKIP
        >>> rules.filter(feature=["chol", "trestbps"])        # doctest: +SKIP
        """

        def mask(frame: pd.DataFrame) -> pd.Series:
            keep = pd.Series(True, index=frame.index)
            for column, wanted in (
                ("original_prediction", from_class),
                ("switched_prediction", to_class),
                ("feature", feature),
                ("method", strategy),
            ):
                if wanted is not None:
                    values = list(wanted) if isinstance(wanted, (list, tuple, set)) else [wanted]
                    keep &= frame[column].isin(values)
            return keep

        rules = self._rules[mask(self._rules)] if len(self._rules) else self._rules
        records = self._records[mask(self._records)] if len(self._records) else self._records
        return RuleSet(rules, records, self._features, self.metadata)

    # ------------------------------------------------------------- display

    def to_table(self, max_rules: int = 12, title: str | None = None) -> Any:
        """Publication-style table of the rules (great_tables; needs ``ramico[viz]``)."""
        from ramico.viz.tables import rules_table

        return rules_table(self, max_rules=max_rules, title=title)

    def plot(
        self,
        *,
        min_support: float = 0.0,
        max_change: float = 1.0,
        max_implausibility: float = 1.0,
        mode: str = "light",
    ) -> Any:
        """All the rules as an interactive graph (plotly; needs ``ramico[viz]``).

        Each rule is an arrow between two values of its feature, with sliders
        on support, size of the change and implausibility; the arguments set
        the sliders when the figure opens. See
        :func:`ramico.viz.plot_rule_graph`, which this method calls.

        Examples
        --------
        >>> viz.show(rules.plot(min_support=2))  # doctest: +SKIP
        """
        from ramico.viz.plots import plot_rule_graph

        return plot_rule_graph(
            self,
            min_support=min_support,
            max_change=max_change,
            max_implausibility=max_implausibility,
            mode=mode,  # type: ignore[arg-type]
        )

    def __repr__(self) -> str:
        return (
            f"RuleSet(strategy={self.strategy!r}, rules={len(self)}, "
            f"records={len(self._records)}, classes={self.classes})"
        )

    def _repr_html_(self, max_rules: int = 12) -> str:
        meta = self.metadata
        engine = meta.get("isolation_forest")
        subtitle = " · ".join(
            part
            for part in (
                f"strategy <b>{html.escape(self.strategy)}</b>",
                f"{meta.get('n_factuals')} factuals × {meta.get('n_counterfactuals')} counterfactuals per class",
                f"isolation forest {html.escape(str(engine))}" if engine else "",
            )
            if part
        )
        head = (
            '<div style="font-family:system-ui,-apple-system,Segoe UI,sans-serif;font-size:13px;'
            'line-height:1.45;max-width:1000px">'
            '<div style="display:flex;gap:10px;align-items:baseline;flex-wrap:wrap">'
            '<span style="font-size:16px;font-weight:700">RaMiCo rules</span>'
            f'<span style="opacity:.75">{len(self)} rules from {len(self._records)} records</span></div>'
            f'<div style="opacity:.75;margin-bottom:6px">{subtitle}</div>'
        )
        if not len(self):
            return head + '<div style="opacity:.75">No rules: no single-feature switch flipped the model.</div></div>'

        frame = self.to_frame()
        several = frame["strategy"].nunique() > 1
        theme = _theme()
        colours = theme.strategy_colors(sorted(frame["strategy"].unique(), key=theme.strategy_sort_key))
        blocks = []
        for (source, target), group in frame.groupby(["from_class", "to_class"], sort=False):
            shown = group.head(max_rules)
            header_cells = ["feature", "switch", "support", "implausibility", "records"] + (
                ["strategy"] if several else []
            )
            align = {"records": "right"}
            th = "".join(
                f'<th style="text-align:{align.get(h, "left")};padding:4px 10px;font-weight:600;opacity:.8">{h}</th>'
                for h in header_cells
            )
            rows = []
            for _, row in shown.iterrows():
                rule = self._rule(row)
                cells = [
                    f"<code>{html.escape(str(rule.feature))}</code>",
                    html.escape(rule._change("→")),
                    _bar(float(rule.support)),
                    _chip(rule.implausibility),
                    str(rule.n_records),
                ] + ([_strategy_tag(str(rule.strategy), colours[rule.strategy])] if several else [])
                rows.append(
                    '<tr style="border-top:1px solid rgba(127,127,127,.2)">'
                    + "".join(
                        f'<td style="padding:4px 10px;white-space:nowrap;text-align:{align.get(h, "left")}">{c}</td>'
                        for h, c in zip(header_cells, cells, strict=True)
                    )
                    + "</tr>"
                )
            more = len(group) - len(shown)
            note = (
                f'<div style="opacity:.6;font-size:12px;padding:2px 10px">… {more} more rules '
                "with lower support: <code>.to_frame()</code> lists them all</div>"
                if more > 0
                else ""
            )
            blocks.append(
                f'<div style="margin-top:10px;font-weight:600">{html.escape(self.class_name(source))} '
                f"→ {html.escape(self.class_name(target))} "
                f'<span style="font-weight:400;opacity:.65">· {len(group)} rules</span></div>'
                '<table style="border-collapse:collapse;width:auto">'
                f"<thead><tr>{th}</tr></thead><tbody>{''.join(rows)}</tbody></table>{note}"
            )
        return head + "".join(blocks) + "</div>"


def empty_rules(features: list[str], metadata: Mapping[str, Any]) -> RuleSet:
    """RuleSet without rules, with the columns of the internal tables."""
    from ramico._aggregation import RULE_COLUMNS as INTERNAL_RULE_COLUMNS
    from ramico._records import RECORD_COLUMNS as INTERNAL_RECORD_COLUMNS

    records = pd.DataFrame(columns=INTERNAL_RECORD_COLUMNS + features + ["sample_index", "method"])
    return RuleSet(pd.DataFrame(columns=INTERNAL_RULE_COLUMNS), records, features, metadata)
