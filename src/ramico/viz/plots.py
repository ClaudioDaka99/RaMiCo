"""Interactive charts (plotly) of rules and benchmark results, in the RaMiCo theme."""

from __future__ import annotations

import base64
import io
import itertools
import logging
import os
from typing import Any

import numpy as np
import pandas as pd

from ramico._ruleset import FACTUAL_PREFIX, Rule, RuleSet, format_data_value
from ramico.exceptions import MissingDependencyError
from ramico.viz.theme import (
    INK,
    Mode,
    plotly_template,
    sequential_colorscale,
    strategy_colors,
    strategy_sort_key,
)

__all__ = [
    "plot_pareto",
    "plot_rule",
    "plot_rule_graph",
    "plot_sensitivity",
    "show",
]

logger = logging.getLogger(__name__)

_SYMBOLS = ("circle", "square", "diamond", "triangle-up", "triangle-down", "cross", "star", "hexagon")

#: Environment variable that turns the static copies on by default.
STATIC_ENV = "RAMICO_STATIC_FIGURES"


def show(figure: Any, static: bool | None = None, scale: float = 1.5) -> None:
    """Display a plotly figure in a notebook, optionally with a static PNG copy.

    The interactive figure needs JavaScript, which GitHub does not run. With
    the static copy, made with kaleido, the figure and its PNG are displayed
    as one output holding both representations: Jupyter and VS Code draw the
    interactive figure, GitHub the PNG. Without it, plotly.js is loaded from
    its CDN rather than embedded in the notebook.

    Parameters
    ----------
    figure : plotly.graph_objects.Figure
        Figure to display.
    static : bool, optional
        Add the PNG copy (skipped with a warning when kaleido cannot export
        the figure). By default only when the environment variable
        :data:`STATIC_ENV` is ``"1"``, as when the tutorials are executed for
        publication.
    scale : float, default=1.5
        Resolution of the PNG copy relative to the size of the figure.
    """
    try:
        from IPython.display import display
    except ImportError as exc:
        raise MissingDependencyError("show displays figures in Jupyter and needs IPython") from exc
    if static is None:
        static = os.environ.get(STATIC_ENV) == "1"
    png = None
    if static:
        try:
            png = figure.to_image(format="png", width=figure.layout.width or 900, scale=scale)
        except Exception as exc:  # kaleido, or the browser it drives, is missing
            logger.warning("No static copy of the figure: %s", exc)
    if png is None:
        figure.show(renderer="plotly_mimetype+notebook_connected")
        return
    import plotly.io as pio

    bundle = pio.renderers["plotly_mimetype"].to_mimebundle(figure.to_dict())
    bundle["image/png"] = base64.b64encode(_compact_png(png)).decode("ascii")
    display(bundle, raw=True)


def _compact_png(png: bytes) -> bytes:
    """The PNG with a 256-colour palette: the same chart in a quarter of the bytes."""
    try:
        from PIL import Image

        image = Image.open(io.BytesIO(png)).convert("RGB")
    except (ImportError, OSError):  # Pillow missing or not a readable image
        return png
    palette = image.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    out = io.BytesIO()
    palette.save(out, "PNG", optimize=True)
    return out.getvalue() if out.tell() < len(png) else png


def _plotly() -> Any:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise MissingDependencyError('plots need plotly: pip install "ramico[viz]"') from exc
    return go


def _title(text: str, subtitle: str, mode: Mode) -> dict[str, Any]:
    colour = INK[mode]["secondary"]
    return {"text": f"{text}<br><span style='font-size:12px;color:{colour}'>{subtitle}</span>"}


def _colour_range(values: Any) -> dict[str, float]:
    """``cmin`` and ``cmax`` spanning the finite ``values`` (a small interval around a single value)."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    low, high = (float(finite.min()), float(finite.max())) if len(finite) else (0.0, 1.0)
    if high - low < 1e-3:
        low, high = low - 5e-4, high + 5e-4
    return {"cmin": low, "cmax": high}


def _implausibility_bar(mode: Mode) -> dict[str, Any]:
    title = dict(text="Implausibility", font=dict(size=12, color=INK[mode]["secondary"]))
    return dict(title=title, thickness=12, outlinewidth=0)


def plot_rule(rules: RuleSet, rule: Rule | int = 0, mode: Mode = "light") -> Any:
    """The records behind one rule: values of the switched feature before and after the switch.

    Categorical values are drawn as a matrix of factual against
    counterfactual values, each cell with its number of records and coloured
    by their mean implausibility; numerical values as one dot per record,
    with the interval of the rule (mean ± standard deviation, shaded) or its
    most frequent value (line), coloured by the implausibility of the record.

    Parameters
    ----------
    rules : RuleSet
        Rules and their records.
    rule : Rule or int, default=0
        The rule, or its position in ``rules``.
    mode : {"light", "dark"}, default="light"
        Theme variant.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go = _plotly()
    if not isinstance(rule, Rule):
        rule = rules[rule]
    behind = rules.records_of(rule)
    values = pd.concat([behind["from_value"], behind["to_value"]]).infer_objects()
    if pd.api.types.is_numeric_dtype(values):
        figure = _rule_strips(go, rule, behind, mode)
    else:
        figure = _rule_matrix(go, behind, mode)
    figure.update_layout(
        template=plotly_template(mode),
        title=_title(
            f"{rule.feature}: {rules.class_name(rule.from_class)} → {rules.class_name(rule.to_class)}",
            f"{len(behind)} records · {rule.strategy} strategy · colour: implausibility of the switched sample",
            mode,
        ),
    )
    return figure


def _rule_matrix(go: Any, behind: pd.DataFrame, mode: Mode) -> Any:
    """Factual values (rows) against counterfactual values (columns), most frequent first."""
    keys = behind.assign(before=behind["from_value"].astype(str), after=behind["to_value"].astype(str))
    cells = keys.groupby(["before", "after"])["implausibility"].agg(["size", "mean"])
    rows = keys["before"].value_counts(sort=True).index.tolist()
    columns = keys["after"].value_counts(sort=True).index.tolist()
    size = cells["size"].unstack().reindex(index=rows, columns=columns)
    mean = cells["mean"].unstack().reindex(index=rows, columns=columns)
    scale = _colour_range(mean.to_numpy())
    ink = INK[mode]
    total = len(behind)
    figure = go.Figure(
        go.Heatmap(
            z=mean.to_numpy(),
            x=columns,
            y=rows,
            colorscale=sequential_colorscale(start=1),
            zmin=scale["cmin"],
            zmax=scale["cmax"],
            xgap=3,
            ygap=3,
            colorbar=_implausibility_bar(mode),
            customdata=size.to_numpy(),
            hovertemplate="%{y} → %{x}<br>%{customdata} records · implausibility %{z:.3f}<extra></extra>",
        )
    )
    for before in rows:
        for after in columns:
            count = size.loc[before, after]
            if pd.isna(count):
                continue
            position = (mean.loc[before, after] - scale["cmin"]) / (scale["cmax"] - scale["cmin"])
            figure.add_annotation(
                x=after, y=before, showarrow=False,
                text=f"<b>{int(count):,}</b> records · {count / total:.0%}<br>{mean.loc[before, after]:.3f}",
                font=dict(size=12, color=ink["surface"] if position > 0.55 else ink["primary"]),
            )  # fmt: skip
    figure.update_xaxes(title="Counterfactual value (after the switch)", showgrid=False, ticks="")
    figure.update_yaxes(title="Factual value", autorange="reversed", showgrid=False, ticks="")
    figure.update_layout(height=220 + 70 * len(rows), margin=dict(t=110))
    return figure


def _rule_strips(go: Any, rule: Rule, behind: pd.DataFrame, mode: Mode) -> Any:
    ink = INK[mode]
    jitter = np.random.default_rng(0)  # positions only, the same at every call
    rows = (
        ("Counterfactual value", behind["to_value"], rule.to_value, rule.to_std),
        ("Factual value", behind["from_value"], rule.from_value, rule.from_std),
    )
    figure = go.Figure()
    for y, (label, values, centre, spread) in enumerate(rows):
        if rule.aggregation == "mean" and not pd.isna(spread):
            figure.add_shape(
                type="rect", x0=centre - spread, x1=centre + spread, y0=y - 0.34, y1=y + 0.34,
                fillcolor=ink["grid"], line_width=0, layer="below",
            )  # fmt: skip
            summary = f"{centre:,.2f} ± {spread:,.2f}"
        else:
            summary = f"mode {centre:,.2f}"
        figure.add_shape(
            type="line", x0=centre, x1=centre, y0=y - 0.4, y1=y + 0.4, line=dict(color=ink["primary"], width=2)
        )
        figure.add_annotation(
            x=centre, y=y + 0.44, text=summary, showarrow=False, yanchor="bottom", font=dict(color=ink["secondary"])
        )
        figure.add_trace(
            go.Scatter(
                x=pd.to_numeric(values),
                y=y + jitter.uniform(-0.28, 0.28, len(values)),
                mode="markers",
                marker=dict(
                    size=7,
                    color=behind["implausibility"],
                    coloraxis="coloraxis",
                    line=dict(width=1, color=ink["surface"]),
                ),
                name=label,
                hovertemplate=f"{label} %{{x}}<br>implausibility %{{marker.color:.3f}}<extra></extra>",
            )
        )
    figure.update_layout(
        coloraxis=dict(colorscale=sequential_colorscale(), cmin=0, cmax=1, colorbar=_implausibility_bar(mode)),
        showlegend=False,
        xaxis_title=rule.feature,
        yaxis=dict(tickvals=[0, 1], ticktext=[row[0] for row in rows], range=[-0.6, 1.75], showgrid=False),
        height=380,
    )
    return figure


#: Steps of the support slider of :func:`plot_rule_graph`, in % of the records.
_SUPPORT_STEPS = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 50.0)
#: Steps of its change slider, as a share of the range of the feature; 1 keeps every rule.
_CHANGE_STEPS = (0.05, 0.1, 0.25, 0.5, 0.75, 1.0)
_GRID = 20  # numerical values are rounded to about 1/20 of the range of their feature
_BAND_GAP = 0.7  # empty space between the bands of two features, in node slots
_SLOT = 26  # pixels per node slot
_ARROW_AT = 18  # the arrowhead sits on this point of the 25 of a curve
_ARROW_WIDTH = 1.5  # every arrow has the same width, in pixels; the support is in its hover text and its slider
_HEAD_SIZE = 9.0  # size of the arrowheads, in pixels
_NODE_SIZE = 9.0  # size of the nodes, in pixels
_STRATEGY_BEND = 0.35  # extra bend of the arrows of each further strategy, in node slots (half of it at the middle)
# A node shared by several arrows can have its label drawn by more than one, and text drawn twice on itself looks
# bolder. Every label is drawn at least this many times, so that all of them look alike.
_LABEL_DRAWS = 3


def _switch_sizes(rules: RuleSet, frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, float]]:
    """How much each rule of ``frame`` changes its feature, from 0 to 1, and the range of each numerical feature.

    For a numerical rule, the distance between the means before and after the
    switch, as a share of the range of the feature over the records (factual
    and switched values). For a categorical rule, 1 when the most frequent
    value changes, 0 otherwise.
    """
    records = rules.records
    spans = {}
    for feature in frame.loc[frame["aggregation"] == "mean", "feature"].unique():
        values = pd.concat([records[FACTUAL_PREFIX + feature], records.loc[records["feature"] == feature, "to_value"]])
        values = pd.to_numeric(values, errors="coerce")
        spans[feature] = float(values.max() - values.min())
    sizes = []
    for feature, kind, before, after in zip(
        frame["feature"], frame["aggregation"], frame["from_value"], frame["to_value"], strict=True
    ):
        if kind == "mean":
            span = spans[feature]
            sizes.append(min(abs(float(after) - float(before)) / span, 1.0) if span > 0 else 0.0)
        else:
            sizes.append(0.0 if before == after else 1.0)
    return np.asarray(sizes, dtype=float), spans


def _grid_step(span: float) -> float:
    """The largest round step, 1, 2 or 5 times a power of ten, not above ``span / _GRID``.

    A value moves by at most half a step, 2.5% of the range of its feature.
    """
    if not span > 0:
        return 1.0
    rough = span / _GRID
    power = 10.0 ** np.floor(np.log10(rough))
    return float(next(m * power for m in (5, 2, 1) if m * power <= rough * (1 + 1e-9)))


def _grid_label(value: float, step: float) -> str:
    decimals = max(0, -int(np.floor(np.log10(step)))) if step < 1 else 0
    return f"{value:,.{decimals}f}"


def _steps(defaults: tuple[float, ...], start: float, keep: Any) -> list[float]:
    """Slider values: the ``defaults`` that ``keep`` accepts, and ``start``."""
    return sorted({value for value in defaults if keep(value)} | {float(start)})


def _non_dominated(edges: list[tuple[int, float, float, float]]) -> list[int]:
    """Edges that no other one beats on every threshold: at least the support, at most the change and implausibility.

    A node is shown when one of its edges is, and the edges shown then
    include one of these: the label of the node is drawn with them only.
    """
    merit = {i: (support, -size, -implausibility) for i, support, size, implausibility in edges}  # larger is better
    return [
        i
        for i, own in merit.items()
        if not any(
            all(a >= b for a, b in zip(other, own, strict=True)) and (other != own or j < i)
            for j, other in merit.items()
            if j != i
        )
    ]


def plot_rule_graph(
    rules: RuleSet,
    min_support: float = 0.0,
    max_change: float = 1.0,
    max_implausibility: float = 1.0,
    mode: Mode = "light",
) -> Any:
    """All the rules as a graph: each rule is an arrow between two values of its feature.

    Each column holds the values of one class and each band one feature, the
    features with the strongest rule at the top. A rule is an arrow from the
    value its records start from, in the column of the starting class, to the
    value they are switched to, in the column of the arrival class; numerical
    values are rounded to about a twentieth of the range of their feature.
    Rules that share a value share its node, so the graph shows which values
    the model moves between. All the arrows have the same width; the colour of
    an arrow is the mining strategy that found the rule, and the rules of each
    strategy are separate arrows (``rules.filter(strategy=...)`` keeps one).
    Arrows towards a later column bend upwards, those back bend downwards.
    Hovering an arrow gives the exact values, the support and the
    implausibility.

    Three sliders thin out the graph, also in the published figure: the
    minimum support of the rules shown; the largest change, the distance
    between the values before and after the switch as a share of the range of
    the feature (categorical rules count as 1, a full change); and the largest
    implausibility of the switched samples.

    Parameters
    ----------
    rules : RuleSet
        Rules to draw.
    min_support : float, default=0
        Minimum support (% of the records) when the figure opens.
    max_change : float, default=1
        Largest change, from 0 to 1, when the figure opens; 1 shows every rule.
    max_implausibility : float, default=1
        Largest implausibility, from 0 to 1, when the figure opens; 1 shows
        every rule.
    mode : {"light", "dark"}, default="light"
        Theme variant.

    Returns
    -------
    plotly.graph_objects.Figure

    Examples
    --------
    >>> viz.show(viz.plot_rule_graph(rules, min_support=2, max_implausibility=0.5))  # doctest: +SKIP
    """
    go = _plotly()
    for name, value in (("max_change", max_change), ("max_implausibility", max_implausibility)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1, got {value!r}")
    ink = INK[mode]
    frame = rules.to_frame()
    if frame.empty:
        figure = go.Figure(layout=dict(template=plotly_template(mode), height=300))
        figure.update_layout(title=_title("All the rules as a graph", "no rules to draw", mode))
        return figure.update_xaxes(visible=False).update_yaxes(visible=False)

    sizes, spans = _switch_sizes(rules, frame)
    supports = frame["support"].to_numpy(dtype=float)
    implausibility = frame["implausibility"].to_numpy(dtype=float)
    present = set(frame["from_class"]) | set(frame["to_class"])
    classes = [c for c in rules.classes if c in present]
    # classes the model predicts but y does not hold get a column too, after those of y
    classes += sorted(present - set(classes), key=str)
    column = {c: i for i, c in enumerate(classes)}
    strategy_colour = strategy_colors(sorted(frame["strategy"].unique(), key=strategy_sort_key), mode)
    numerical = dict(zip(frame["feature"], frame["aggregation"] == "mean", strict=False))
    steps = {feature: _grid_step(span) for feature, span in spans.items()}

    def node(label: Any, feature: str, value: Any) -> tuple:
        """A node: class, feature and value, numerical values on the grid of their feature."""
        if numerical[feature]:
            step = steps[feature]
            return (label, feature, round(float(value) / step) * step)
        return (label, feature, value)

    def text(key: tuple) -> str:
        return _grid_label(key[2], steps[key[1]]) if numerical[key[1]] else format_data_value(key[2])

    ends = [
        (node(r.from_class, r.feature, r.from_value), node(r.to_class, r.feature, r.to_value))
        for r in frame.itertuples(index=False)
    ]
    incident: dict[tuple, list[tuple[int, float, float, float]]] = {}
    for i, pair in enumerate(ends):
        for key in pair:
            incident.setdefault(key, []).append((i, supports[i], sizes[i], implausibility[i]))
    weight = {key: sum(edge[1] for edge in edges) for key, edges in incident.items()}

    # one band per feature, the strongest rule first: raising the minimum support empties the bands from the bottom
    strongest = frame.groupby("feature", sort=False)["support"].agg(["max", "sum"])
    features = strongest.sort_values(["max", "sum"], ascending=False, kind="stable").index
    y: dict[tuple, float] = {}
    ticks, separators, top = [], [], 0.0
    for feature in features:
        columns = {
            c: sorted(
                (key for key in incident if key[0] == c and key[1] == feature),
                key=(lambda k: -k[2]) if numerical[feature] else (lambda k: (-weight[k], str(k[2]))),
            )
            for c in classes
        }
        height = max(1, *(len(keys) for keys in columns.values()))
        for keys in columns.values():
            offset = (height - len(keys)) / 2  # fewer values: centred in the band
            for slot, key in enumerate(keys):
                y[key] = top - offset - slot
        ticks.append((top - (height - 1) / 2, str(feature)))
        top -= height + _BAND_GAP
        separators.append(top + (_BAND_GAP + 1) / 2)
    bottom = top + _BAND_GAP + 1

    labelled: dict[int, list[tuple]] = {}
    copies: dict[tuple, int] = {}  # number of arrows that draw the label of each node
    for key, edges in incident.items():
        chosen = _non_dominated(edges)
        copies[key] = len(chosen)
        for i in chosen:
            labelled.setdefault(i, []).append(key)

    # each slider hides arrows through an attribute of its own, so that they combine: the minimum support through
    # the opacity, the largest change through the visibility, the largest implausibility by moving the arrow to an
    # x axis that shows nothing
    def kept(threshold: float, values: np.ndarray, at_least: bool) -> list[bool]:
        if not at_least and threshold >= 1.0:  # "any": also values above 1 (implausibility is not clipped) and NaN
            return [True] * len(values)
        return [bool(v >= threshold - 1e-9) if at_least else bool(v <= threshold + 1e-9) for v in values]

    rank = {s: r for r, s in enumerate(strategy_colour)}
    t = np.linspace(0.0, 1.0, 25)[:, None]
    last = len(classes) - 1
    strong, near, plausible = (
        kept(min_support, supports, True),
        kept(max_change, sizes, False),
        kept(max_implausibility, implausibility, False),
    )
    figure = go.Figure()
    for i, (row, (start, end)) in enumerate(zip(frame.itertuples(index=False), ends, strict=True)):
        p0, p2 = np.array([column[start[0]], y[start]]), np.array([column[end[0]], y[end]])
        direction = 1.0 if p2[0] > p0[0] else -1.0  # one way above, the way back below
        # the arrows of the strategies apart, so that arrows on top of each other do not look like a wider one
        p1 = (p0 + p2) / 2 + [0.0, direction * (0.3 + _STRATEGY_BEND * rank[row.strategy])]
        curve = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2
        colour = strategy_colour[row.strategy]
        nodes = []  # the two ends, each repeated so that every label is drawn as often (see _LABEL_DRAWS)
        for key in (start, end):
            place = column[key[0]]
            label = text(key) if key in labelled.get(i, []) else ""
            repeat = -(-_LABEL_DRAWS // copies[key]) if label else 1
            where = "middle left" if place == 0 else "middle right" if place == last else "top center"
            nodes.append([(_NODE_SIZE, ink["secondary"], label, where)] * repeat)  # neutral: colour is the strategy
        middle = [(0.0, colour, "", "middle center")] * (len(curve) - 2)
        points = nodes[0] + middle + nodes[1]
        path = np.concatenate(
            [np.repeat(curve[:1], len(nodes[0]), axis=0), curve[1:-1], np.repeat(curve[-1:], len(nodes[1]), axis=0)]
        )
        size, fill, labels, position = (list(values) for values in zip(*points, strict=True))
        change = f"change: {sizes[i]:.0%} of the range" if numerical[row.feature] else "change: another category"
        hover = (
            f"<b>{row.feature}</b> · {rules.class_name(row.from_class)} → {rules.class_name(row.to_class)}"
            f"<br>{rules._rule(pd.Series(row._asdict()))._change('→')}<br>support {supports[i]:.2f}% · "
            f"{row.n_records} records<br>implausibility {row.implausibility:.3f} · {change}<br>strategy {row.strategy}"
        )
        common = dict(
            opacity=1.0 if strong[i] else 0.0, visible=near[i], xaxis="x" if plausible[i] else "x2", showlegend=False
        )
        figure.add_trace(  # the curve, with its two nodes
            go.Scatter(
                x=path[:, 0], y=path[:, 1], mode="lines+markers+text", line=dict(color=colour, width=_ARROW_WIDTH),
                marker=dict(size=size, color=fill, opacity=1.0, line=dict(width=1.5, color=ink["surface"])),
                text=labels, textposition=position, textfont=dict(size=11, color=ink["primary"]),
                hovertext=hover, hoverinfo="text" if strong[i] else "skip", **common,
            )
        )  # fmt: skip
        figure.add_trace(  # its head, turned along the curve
            go.Scatter(
                x=curve[_ARROW_AT - 1 : _ARROW_AT + 1, 0], y=curve[_ARROW_AT - 1 : _ARROW_AT + 1, 1], mode="markers",
                marker=dict(size=[0, _HEAD_SIZE], symbol="arrow", angleref="previous", color=colour,
                            line=dict(width=0)),
                hoverinfo="skip", **common,
            )
        )  # fmt: skip
    arrows = list(range(2 * len(frame)))
    for strategy, colour in strategy_colour.items():  # the legend of the colours, after the arrows: drawn empty
        figure.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="lines", line=dict(color=colour, width=3), name=strategy, hoverinfo="skip"
            )
        )

    def twice(values: list) -> list:
        return [value for value in values for _ in range(2)]  # the curve and the head of each rule

    support_values = _steps(_SUPPORT_STEPS, min_support, lambda s: s < supports.max())
    change_values = _steps(_CHANGE_STEPS, max_change, lambda c: True)
    # quintiles of the implausibility of the rules, rounded up: each step keeps at least its share of the rules
    quintiles = np.ceil(np.quantile(implausibility, [0.2, 0.4, 0.6, 0.8]) * 100) / 100
    implausibility_values = sorted({float(v) for v in quintiles if v < 1.0} | {1.0, float(max_implausibility)})
    support_steps = [
        dict(
            method="restyle",
            label=f"{s:g}%",
            args=[
                {
                    "opacity": twice([1.0 if ok else 0.0 for ok in kept(s, supports, True)]),
                    "hoverinfo": twice(["text" if ok else "skip" for ok in kept(s, supports, True)]),
                },
                arrows,
            ],
        )
        for s in support_values
    ]
    change_steps = [
        dict(
            method="restyle",
            label="any" if c >= 1 else f"{c:.0%}",
            args=[{"visible": twice(kept(c, sizes, False))}, arrows],
        )
        for c in change_values
    ]
    implausibility_steps = [
        dict(
            method="restyle",
            label="any" if m >= 1 else f"{m:.2f}",
            args=[{"xaxis": twice(["x" if ok else "x2" for ok in kept(m, implausibility, False)])}, arrows],
        )
        for m in implausibility_values
    ]
    slider = dict(
        y=0.0, yanchor="top", len=0.3, pad=dict(t=36, b=10), font=dict(size=11, color=ink["secondary"]),
        bgcolor=ink["grid"], bordercolor=ink["axis"], activebgcolor=ink["axis"], tickcolor=ink["muted"],
    )  # fmt: skip
    current = dict(font=dict(size=13, color=ink["primary"]))
    sliders = [
        dict(
            slider, x=0.0, active=support_values.index(float(min_support)), steps=support_steps,
            currentvalue=dict(current, prefix="Minimum support: "),
        ),
        dict(
            slider, x=0.35, active=change_values.index(float(max_change)), steps=change_steps,
            currentvalue=dict(current, prefix="Largest change: "),
        ),
        dict(
            slider, x=0.7, active=implausibility_values.index(float(max_implausibility)), steps=implausibility_steps,
            currentvalue=dict(current, prefix="Largest implausibility: "),
        ),
    ]  # fmt: skip
    span = len(classes) - 1
    for c in classes:
        figure.add_annotation(
            x=column[c], y=0.75, text=f"<b>{rules.class_name(c)}</b>", showarrow=False, yanchor="bottom",
            font=dict(size=13, color=ink["primary"]),
        )  # fmt: skip
    for level in separators[:-1]:
        figure.add_shape(type="line", x0=-0.8, x1=span + 0.8, y0=level, y1=level, line=dict(color=ink["grid"], width=1))
    figure.update_xaxes(visible=False, range=[-0.8, span + 0.8])
    figure.update_yaxes(
        tickvals=[v for v, _ in ticks], ticktext=[name for _, name in ticks], range=[bottom - 0.6, 1.3],
        showgrid=False, ticks="", showline=False, fixedrange=True,
    )  # fmt: skip
    figure.update_layout(
        template=plotly_template(mode),
        title=_title(
            "All the rules as a graph",
            f"{len(frame)} rules · arrows from a value of the starting class to one of the arrival class"
            " · colour: mining strategy",
            mode,
        ),
        # the axis of the arrows hidden by the implausibility slider: far away from anything drawn
        xaxis2=dict(overlaying="x", visible=False, range=[1e6, 1e6 + 1], fixedrange=True),
        height=int(280 + _SLOT * (1.9 - bottom)),
        margin=dict(t=130, b=110, r=30),
        hovermode="closest",
        showlegend=True,
        # a key, not a filter: clicking would fight the change slider, which also sets the visibility of the arrows
        legend=dict(
            orientation="h",
            x=0.0,
            xanchor="left",
            y=1.0,
            yanchor="bottom",
            itemclick=False,
            itemdoubleclick=False,
            title=dict(text="Mining strategy", font=dict(size=12, color=ink["secondary"])),
            font=dict(size=12, color=ink["primary"]),
        ),
        sliders=sliders,
    )
    return figure


def plot_sensitivity(results: pd.DataFrame, mode: Mode = "light") -> Any:
    """One heatmap per metric: its mean for every (s, k), with the value written in each cell.

    Parameters
    ----------
    results : pandas.DataFrame
        Outputs of :func:`ramico.benchmark.run_benchmark` for several values of
        ``n_factuals`` (s) and ``n_counterfactuals`` (k), concatenated.
    mode : {"light", "dark"}, default="light"
        Theme variant.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go = _plotly()
    from plotly.subplots import make_subplots

    from ramico.benchmark import METRICS

    metrics = [m for m in METRICS if m in results.columns]
    titles = {
        "fidelity": "Fidelity (%) ↑",
        "incomprehensibility": "Incomprehensibility ↓",
        "implausibility": "Implausibility ↓",
    }
    figure = make_subplots(
        rows=1, cols=len(metrics), subplot_titles=[titles.get(m, m) for m in metrics], horizontal_spacing=0.07
    )
    for i, metric in enumerate(metrics, 1):
        grouped = results.groupby(["n_factuals", "n_counterfactuals"])[metric]
        mean, std = grouped.mean().unstack(), grouped.std().unstack()
        digits = 1 if metric == "fidelity" else 2 if metric == "incomprehensibility" else 3
        figure.add_trace(
            go.Heatmap(
                z=mean.to_numpy(), x=[str(k) for k in mean.columns], y=[str(s) for s in mean.index],
                text=mean.map(lambda v, d=digits: f"{v:.{d}f}").to_numpy(), texttemplate="%{text}",
                textfont=dict(color=INK[mode]["primary"], size=13), customdata=std.to_numpy(),
                colorscale=sequential_colorscale(stop=7), showscale=False, xgap=2, ygap=2,
                hovertemplate=f"s = %{{y}}, k = %{{x}}<br>{metric} %{{z:.{digits}f}} ± %{{customdata:.{digits}f}}"
                "<extra></extra>",
            ),
            row=1, col=i,
        )  # fmt: skip
        figure.update_xaxes(title_text="k counterfactuals per factual", type="category", row=1, col=i)
        figure.update_yaxes(title_text="s factuals per class" if i == 1 else None, type="category", row=1, col=i)
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(showgrid=False)
    figure.update_annotations(font=dict(size=14, color=INK[mode]["primary"]))
    runs = results.groupby(["n_factuals", "n_counterfactuals"]).size()
    figure.update_layout(
        template=plotly_template(mode),
        title=_title(
            "Sensitivity to the number of factuals and counterfactuals",
            f"mean over {_span(runs)} runs per cell · darker: higher value",
            mode,
        ),
        height=420,
        margin=dict(t=120),
    )
    return figure


#: Plot area of plot_pareto in pixels: its height is fixed, its width follows the figure. The first width is that of
#: the static export (900 px); the labels stay clear also in figures 600 and 1200 px wide, as notebooks draw them.
_PARETO_HEIGHT = 370.0
_PARETO_WIDTHS = (650.0, 350.0, 950.0)
_MARKER_RADIUS = 9.0
_LABEL_HEIGHT, _CHAR_WIDTH = 18.0, 7.5
_STRETCH = [np.array([width / _PARETO_WIDTHS[0], 1.0]) for width in _PARETO_WIDTHS]


def _views(points: np.ndarray) -> list[np.ndarray]:
    """``points``, in pixels of the first width of ``_PARETO_WIDTHS``, at every width (only x stretches)."""
    return [points * stretch for stretch in _STRETCH]


def _crossings(starts: np.ndarray, ends: np.ndarray) -> int:
    """Number of pairs of segments ``starts[i]``-``ends[i]`` that cross."""

    def turn(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return float(np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])))

    count = 0
    for i in range(len(starts)):
        for j in range(i + 1, len(starts)):
            p, q, r, s = starts[i], ends[i], starts[j], ends[j]
            if turn(p, q, r) * turn(p, q, s) < 0 and turn(r, s, p) * turn(r, s, q) < 0:
                count += 1
    return count


def _hits(starts: np.ndarray, ends: np.ndarray, markers: np.ndarray, radius: float) -> int:
    """Number of times a segment ``starts[i]``-``ends[i]`` passes over a marker other than its own start."""
    count = 0
    for a, b in zip(starts, ends, strict=True):
        direction = b - a
        length = float(direction @ direction)
        for point in markers:
            if length == 0 or np.allclose(point, a):
                continue
            t = float(np.clip((point - a) @ direction / length, 0.0, 1.0))
            if np.linalg.norm(a + t * direction - point) < radius:
                count += 1
    return count


def _covers(box: tuple[float, float, float, float], a: np.ndarray, b: np.ndarray) -> bool:
    """Whether the segment a-b passes through the box (x0, x1, y0, y1)."""
    x0, x1, y0, y1 = box
    return any(x0 < x < x1 and y0 < y < y1 for x, y in np.linspace(a, b, 24))


#: Positions of a label next to its marker, in order of preference: plotly anchors and the shift towards them.
_PLACES: dict[str, dict[str, Any]] = {
    "right": dict(xanchor="left", yanchor="middle", xshift=1, yshift=0),
    "left": dict(xanchor="right", yanchor="middle", xshift=-1, yshift=0),
    "above": dict(xanchor="center", yanchor="bottom", xshift=0, yshift=1),
    "below": dict(xanchor="center", yanchor="top", xshift=0, yshift=-1),
}


def _label_width(text: str) -> float:
    return _CHAR_WIDTH * len(text) + 6


def _box(point: np.ndarray, width: float, anchor: dict[str, Any]) -> tuple[float, float, float, float]:
    """Pixel box (x0, x1, y0, y1) of a label of ``width`` written next to ``point`` as ``anchor`` says."""
    x, y = point
    gap = _MARKER_RADIUS + 3
    left = {"left": x + gap, "right": x - gap - width}.get(anchor["xanchor"], x - width / 2)
    bottom = {"bottom": y + gap, "top": y - gap - _LABEL_HEIGHT}.get(anchor["yanchor"], y - _LABEL_HEIGHT / 2)
    return (left, left + width, bottom, bottom + _LABEL_HEIGHT)


def _overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]


def _under(box: tuple[float, float, float, float], points: np.ndarray, own: int | None = None) -> int:
    """Number of markers, other than ``points[own]``, that a label box covers."""
    return sum(
        box[0] - _MARKER_RADIUS < u < box[1] + _MARKER_RADIUS and box[2] - _MARKER_RADIUS < v < box[3] + _MARKER_RADIUS
        for j, (u, v) in enumerate(points)
        if j != own
    )


def _direct_labels(
    views: list[np.ndarray],
    texts: list[str],
    segments: list[list[tuple[np.ndarray, np.ndarray]]],
    boxes: list[list[tuple[float, float, float, float]]],
    candidates: list[int],
) -> dict[int, str]:
    """Place (a key of ``_PLACES``) of the label of each candidate that can be written next to its marker.

    ``views`` are the markers in pixels at each width of ``_PARETO_WIDTHS``, with the ``segments`` and the label
    ``boxes`` to keep clear at that width. A label takes the first place where, at every width, it stays in the
    plot and covers no other label, marker or segment; candidates without such a place are left out.
    """
    placed = [list(view_boxes) for view_boxes in boxes]
    places = {}
    for i in candidates:
        for place, anchor in _PLACES.items():
            shown = [_box(points[i], _label_width(texts[i]), anchor) for points in views]
            free = all(
                box[0] >= 0
                and box[1] <= width
                and box[2] >= 0
                and box[3] <= _PARETO_HEIGHT
                and not any(_overlap(box, other) for other in placed[k])
                and not _under(box, views[k], own=i)
                and not any(_covers(box, a, b) for a, b in segments[k])
                for k, (box, width) in enumerate(zip(shown, _PARETO_WIDTHS, strict=True))
            )
            if free:
                for k, box in enumerate(shown):
                    placed[k].append(box)
                places[i] = place
                break
    return places


def _label_slots(
    points: list[np.ndarray], slots: list[np.ndarray], markers: list[np.ndarray], covered: np.ndarray | None = None
) -> list[int]:
    """Slot of each point: the fewest conflicts, then the shortest leader lines.

    Each argument but ``covered`` holds one view per width, in pixels: the points to label, the slots of their
    labels and all the markers to keep clear; the views differ only by a horizontal stretch. Conflicts are leader
    lines that cross (the same in every view) or pass over a marker (counted in every view), and labels that cover
    one (``covered``: the markers under the label of each slot, over all the views). The lengths are those of the
    first view. There may be more slots than points. Beyond seven points they take the first slots in order of
    height.
    """
    covered = np.zeros(len(slots[0]), dtype=int) if covered is None else covered
    if len(points[0]) > 7:
        order = np.argsort(-points[0][:, 1], kind="stable")
        assignment = [0] * len(points[0])
        for slot, point in enumerate(order):
            assignment[point] = slot
        return assignment
    n, m = len(points[0]), len(slots[0])
    # the conflicts of each leader line (point i to slot s) and of each pair of them, then every assignment at once
    hits = np.array(
        [
            [
                sum(_hits(p[i : i + 1], s[t : t + 1], marks, _MARKER_RADIUS) for p, s, marks in
                    zip(points, slots, markers, strict=True))
                for t in range(m)
            ]
            for i in range(n)
        ]
    )  # fmt: skip
    pairs = list(itertools.combinations(range(n), 2))
    crossing = {
        (i, j): np.array([[_crossings(points[0][[i, j]], slots[0][[s, t]]) for t in range(m)] for s in range(m)])
        for i, j in pairs
    }
    lengths = np.linalg.norm(slots[0][None] - points[0][:, None], axis=-1)
    assignments = np.array(list(itertools.permutations(range(m), n)))
    rows = np.arange(n)
    conflicts = hits[rows, assignments].sum(axis=1) + covered[assignments].sum(axis=1)
    for i, j in pairs:
        conflicts += crossing[i, j][assignments[:, i], assignments[:, j]]
    best = np.lexsort((lengths[rows, assignments].sum(axis=1), conflicts))[0]  # stable: the first of the ties
    return [int(slot) for slot in assignments[best]]


def _leader_columns(points: np.ndarray, texts: list[str], group: list[int]) -> dict[int, tuple[np.ndarray, bool]]:
    """End of the leader line of each point of ``group`` and whether its label lies right of it.

    ``points`` are in pixels of the first width of ``_PARETO_WIDTHS``, and so are the ends. The labels form
    columns 26 px apart, 60 px right and left of the group where they fit in the plot at every width; each takes
    the slot whose leader line crosses no other and passes over no marker, if there is one.
    """
    crowd = points[group]
    widest = max(_label_width(texts[i]) for i in group)
    n = len(group)
    slots_y = crowd[:, 1].mean() + (np.arange(n)[::-1] - (n - 1) / 2) * 26
    slots_y += max(0.0, 13 - slots_y.min()) - max(0.0, slots_y.max() - (_PARETO_HEIGHT - 13))  # inside the plot

    def fits(x: float, right: bool) -> bool:
        place = _PLACES["right" if right else "left"]
        boxes = [_box(np.array([x, 0.0]) * stretch, widest, place) for stretch in _STRETCH]
        return all(box[0] >= 0 and box[1] <= width for box, width in zip(boxes, _PARETO_WIDTHS, strict=True))

    sides = [(True, crowd[:, 0].max() + 60), (False, crowd[:, 0].min() - 60)]
    sides = [(right, x) for right, x in sides if fits(x, right)]
    sides = sides[: 1 if n > 5 else 2] or [(True, crowd[:, 0].max() + 60)]
    slots = np.array([(x, y) for _, x in sides for y in slots_y])
    rights = [right for right, _ in sides for _ in slots_y]
    slot_views, markers = _views(slots), _views(points)
    covered = np.array(  # markers under the label of each slot, at every width
        [
            sum(
                _under(_box(view[s], widest, _PLACES["right" if right else "left"]), marks)
                for view, marks in zip(slot_views, markers, strict=True)
            )
            for s, right in enumerate(rights)
        ]
    )
    assignment = _label_slots(_views(crowd), slot_views, markers, covered)
    return {i: (slots[slot], rights[slot]) for i, slot in zip(group, assignment, strict=True)}


def _label_means(
    figure: Any, means: pd.DataFrame, front: pd.DataFrame, x_range: list[float], y_range: list[float], mode: Mode
) -> None:
    """Name the mean of each strategy: beside its marker when it stands apart, else in a column with a leader line.

    A label beside a marker that nearly touches another would be ambiguous:
    such markers, and those with no room around them, get leader lines. The
    places are chosen so that the labels stay clear at every width of
    ``_PARETO_WIDTHS``, since an interactive figure takes the width of its cell.
    """
    origin = np.array([x_range[0], y_range[0]])
    scale = np.array([_PARETO_WIDTHS[0] / (x_range[1] - x_range[0]), _PARETO_HEIGHT / (y_range[1] - y_range[0])])
    columns = ["incomprehensibility", "fidelity"]
    points = (means[columns].to_numpy(float) - origin) * scale  # pixels at the first width, y up
    views = _views(points)
    front_views = [list(itertools.pairwise(view)) for view in _views((front[columns].to_numpy(float) - origin) * scale)]
    texts = [str(strategy) for strategy in means.index]
    n = len(points)
    gaps = np.min([np.linalg.norm(view[:, None] - view[None], axis=-1) for view in views], axis=0)
    np.fill_diagonal(gaps, np.inf)
    group = [i for i in range(n) if gaps[i].min() < 2.5 * _MARKER_RADIUS]
    while True:  # the leader lines first, then the direct labels around them; a marker without room joins the group
        leaders = _leader_columns(points, texts, group) if group else {}
        boxes, segments = [], []
        for view, stretch, front_view in zip(views, _STRETCH, front_views, strict=True):
            ends = {i: end * stretch for i, (end, _) in leaders.items()}
            boxes.append(
                [_box(ends[i], _label_width(texts[i]), _PLACES["right" if right else "left"]) for i, (_, right) in
                 leaders.items()]
            )  # fmt: skip
            segments.append(front_view + [(view[i], ends[i]) for i in leaders])
        rest = [i for i in range(n) if i not in leaders]
        places = _direct_labels(views, texts, segments, boxes, rest)
        if len(places) == len(rest):
            break
        group += [i for i in rest if i not in places]
    ink = INK[mode]
    common = dict(font=dict(color=ink["primary"]), bgcolor=ink["surface"], borderpad=2)
    gap = _MARKER_RADIUS + 3
    for i, (x, y) in enumerate(means[columns].to_numpy(float)):
        if i in places:
            anchor = _PLACES[places[i]]
            figure.add_annotation(
                x=x, y=y, text=texts[i], showarrow=False, xanchor=anchor["xanchor"], yanchor=anchor["yanchor"],
                xshift=gap * anchor["xshift"], yshift=gap * anchor["yshift"], **common,
            )  # fmt: skip
        else:
            end, right = leaders[i]
            ax, ay = end / scale + origin
            figure.add_annotation(
                x=x, y=y, text=texts[i], ax=ax, ay=ay, axref="x", ayref="y", xanchor="left" if right else "right",
                showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor=ink["muted"], standoff=_MARKER_RADIUS, **common,
            )  # fmt: skip


def _pareto_front(points: pd.DataFrame) -> pd.DataFrame:
    """Strategies not dominated in (higher fidelity, lower incomprehensibility)."""
    keep = []
    for i, p in points.iterrows():
        dominated = (
            (points["fidelity"] >= p["fidelity"])
            & (points["incomprehensibility"] <= p["incomprehensibility"])
            & ((points["fidelity"] > p["fidelity"]) | (points["incomprehensibility"] < p["incomprehensibility"]))
        ).any()
        if not dominated:
            keep.append(i)
    return points.loc[keep].sort_values("incomprehensibility")


def plot_pareto(results: pd.DataFrame, mode: Mode = "light", title: str = "Fidelity against rule complexity") -> Any:
    """Fidelity against incomprehensibility per strategy, coloured by implausibility.

    Small markers are single runs, large labelled markers the means of each
    strategy; the dashed line joins the strategies on the Pareto front (no
    other strategy is both more faithful and more concise).

    Parameters
    ----------
    results : pandas.DataFrame
        Output of :func:`ramico.benchmark.run_benchmark`.
    mode : {"light", "dark"}, default="light"
        Theme variant.
    title : str, default="Fidelity against rule complexity"
        Title of the chart, above the subtitle that gives the number of runs.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go = _plotly()
    metrics = ["fidelity", "incomprehensibility", "implausibility"]
    means = results.groupby("strategy")[metrics].mean()
    strategies = sorted(means.index, key=strategy_sort_key)
    low, high = float(results["implausibility"].min()), float(results["implausibility"].max())
    coloraxis = dict(
        colorscale=sequential_colorscale(start=2),  # the lightest steps vanish on the background
        cmin=low,
        cmax=high if high > low else low + 1e-9,
        colorbar=_implausibility_bar(mode),
    )
    neutral = INK[mode]["secondary"]
    outline = dict(width=1, color=INK[mode]["muted"])
    symbols = {strategy: _SYMBOLS[i % len(_SYMBOLS)] for i, strategy in enumerate(strategies)}
    figure = go.Figure()
    for strategy in strategies:
        runs = results[results["strategy"] == strategy]
        symbol = symbols[strategy]
        figure.add_trace(
            go.Scatter(
                x=runs["incomprehensibility"],
                y=runs["fidelity"],
                mode="markers",
                marker=dict(
                    symbol=symbol,
                    size=8,
                    color=runs["implausibility"],
                    coloraxis="coloraxis",
                    opacity=0.6,
                    line=outline,
                ),
                name=strategy,
                legendgroup=strategy,
                showlegend=False,
                hovertemplate=f"<b>{strategy}</b> run<br>fidelity %{{y:.1f}}%<br>"
                "incomprehensibility %{x:.2f}<br>implausibility %{marker.color:.3f}<extra></extra>",
            )
        )
    # the front under the means, dashed so that it never reads as the leader line of a label
    front = _pareto_front(means.reset_index())
    if len(front) > 1:
        figure.add_trace(
            go.Scatter(
                x=front["incomprehensibility"],
                y=front["fidelity"],
                mode="lines",
                line=dict(color=INK[mode]["primary"], width=1.5, dash="dash"),
                name="Pareto front",
                legendrank=2000,
                hoverinfo="skip",
            )
        )
    for strategy in strategies:
        symbol = symbols[strategy]
        mean = means.loc[strategy]
        figure.add_trace(
            go.Scatter(
                x=[mean["incomprehensibility"]],
                y=[mean["fidelity"]],
                mode="markers",
                marker=dict(
                    symbol=symbol,
                    size=15,
                    color=[mean["implausibility"]],
                    coloraxis="coloraxis",
                    line=dict(width=1.5, color=INK[mode]["primary"]),
                ),
                name=strategy,
                legendgroup=strategy,
                showlegend=False,
                hovertemplate=f"<b>{strategy}</b> mean<br>fidelity %{{y:.1f}}%<br>"
                "incomprehensibility %{x:.2f}<br>implausibility %{marker.color:.3f}<extra></extra>",
            )
        )
        # legend entry: the symbol in neutral ink (the colour encodes implausibility)
        figure.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=strategy,
                legendgroup=strategy,
                marker=dict(symbol=symbol, size=11, color=neutral),
            )
        )
    # explicit ranges, so that the labels can be placed in data coordinates; the legend below the axis title, where
    # it can wrap on narrow figures without covering the subtitle
    top, bottom = 100, 125
    x_all = np.concatenate([results["incomprehensibility"].to_numpy(float), means["incomprehensibility"].to_numpy()])
    y_all = np.concatenate([results["fidelity"].to_numpy(float), means["fidelity"].to_numpy()])
    x_range = _padded(x_all.min(), x_all.max())
    y_range = _padded(y_all.min(), y_all.max())
    _label_means(figure, means, front, x_range, y_range, mode)
    figure.update_layout(
        template=plotly_template(mode),
        coloraxis=coloraxis,
        title=_title(
            title,
            f"{int(results['seed'].nunique()) if 'seed' in results else len(results)} runs per strategy · "
            "colour: implausibility (lower is more realistic)",
            mode,
        ),
        xaxis=dict(title="Incomprehensibility (conditions per rule chain, lower is better)", range=x_range),
        yaxis=dict(title="Fidelity (%, higher is better)", range=y_range),
        legend=dict(orientation="h", yanchor="top", y=-62 / _PARETO_HEIGHT, x=0, title=dict(text="Strategy: ")),
        margin=dict(t=top, b=bottom),
        height=top + _PARETO_HEIGHT + bottom,
    )
    return figure


def _padded(low: float, high: float, share: float = 0.06) -> list[float]:
    """The interval [low, high] widened on both sides by ``share`` of its length (by 1 if it is empty)."""
    pad = share * (high - low) if high > low else 1.0
    return [float(low - pad), float(high + pad)]


def _span(counts: pd.Series) -> str:
    """ "n" or "n-m": the range of the numbers of runs."""
    low, high = int(counts.min()), int(counts.max())
    return str(low) if low == high else f"{low}-{high}"
