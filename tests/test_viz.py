"""Tests for ramico.viz (theme, tables, plots) and the RuleSet display methods."""

from __future__ import annotations

import base64
import io
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent / "golden"))
import _harness as h

from ramico import RaMiCo
from ramico.benchmark import balanced_split, run_benchmark
from ramico.viz import theme


class FittedToy(h.ToyModel):
    def fit(self, X, y):
        return self


@pytest.fixture(scope="module")
def rules():
    X, y = h.make_dataset(h.Case("viz", "mixed", 2, "random", n_samples=160))
    return RaMiCo(h.ToyModel(2), n_factuals=5, n_counterfactuals=4, random_state=1, isolation_forest="native").fit(
        X, y, categorical_features=["color", "segment", "flag"], class_names={0: "rejected", 1: "accepted"}
    )


@pytest.fixture(scope="module")
def results():
    X, y = h.make_dataset(h.Case("viz", "mixed", 2, "random", n_samples=160))
    return run_benchmark(
        lambda seed: FittedToy(2),
        X,
        y,
        strategies=["ensemble", "boundary", "random"],
        seeds=[1, 2, 3],
        n_factuals=4,
        n_counterfactuals=3,
        split=balanced_split(0.1),
        k=3,
        isolation_forest="native",
        categorical_features=["color", "segment", "flag"],
        progress=False,
    )


def test_strategy_colours_follow_the_strategy():
    colours = theme.strategy_colors(["random", "ensemble", "mine"])
    assert colours["ensemble"] == theme.CATEGORICAL["light"][0]
    assert colours["random"] == theme.CATEGORICAL["light"][3]
    assert colours["mine"] not in {colours["ensemble"], colours["random"]}
    assert theme.strategy_colors(["random"])["random"] == colours["random"]  # never recoloured
    assert theme.strategy_colors(["ensemble"], mode="dark")["ensemble"] == theme.CATEGORICAL["dark"][0]
    components = theme.strategy_colors(["prototype", "nearest-boundary", "random-pairs"])
    assert components["random-pairs"] == theme.CATEGORICAL["light"][6]  # not a second green
    with pytest.raises(ValueError):
        theme.strategy_colors(["ensemble"], mode="sepia")


def test_sequential_scale():
    assert theme.sequential_color(0) == theme.SEQUENTIAL[0]
    assert theme.sequential_color(1) == theme.SEQUENTIAL[-1]
    assert theme.sequential_color(5, 0, 1) == theme.SEQUENTIAL[-1]
    scale = theme.sequential_colorscale()
    assert scale[0][0] == 0 and scale[-1][0] == 1


def test_theme_applies_to_plotly_and_matplotlib():
    pio = pytest.importorskip("plotly.io")
    mpl = pytest.importorskip("matplotlib")
    theme.apply("dark")
    assert pio.templates.default == "ramico_dark"
    assert mpl.rcParams["axes.facecolor"] == theme.INK["dark"]["surface"]
    theme.apply()
    assert pio.templates.default == "ramico"
    assert theme.matplotlib_rc("light")["lines.linewidth"] == 2


def test_rules_table_and_plot(rules):
    pytest.importorskip("great_tables")
    pytest.importorskip("plotly")
    html = rules.to_table(max_rules=3).as_raw_html()
    assert "Counterfactual rules" in html and "rejected → accepted" in html
    assert "Showing up to 3 rules" in html
    from ramico import viz

    figure = rules.plot(min_support=2.5, max_change=0.3, max_implausibility=0.45, mode="dark")
    expected = viz.plot_rule_graph(rules, min_support=2.5, max_change=0.3, max_implausibility=0.45, mode="dark")
    assert figure.to_dict() == expected.to_dict()
    assert not hasattr(viz, "plot_rules")


def test_comparison_table_and_pareto(results):
    pytest.importorskip("great_tables")
    pytest.importorskip("plotly")
    pytest.importorskip("scipy")
    from ramico import viz

    html = viz.comparison_table(results).as_raw_html()
    assert "Mining strategies compared" in html and "Mann-Whitney" in html
    figure = viz.plot_pareto(results)
    names = {trace.name for trace in figure.data}
    assert {"ensemble", "boundary", "random"} <= names
    assert sorted(annotation.text for annotation in figure.layout.annotations) == sorted(results["strategy"].unique())


def test_pareto_labels_are_written_beside_spread_means():
    pytest.importorskip("plotly")
    from ramico import viz

    centres = {"ensemble": (3.1, 78), "boundary": (3.0, 47), "prototype": (2.7, 73), "random": (3.3, 70)}
    results = pd.DataFrame(
        [
            {"seed": seed, "strategy": name, "incomprehensibility": x + 0.05 * seed, "fidelity": y + seed,
             "implausibility": 0.3}
            for name, (x, y) in centres.items() for seed in range(3)
        ]
    )  # fmt: skip
    annotations = viz.plot_pareto(results).layout.annotations
    assert len(annotations) == 4 and not any(annotation.showarrow for annotation in annotations)
    figure = viz.plot_pareto(results, title="MLP: fidelity against rule complexity")
    front = next(trace for trace in figure.data if trace.name == "Pareto front")
    assert front.line.dash == "dash"
    assert figure.layout.legend.yanchor == "top" and figure.layout.legend.y < 0  # below the plot, clear of the title
    assert figure.layout.title.text.startswith("MLP: fidelity") and "3 runs per strategy" in figure.layout.title.text


def test_pareto_labels_of_close_means_have_leader_lines_that_do_not_cross():
    pytest.importorskip("plotly")
    from ramico import viz
    from ramico.viz.plots import _crossings

    # means a few pixels apart: no room to write the labels beside them
    centres = {"ensemble": (2.10, 60.0), "boundary": (2.11, 60.1), "prototype": (2.10, 60.2), "random": (2.11, 59.9)}
    offsets = [(-0.6, -10), (0.6, 10), (0.6, -10), (-0.6, 10)]  # the means stay at the centres
    results = pd.DataFrame(
        [
            {"seed": seed, "strategy": name, "incomprehensibility": x + dx, "fidelity": y + dy, "implausibility": 0.5}
            for name, (x, y) in centres.items() for seed, (dx, dy) in enumerate(offsets)
        ]
    )  # fmt: skip
    annotations = viz.plot_pareto(results).layout.annotations
    assert all(annotation.showarrow and annotation.standoff > 0 for annotation in annotations)
    starts = np.array([(a.x, a.y) for a in annotations])
    ends = np.array([(a.ax, a.ay) for a in annotations])
    scale = np.array([1 / np.ptp(starts[:, 0]), 1 / np.ptp(starts[:, 1])])  # crossings do not depend on the scale
    assert _crossings(starts * scale, ends * scale) == 0


def test_label_slots_avoid_crossings():
    from ramico.viz.plots import _crossings, _label_slots

    points = np.array([[0.0, 0.0], [0.0, 10.0]])
    slots = np.array([[50.0, 10.0], [50.0, 0.0]])
    assignment = _label_slots([points], [slots], [points])
    assert assignment == [1, 0] and _crossings(points, slots[assignment]) == 0


def test_pareto_labels_mix_leader_lines_and_direct_labels():
    pytest.importorskip("plotly")
    from ramico import viz

    # random and prototype almost touch; ensemble and boundary stand apart
    centres = {"ensemble": (2.73, 83.2), "boundary": (2.48, 53.4), "prototype": (2.60, 84.3), "random": (2.58, 84.5)}
    offsets = [(-0.3, -10), (0.3, 10), (0.3, -10), (-0.3, 10)]
    results = pd.DataFrame(
        [
            {"seed": seed, "strategy": name, "incomprehensibility": x + dx, "fidelity": y + dy, "implausibility": 0.3}
            for name, (x, y) in centres.items() for seed, (dx, dy) in enumerate(offsets)
        ]
    )  # fmt: skip
    arrows = {a.text: bool(a.showarrow) for a in viz.plot_pareto(results).layout.annotations}
    assert arrows == {"ensemble": False, "boundary": False, "prototype": True, "random": True}


class FakeFigure:
    """Records what show() asks of a plotly figure."""

    def __init__(self, fail: bool = False) -> None:
        self.layout = types.SimpleNamespace(width=None)
        self.data: list = []
        self.fail = fail
        self.renderers: list[str] = []
        self.export: dict = {}

    def show(self, renderer):
        self.renderers.append(renderer)

    def to_dict(self):
        return {"data": self.data, "layout": {}}

    def to_image(self, **kwargs):
        if self.fail:
            raise RuntimeError("no browser for kaleido")
        self.export = kwargs
        return b"\x89PNG\r\n\x1a\n" + bytes(16)


@pytest.fixture
def displayed(monkeypatch):
    pytest.importorskip("IPython")
    outputs = []
    monkeypatch.setattr("IPython.display.display", lambda *objects, **options: outputs.append((objects, options)))
    return outputs


def test_show_puts_the_figure_and_its_png_in_one_output(displayed):
    pytest.importorskip("plotly")
    from ramico import viz

    figure = FakeFigure()
    viz.show(figure, static=True)
    assert figure.renderers == []  # no second, interactive-only output
    assert figure.export == {"format": "png", "width": 900, "scale": 1.5}
    [((bundle,), options)] = displayed
    assert options == {"raw": True}
    assert set(bundle) == {"application/vnd.plotly.v1+json", "image/png"}
    assert base64.b64decode(bundle["image/png"]).startswith(b"\x89PNG")


def test_show_without_static_copy(displayed, caplog):
    from ramico import viz

    interactive, failing = FakeFigure(), FakeFigure(fail=True)
    viz.show(interactive, static=False)
    viz.show(failing, static=True)
    assert displayed == [] and "No static copy of the figure" in caplog.text
    assert interactive.renderers == failing.renderers == ["plotly_mimetype+notebook_connected"]  # plotly.js from a CDN


def test_rule_plot_draws_a_matrix_for_categorical_values(rules):
    pytest.importorskip("plotly")
    from ramico import viz

    position = next(i for i, rule in enumerate(rules) if rule.feature in ("color", "segment"))
    figure = viz.plot_rule(rules, position)
    assert figure.data[0].type == "heatmap"
    counts = np.nan_to_num(np.asarray(figure.data[0].customdata, dtype=float))
    assert counts.sum() == rules[position].n_records
    assert len(figure.layout.annotations) == np.count_nonzero(counts)  # one label per non-empty cell
    assert "→" in figure.layout.title.text


def test_rule_plot_draws_dots_for_numerical_values(rules):
    pytest.importorskip("plotly")
    from ramico import viz

    rule = next(rule for rule in rules if rule.feature in ("age", "score"))
    figure = viz.plot_rule(rules, rule)
    assert [trace.type for trace in figure.data] == ["scatter", "scatter"]
    assert sum(len(trace.x) for trace in figure.data) == 2 * rule.n_records


def test_rule_plot_shades_the_interval_of_numerical_rules():
    pytest.importorskip("plotly")
    from ramico import viz

    X, y = h.make_dataset(h.Case("viz", "numeric", 2, "random", n_samples=120))
    explainer = RaMiCo(h.ToyModel(2), "random", n_factuals=5, n_counterfactuals=4, random_state=1)
    explainer.isolation_forest = "native"
    numeric = explainer.fit(X, y)
    assert numeric[0].aggregation == "mean"
    shapes = [shape.type for shape in viz.plot_rule(numeric, 0).layout.shapes]
    assert shapes.count("rect") == 2 and shapes.count("line") == 2  # interval and mean, before and after


def test_static_copies_follow_the_environment(displayed, monkeypatch):
    from ramico import viz

    pytest.importorskip("plotly")
    monkeypatch.delenv(viz.plots.STATIC_ENV, raising=False)
    viz.show(FakeFigure())  # interactive session: the figure alone
    assert displayed == []
    monkeypatch.setenv(viz.plots.STATIC_ENV, "1")  # publication run
    viz.show(FakeFigure())
    assert len(displayed) == 1


def _is_shown(trace):
    """Shown under the three sliders: visible (change), opaque (support) and on the main x axis (implausibility)."""
    return trace.visible is not False and trace.opacity > 0 and trace.xaxis in (None, "x")


def _arrows(figure):
    """The curves and the heads of the arrows, one of each per rule (the legend entries left out)."""
    traces = [trace for trace in figure.data if trace.showlegend is False]
    return traces[::2], traces[1::2]


def _shown(figure):
    """Positions of the rules whose arrows are shown."""
    return [i for i, curve in enumerate(_arrows(figure)[0]) if _is_shown(curve)]


def _labelled_nodes(figure, rules_shown):
    """The nodes whose label is drawn by a shown arrow, and the ends of the shown arrows."""
    curves = _arrows(figure)[0]
    labelled, ends = set(), set()
    for i in rules_shown:
        curve = curves[i]
        ends |= {(curve.x[0], curve.y[0]), (curve.x[-1], curve.y[-1])}
        labelled |= {(x, y) for x, y, text in zip(curve.x, curve.y, curve.text, strict=True) if text}
    return labelled, ends


def test_switch_sizes(rules):
    from ramico.viz.plots import _switch_sizes

    frame, records = rules.to_frame(), rules.records
    sizes, spans = _switch_sizes(rules, frame)
    assert len(sizes) == len(frame) and ((sizes >= 0) & (sizes <= 1)).all()
    for i, row in frame.iterrows():
        if row["aggregation"] == "mode":
            assert sizes[i] == float(row["from_value"] != row["to_value"])
            continue
        switched = records.loc[records["feature"] == row["feature"], "to_value"]
        values = pd.to_numeric(pd.concat([records["factual." + row["feature"]], switched]))
        assert spans[row["feature"]] == pytest.approx(values.max() - values.min())
        expected = min(abs(row["to_value"] - row["from_value"]) / (values.max() - values.min()), 1.0)
        assert sizes[i] == pytest.approx(expected)


def test_rule_graph_draws_one_arrow_per_rule(rules):
    pytest.importorskip("plotly")
    from ramico import viz
    from ramico.viz.plots import _ARROW_WIDTH, _HEAD_SIZE, _LABEL_DRAWS

    frame = rules.to_frame()
    figure = viz.plot_rule_graph(rules)
    strategies = sorted(frame["strategy"].unique(), key=theme.strategy_sort_key)
    assert len(figure.data) == 2 * len(frame) + len(strategies)  # a curve and a head per rule, a legend entry each
    assert figure.data[2 * len(frame)].showlegend is None  # the legend entries come after every arrow
    assert _shown(figure) == list(range(len(frame)))
    curves, heads = _arrows(figure)
    assert all(head.marker.symbol == "arrow" and head.marker.angleref == "previous" for head in heads)
    assert {curve.line.width for curve in curves} == {_ARROW_WIDTH}  # the same width whatever the support
    assert {head.marker.size[-1] for head in heads} == {_HEAD_SIZE}
    # the colour of an arrow is the colour of its strategy, given by the legend; the nodes are neutral
    colours = theme.strategy_colors(strategies)
    assert [curve.line.color for curve in curves] == [colours[s] for s in frame["strategy"]]
    assert [head.marker.color for head in heads] == [colours[s] for s in frame["strategy"]]
    assert {curve.marker.color[0] for curve in curves} == {theme.INK["light"]["secondary"]}
    legend = figure.data[2 * len(frame) :]
    assert [(entry.name, entry.line.color) for entry in legend] == list(colours.items())
    assert figure.layout.legend.itemclick is False  # a key only: clicking would fight the change slider
    assert "colour: mining strategy" in figure.layout.title.text and "implausibility" not in figure.layout.title.text
    assert all(f"strategy {s}" in curve.hovertext for curve, s in zip(curves, frame["strategy"], strict=True))
    assert [a.text for a in figure.layout.annotations] == [f"<b>{rules.class_name(c)}</b>" for c in rules.classes]
    strongest = frame.groupby("feature", sort=False)["support"].agg(["max", "sum"])
    order = strongest.sort_values(["max", "sum"], ascending=False, kind="stable").index
    assert list(figure.layout.yaxis.ticktext) == list(order)  # the bands, strongest rule first
    # rules that share a value share its node: each node is drawn at one place, and labelled
    labelled, ends = _labelled_nodes(figure, range(len(frame)))
    assert labelled == ends
    draws = {}
    for curve in curves:
        for x, y, text in zip(curve.x, curve.y, curve.text, strict=True):
            if text:
                draws[(x, y, text)] = draws.get((x, y, text), 0) + 1
    assert len({(x, y) for x, y, _ in draws}) == len(draws)  # one label per node
    assert min(draws.values()) >= _LABEL_DRAWS  # each label drawn as often: none looks bolder


def test_rule_graph_sliders_filter_arrows_and_nodes(rules):
    pytest.importorskip("plotly")
    from ramico import viz
    from ramico.viz.plots import _switch_sizes

    frame = rules.to_frame()
    supports, implausibility = frame["support"].to_numpy(), frame["implausibility"].to_numpy()
    sizes, _ = _switch_sizes(rules, frame)
    figure = viz.plot_rule_graph(rules)
    support_slider, change_slider, implausibility_slider = figure.layout.sliders
    assert support_slider.steps[support_slider.active].label == "0%"
    assert change_slider.steps[change_slider.active].label == "any"
    assert implausibility_slider.steps[implausibility_slider.active].label == "any"
    assert figure.layout.xaxis2.visible is False and figure.layout.xaxis2.range[0] > 1e5  # where hidden arrows go
    assert 3 <= len(implausibility_slider.steps) <= 5  # quintiles of the implausibility of the rules, and any
    for support_step in support_slider.steps:
        figure.plotly_restyle(*support_step.args)
        for change_step in change_slider.steps:
            figure.plotly_restyle(*change_step.args)
            for implausibility_step in implausibility_slider.steps:
                figure.plotly_restyle(*implausibility_step.args)
                low = float(support_step.label.rstrip("%"))
                high = 1.0 if change_step.label == "any" else float(change_step.label.rstrip("%")) / 100
                worst = 1.0 if implausibility_step.label == "any" else float(implausibility_step.label)
                expected = [
                    i
                    for i in range(len(frame))
                    if supports[i] >= low - 1e-9 and sizes[i] <= high + 1e-9 and implausibility[i] <= worst + 1e-9
                ]
                state = (support_step.label, change_step.label, implausibility_step.label)
                assert _shown(figure) == expected, state
                assert [i for i, head in enumerate(_arrows(figure)[1]) if _is_shown(head)] == expected, state
                labelled, ends = _labelled_nodes(figure, expected)
                assert labelled == ends, state  # a node keeps its label while one of its arrows is shown
            curves = _arrows(figure)[0]
            hidden = [curves[i].hoverinfo for i in range(len(frame)) if supports[i] < low - 1e-9]
            assert set(hidden) <= {"skip"}  # transparent arrows give no tooltip


def test_rule_graph_gives_a_column_to_classes_missing_from_y():
    pytest.importorskip("plotly")
    from ramico import viz

    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.uniform(0, 3, 150), "b": rng.uniform(0, 3, 150)})

    def model(frame):  # class 2 only where "a" is highest: a class the labels do not hold
        a, b = frame["a"].to_numpy(dtype=float), frame["b"].to_numpy(dtype=float)
        return np.where(a > 2.9, 2, (a + b > 3).astype(int))

    y = np.minimum(model(X), 1)
    rules = RaMiCo(model, "random", n_factuals=15, n_counterfactuals=15, random_state=0, isolation_forest="native").fit(
        X, y
    )
    assert 2 in set(rules.to_frame()["to_class"]) and 2 not in rules.classes
    assert _shown(viz.plot_rule_graph(rules)) == list(range(len(rules)))


@pytest.mark.parametrize("value", [1.05, float("nan")])
def test_rule_graph_shows_every_rule_at_any_implausibility(rules, value):
    pytest.importorskip("plotly")
    from ramico import RuleSet, viz

    # implausibility is normalised by the training scores, not clipped: a rule can go above 1
    internal = rules._rules.copy()
    internal.loc[0, "anomaly_score_mean"] = value
    tweaked = RuleSet(internal, rules._records, rules._features, rules.metadata)
    assert _shown(viz.plot_rule_graph(tweaked)) == list(range(len(tweaked)))


def test_ruleset_plot_takes_keyword_arguments(rules):
    pytest.importorskip("plotly")
    with pytest.raises(TypeError):
        rules.plot(10)  # the old plot(top=...) must not silently become a minimum support


def test_rule_graph_opens_with_the_given_thresholds(rules):
    pytest.importorskip("plotly")
    from ramico import viz
    from ramico.viz.plots import _switch_sizes

    frame = rules.to_frame()
    sizes, _ = _switch_sizes(rules, frame)
    figure = viz.plot_rule_graph(rules, min_support=2.5, max_change=0.3, max_implausibility=0.45, mode="dark")
    support_slider, change_slider, implausibility_slider = figure.layout.sliders
    assert support_slider.steps[support_slider.active].label == "2.5%"
    assert change_slider.steps[change_slider.active].label == "30%"
    assert implausibility_slider.steps[implausibility_slider.active].label == "0.45"
    expected = [
        i
        for i, (s, z, m) in enumerate(zip(frame["support"], sizes, frame["implausibility"], strict=True))
        if s >= 2.5 and z <= 0.3 and m <= 0.45
    ]
    assert _shown(figure) == expected
    assert figure.layout.template.layout.paper_bgcolor == theme.INK["dark"]["surface"]
    for name in ("max_change", "max_implausibility"):
        with pytest.raises(ValueError, match=name):
            viz.plot_rule_graph(rules, **{name: 2})
    empty = viz.plot_rule_graph(rules.filter(feature="no such feature"))
    assert len(empty.data) == 0 and "no rules" in empty.layout.title.text


def test_rule_graph_rounds_numerical_values_to_a_grid():
    from ramico.viz.plots import _grid_label, _grid_step

    assert _grid_step(438.0) == 20.0 and _grid_step(3.0) == 0.1 and _grid_step(100.0) == 5.0
    assert _grid_step(6.2) == 0.2 and _grid_step(99.0) == 2.0  # never above a twentieth of the range
    assert _grid_step(0.0) == 1.0  # a constant feature
    assert _grid_label(1.4000000000000001, 0.1) == "1.4" and _grid_label(280.0, 20.0) == "280"
    assert _grid_label(0.05, 0.05) == "0.05" and _grid_label(45000.0, 5000.0) == "45,000"


def test_static_copies_are_compressed():
    Image = pytest.importorskip("PIL.Image")  # Pillow comes with matplotlib, an optional dependency

    from ramico.viz.plots import _compact_png

    image, buffer = Image.new("RGB", (400, 300)), io.BytesIO()
    for x in range(400):  # a gradient: many colours
        for y in range(0, 300, 30):
            image.putpixel((x, y), (x % 256, y % 256, 120))
    image.save(buffer, "PNG")
    compact = _compact_png(buffer.getvalue())
    assert len(compact) <= len(buffer.getvalue()) and Image.open(io.BytesIO(compact)).size == (400, 300)
    assert _compact_png(b"not an image") == b"not an image"


def test_sensitivity_table_and_heatmaps():
    pytest.importorskip("great_tables")
    pytest.importorskip("plotly")
    from ramico import viz

    rng = np.random.default_rng(0)
    results = pd.DataFrame(
        [
            {"seed": seed, "strategy": "ensemble", "n_factuals": s, "n_counterfactuals": k,
             "fidelity": 50 + s / 10 + rng.normal(), "incomprehensibility": 2 + k / 50, "implausibility": 0.5}
            for s in (10, 25) for k in (5, 10, 20) for seed in range(3)
        ]
    )  # fmt: skip
    html = viz.sensitivity_table(results).as_raw_html()
    assert "Sensitivity to s and k" in html and "3 runs per cell" in html and html.count("±") >= 18
    figure = viz.plot_sensitivity(results)
    assert [trace.type for trace in figure.data] == ["heatmap"] * 3
    assert figure.data[0].z.shape == (2, 3)  # s by k


def test_comparison_table_by_model(results):
    pytest.importorskip("great_tables")
    pytest.importorskip("scipy")
    from ramico import viz

    both = pd.concat([results.assign(model="MLP"), results.assign(model="XGBoost")], ignore_index=True)
    html = viz.comparison_table(both, by="model").as_raw_html()
    assert "MLP" in html and "XGBoost" in html and "within each metric and model" in html
    assert "3 runs per strategy and model" in html
    assert html.count("±") == 2 * 3 * 3 + 1  # two models × three metrics × three strategies, and the subtitle
