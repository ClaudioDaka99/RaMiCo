"""The single visual theme of RaMiCo: palette, plotly templates, matplotlib style.

Colours come from a reference palette validated for colour-vision
deficiencies, separately for light and dark backgrounds:

* categorical slots, assigned to entities in a fixed order (a strategy keeps
  its colour whatever else is plotted);
* one sequential blue ramp for magnitudes such as implausibility;
* text and chrome colours; text never takes a series colour.

Call :func:`apply` once at the top of a notebook to use the theme in plotly
and matplotlib figures.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

Mode = Literal["light", "dark"]

CATEGORICAL: dict[str, tuple[str, ...]] = {
    "light": ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"),
    "dark": ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
}
#: Sequential blue ramp, light to dark (steps 100 to 700).
SEQUENTIAL: tuple[str, ...] = (
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
)
INK: dict[str, dict[str, str]] = {
    "light": {
        "surface": "#fcfcfb",
        "page": "#f9f9f7",
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
    },
    "dark": {
        "surface": "#1a1a19",
        "page": "#0d0d0d",
        "primary": "#f0efec",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
    },
}
FONT_FAMILY = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MATPLOTLIB_FONTS = ["Segoe UI", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]

#: Categorical slot of each built-in strategy: the reference first, then the
#: paper's strategies, then the components of the ensemble. The components
#: (prototype, nearest-boundary, random-pairs) are shown together, so
#: random-pairs skips slot 5, a second green, for the violet of slot 6.
STRATEGY_SLOTS = {
    "ensemble": 0,
    "boundary": 1,
    "prototype": 2,
    "random": 3,
    "nearest-boundary": 4,
    "random-pairs": 6,
}
STRATEGY_ORDER = tuple(STRATEGY_SLOTS)
#: Slots given, in this order, to strategies that are not built in.
_FREE_SLOTS = (7, 5)


def _check_mode(mode: str) -> str:
    if mode not in CATEGORICAL:
        raise ValueError(f"mode must be 'light' or 'dark', got {mode!r}")
    return mode


def strategy_sort_key(strategy: str) -> tuple[int, str]:
    """Sort key placing the built-in strategies first, in their colour order."""
    if strategy in STRATEGY_ORDER:
        return (STRATEGY_ORDER.index(strategy), "")
    return (len(STRATEGY_ORDER), strategy)


def strategy_colors(strategies: Iterable[str], mode: Mode = "light") -> dict[str, str]:
    """Colour of each strategy: fixed slots for the built-in ones, free slots in order for others.

    Strategies beyond the eight slots are drawn in the muted ink: fold them
    into "other" or use small multiples rather than inventing more hues.
    """
    slots = CATEGORICAL[_check_mode(mode)]
    colours = {name: slots[i] for name, i in STRATEGY_SLOTS.items()}
    free = [slots[i] for i in _FREE_SLOTS]
    out = {}
    for name in strategies:
        if name not in colours:
            colours[name] = free.pop(0) if free else INK[mode]["muted"]
        out[name] = colours[name]
    return out


def sequential_colorscale(start: int = 0, stop: int | None = None) -> list[list[Any]]:
    """The sequential ramp as a plotly colorscale, steps ``start`` to ``stop`` (0 is the lightest)."""
    steps = SEQUENTIAL[start:stop]
    last = len(steps) - 1
    return [[i / last, colour] for i, colour in enumerate(steps)]


def sequential_color(value: float, low: float = 0.0, high: float = 1.0) -> str:
    """Step of the sequential ramp for ``value`` in ``[low, high]``."""
    t = 0.0 if high == low else min(max((value - low) / (high - low), 0.0), 1.0)
    return SEQUENTIAL[round(t * (len(SEQUENTIAL) - 1))]


def plotly_template(mode: Mode = "light") -> Any:
    """Plotly template of the theme (hairline grid, thin marks, ringed markers)."""
    import plotly.graph_objects as go

    ink = INK[_check_mode(mode)]
    axis = dict(
        gridcolor=ink["grid"],
        gridwidth=1,
        linecolor=ink["axis"],
        zeroline=False,
        ticks="outside",
        tickcolor=ink["axis"],
        tickfont=dict(color=ink["muted"]),
        title=dict(font=dict(color=ink["secondary"])),
        automargin=True,
    )
    return go.layout.Template(
        layout=dict(
            font=dict(family=FONT_FAMILY, size=13, color=ink["primary"]),
            paper_bgcolor=ink["surface"],
            plot_bgcolor=ink["surface"],
            colorway=list(CATEGORICAL[mode]),
            colorscale=dict(sequential=sequential_colorscale()),
            title=dict(font=dict(size=17, color=ink["primary"]), x=0, xanchor="left", pad=dict(l=12)),
            xaxis=axis,
            yaxis=axis,
            legend=dict(font=dict(color=ink["secondary"]), bgcolor="rgba(0,0,0,0)"),
            hoverlabel=dict(font=dict(family=FONT_FAMILY)),
            margin=dict(l=70, r=30, t=70, b=60),
            bargap=0.35,
            barcornerradius=4,
        ),
        data=dict(
            scatter=[go.Scatter(line=dict(width=2), marker=dict(size=9, line=dict(width=2, color=ink["surface"])))],
            bar=[go.Bar(marker=dict(line=dict(width=0)))],
        ),
    )


def matplotlib_rc(mode: Mode = "light") -> dict[str, Any]:
    """matplotlib rcParams of the theme."""
    from cycler import cycler

    ink = INK[_check_mode(mode)]
    return {
        "figure.facecolor": ink["surface"],
        "axes.facecolor": ink["surface"],
        "savefig.facecolor": ink["surface"],
        "axes.edgecolor": ink["axis"],
        "axes.labelcolor": ink["secondary"],
        "axes.titlecolor": ink["primary"],
        "axes.titlesize": 14,
        "axes.titlelocation": "left",
        "axes.prop_cycle": cycler(color=list(CATEGORICAL[mode])),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": ink["grid"],
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": ink["muted"],
        "ytick.color": ink["muted"],
        "text.color": ink["primary"],
        "legend.frameon": False,
        "lines.linewidth": 2,
        "lines.markersize": 8,
        "font.family": "sans-serif",
        "font.sans-serif": MATPLOTLIB_FONTS,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }


def apply(mode: Mode = "light") -> None:
    """Use the theme for the plotly and matplotlib figures created afterwards.

    Registers the plotly templates ``"ramico"`` and ``"ramico_dark"`` and makes
    the requested one the default, then updates matplotlib's rcParams. A
    library that is not installed is skipped.
    """
    _check_mode(mode)
    try:
        import plotly.io as pio
    except ImportError:
        pass
    else:
        pio.templates["ramico"] = plotly_template("light")
        pio.templates["ramico_dark"] = plotly_template("dark")
        pio.templates.default = "ramico" if mode == "light" else "ramico_dark"
    try:
        import matplotlib as mpl
    except ImportError:
        pass
    else:
        params: Any = mpl.rcParams  # its typed keys reject a plain dict[str, Any]
        params.update(matplotlib_rc(mode))
