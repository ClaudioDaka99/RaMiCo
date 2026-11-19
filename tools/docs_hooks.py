"""MkDocs hook: publish the executed tutorials of examples/ as pages of the documentation.

MkDocs reads pages only from docs/, so the notebooks are copied to docs/tutorials/ (ignored by git) before each build.
Their figures are outputs holding a plotly figure and its PNG copy (ramico.viz.show): the page gets the interactive
figure, as HTML, in place of both.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PLOTLY = "application/vnd.plotly.v1+json"
#: The tutorials published for now; the other notebooks of examples/ stay out of the documentation.
TUTORIALS = ("01_quickstart", "02_mining_strategies")


def interactive(notebook: dict[str, Any]) -> dict[str, Any]:
    """The notebook with each plotly output as HTML; plotly.js is loaded once, from its CDN."""
    import plotly.io as pio

    first = True
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            figure = output.get("data", {}).get(PLOTLY)
            if figure is None:
                continue
            html = pio.to_html(
                {"data": figure["data"], "layout": figure["layout"]},
                config=figure.get("config"),
                include_plotlyjs="cdn" if first else False,
                full_html=False,
            )
            output["data"] = {"text/html": html, "text/plain": "<Figure>"}
            first = False
    return notebook


def on_pre_build(config: Any) -> None:
    target = ROOT / "docs" / "tutorials"
    target.mkdir(exist_ok=True)
    for stale in target.glob("*.ipynb"):  # copies of earlier builds, of tutorials no longer published
        if stale.stem not in TUTORIALS:
            stale.unlink()
    for notebook in (ROOT / "examples" / f"{name}.ipynb" for name in TUTORIALS):
        content = json.loads(notebook.read_text(encoding="utf-8"))
        (target / notebook.name).write_text(json.dumps(interactive(content), indent=1), encoding="utf-8")
