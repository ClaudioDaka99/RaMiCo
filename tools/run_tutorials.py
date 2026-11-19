"""Execute the tutorials and store them with their outputs, as published.

    python tools/run_tutorials.py                  # the published tutorials with their own QUICK, in place
    python tools/run_tutorials.py 02               # some of them; any notebook of examples/ by its prefix
    python tools/run_tutorials.py --quick --out build/notebooks   # quick mode, elsewhere (as in CI)
    python tools/run_tutorials.py --full 02        # full mode, whatever the QUICK of the notebook

Each notebook runs with papermill from examples/, with the static PNG copies of the figures turned on
(RAMICO_STATIC_FIGURES=1) so that GitHub shows them; the execution metadata papermill adds (timings, timestamps)
is dropped so that a rerun changes only the outputs. Without --quick or --full the notebook keeps the QUICK of its
parameters cell and papermill adds no cell.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import nbformat

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
#: The tutorials published for now, run when no prefix is given.
TUTORIALS = ("01_quickstart", "02_mining_strategies")


def run(notebook: Path, output: Path, quick: bool | None, timeout: int | None) -> float:
    env = dict(os.environ, RAMICO_STATIC_FIGURES="1", PYTHONIOENCODING="utf-8")
    command = [sys.executable, "-m", "papermill", str(notebook), str(output), "--cwd", str(EXAMPLES)]
    command += ["-k", "python3", "--log-level", "WARNING", "--no-progress-bar"]
    if quick is not None:
        command += ["-p", "QUICK", str(quick)]
    if timeout:
        command += ["--execution-timeout", str(timeout)]
    start = time.perf_counter()
    subprocess.run(command, check=True, env=env)
    elapsed = time.perf_counter() - start
    executed = nbformat.read(output, as_version=4)
    executed.metadata.pop("papermill", None)
    for cell in executed.cells:
        cell.metadata.pop("papermill", None)
        cell.metadata.pop("execution", None)
        for item in cell.get("outputs", []):
            item.get("metadata", {}).pop("papermill", None)
    nbformat.write(executed, output)
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "tutorials", nargs="*", help="prefixes of the notebooks to run, e.g. 01 02 (the published ones by default)"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="run with QUICK = True")
    mode.add_argument("--full", action="store_true", help="run with QUICK = False")
    parser.add_argument("--out", type=Path, help="folder for the executed notebooks (default: in place)")
    parser.add_argument("--timeout", type=int, help="seconds allowed to each cell")
    args = parser.parse_args()
    notebooks = sorted(EXAMPLES.glob("[0-9][0-9]_*.ipynb"))
    if args.tutorials:
        notebooks = [n for n in notebooks if any(n.name.startswith(prefix) for prefix in args.tutorials)]
    else:
        notebooks = [n for n in notebooks if n.stem in TUTORIALS]
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
    for notebook in notebooks:
        output = (args.out / notebook.name) if args.out else notebook
        quick = True if args.quick else False if args.full else None
        elapsed = run(notebook, output.resolve(), quick, args.timeout)
        print(f"{notebook.name}: {elapsed:.0f} s, {output.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
