"""Generate the golden fixtures from the original implementation.

Run from the repository root, in the reference environment pinned in
``tests/golden/requirements-reference.txt``::

    python tests/golden/generate_fixtures.py [--force]

The fixtures freeze the behaviour of the original code, with the sorted
k-medoids initialisation described in ``_harness.py``; the commit they
come from is recorded in ``tests/fixtures/golden/MANIFEST.json``. To
regenerate them, check out commit 0996a2c, the last one where the adapter
drives the original implementation, and run this script there. Never
regenerate the fixtures from refactored code: the golden tests would
become a tautology.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import inspect
import json
import logging
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import _harness as h

PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "scikit-learn-extra", "gower")

ORDERING_CONVENTIONS = [
    "nearest-neighbour selections break ties by position (stable argsort)",
    "k-medoids heuristic initialisation lists the medoids in sorted order",
]


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the golden fixtures.")
    parser.add_argument("--force", action="store_true", help="overwrite existing fixtures")
    args = parser.parse_args()

    if "stable_ties" not in inspect.signature(h.run_case).parameters:
        sys.exit(
            "The golden fixtures are frozen: the adapter in _harness.py now drives the "
            "refactored code. To regenerate them, check out commit 0996a2c, where it still "
            "drives the original implementation, and run this script there."
        )
    logging.getLogger("ramico").setLevel(logging.ERROR)

    existing = [c.name for c in h.CASES if h.fixture_path(c).exists()]
    if existing and not args.force:
        sys.exit(f"fixtures already exist for {existing}; use --force to overwrite them")

    cases = {}
    for case in h.CASES:
        outputs = h.outputs_to_jsonable(h.run_case(case, stable_ties=True))
        # Same run with the numpy default (unstable) argsort, for the record:
        # with ties its choice depends on the numpy version and on the CPU.
        default = h.outputs_to_jsonable(h.run_case(case, stable_ties=False))
        differing = [
            key
            for key in outputs
            if json.dumps(outputs[key], sort_keys=True) != json.dumps(default[key], sort_keys=True)
        ]
        h.write_fixture(case, h.run_case(case, stable_ties=True))
        cases[case.name] = {
            "n_records": len(outputs["records"]["rows"]),
            "n_rules": len(outputs["rules"]["rows"]),
            "outputs_differing_with_default_argsort": differing,
            "fidelity_with_default_argsort": default["metrics"]["fidelity"],
        }
        print(
            f"{case.name:28s} records={cases[case.name]['n_records']:4d} "
            f"rules={cases[case.name]['n_rules']:3d} differing_with_default_argsort={differing}"
        )

    manifest = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "source_commit": _git("rev-parse", "HEAD"),
        "code_under_test_clean": _git("diff", "--name-only", "HEAD", "--", "src/ramico") == "",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
        "tolerances": {"rtol": h.RTOL, "atol": h.ATOL},
        "ordering_conventions": ORDERING_CONVENTIONS,
        "cases": cases,
    }
    (h.FIXTURES_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
