"""The command-line tools of the repository."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parent.parent / "tools" / "run_tutorials.py"


def test_run_tutorials_with_a_model_needs_an_output_folder():
    pytest.importorskip("nbformat")
    result = subprocess.run([sys.executable, str(RUNNER), "--model", "mlp"], capture_output=True, text=True)
    assert result.returncode == 2 and "--model needs --out" in result.stderr
