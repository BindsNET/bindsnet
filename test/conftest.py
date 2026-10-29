"""Runs before every test module: pin the CPU maths libraries (see
``test/regression/cpu_pins.py``). Must not import torch."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "regression"))

import cpu_pins  # noqa: E402

cpu_pins.apply()
