"""Unit tests for the didregress command.
The test constructs a tiny panel dataset with a binary treatment and a time indicator.
The true DID effect is known (difference in differences of 2) and we verify that
the interaction coefficient matches this value within a small tolerance.
"""

import unittest

import pandas as pd

from openstata.cli.main import create_runner
from openstata.core.dataset import Dataset
from openstata.plugins.base import ConsoleWriter


class RecordingConsole(ConsoleWriter):
    """Console that keeps every line it receives so tests can inspect output."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def write(self, text: str = "") -> None:
        self.lines.append(text)

    def write_error(self, text: str) -> None:
        self.lines.append(text)


class TestDidregress(unittest.TestCase):
    def setUp(self) -> None:
        # Simple panel: two periods (0,1), two groups (control=0, treated=1)
        # Outcome: y = 5 + 3*treated + 2*time + 2*treated*time (deterministic)
        self.df = pd.DataFrame({
            "time": [0, 1, 0, 1],
            "treated": [0, 0, 1, 1],
            "y": [5, 7, 8, 12],  # DID effect = 2
        })
        self.dataset = Dataset()
        self.dataset.df = self.df.copy()
        self.console = RecordingConsole()
        self.runner = create_runner()
        self.runner.dataset = self.dataset
        self.runner.console = self.console

    def _interaction_coef(self) -> float:
        """Parse the printed coefficient table for the treated#c.time row."""
        row = next(
            line for line in self.console.lines if "treated#c.time" in line
        )
        # Format: "  treated#c.time | <coef>  <se>  ..."
        return float(row.split("|")[1].split()[0])

    def test_basic_did(self):
        self.runner.run_source("didregress y, treat(treated) time(time)")
        # The interaction term's coefficient is the DID estimate
        self.assertAlmostEqual(self._interaction_coef(), 2.0, places=6)

    def test_did_intercept(self):
        # Control group baseline at time 0 is 5.0 (the _cons coefficient)
        self.runner.run_source("didregress y, treat(treated) time(time)")
        row = next(line for line in self.console.lines if "_cons" in line)
        self.assertAlmostEqual(float(row.split("|")[1].split()[0]), 5.0, places=6)


if __name__ == "__main__":
    unittest.main()
