"""Unit tests for the gencohort command.
The test creates a tiny panel dataset, runs the gencohort command via the
CLI runner, and verifies that the new cohort variable matches the expected
first‑time‑period per group.
"""

import unittest
from openstata.cli.main import create_runner
from openstata.plugins.registry import CommandRegistry
from openstata.commands import register_builtin_commands
from openstata.core.dataset import Dataset
import pandas as pd


class TestGencohort(unittest.TestCase):
    def setUp(self) -> None:
        # Simple panel: two groups (1,2) observed in three years (2000‑2002)
        self.df = pd.DataFrame({
            "year": [2000, 2001, 2002, 2000, 2001, 2002],
            "id":   [1, 1, 1, 2, 2, 2],
            "y":    [10, 12, 14, 20, 22, 24],
        })
        # Initialise a Dataset with this DataFrame
        self.dataset = Dataset()
        self.dataset.df = self.df.copy()
        # Build a runner that uses the dataset
        registry = CommandRegistry()
        register_builtin_commands(registry)
        self.runner = create_runner()
        # Replace the runner's dataset with our pre‑loaded one
        self.runner.dataset = self.dataset

    def test_cohort_creation_default_name(self):
        # Run gencohort with default cohort variable name
        src = "gencohort, timevar(year) groupvar(id)"
        self.runner.run_source(src)
        # Expected cohort: first year per id (2000 for both groups)
        expected = pd.Series([2000, 2000, 2000, 2000, 2000, 2000], name="cohort")
        pd.testing.assert_series_equal(self.dataset.df["cohort"], expected)

    def test_cohort_creation_custom_name(self):
        src = "gencohort, timevar(year) groupvar(id) cohortvar(cohort_id)"
        self.runner.run_source(src)
        expected = pd.Series([2000, 2000, 2000, 2000, 2000, 2000], name="cohort_id")
        pd.testing.assert_series_equal(self.dataset.df["cohort_id"], expected)

if __name__ == "__main__":
    unittest.main()
