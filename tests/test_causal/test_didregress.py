"""Unit tests for the didregress command.

The tests construct small datasets where the model outcome is deterministic
and the standard errors can be verified against the textbook sandwich
formulas, computed independently below with plain numpy/pandas.

VCE conventions (same as `regress`):
  default: V = s2 * (X'X)^-1,  s2 = RSS/(n-k),               df = n-k
  robust:  V = (n/(n-k)) (X'X)^-1 (X' diag(e^2) X) (X'X)^-1, df = n-k
  cluster: V = c (X'X)^-1 (sum_g X_g' e_g e_g' X_g) (X'X)^-1,
           c = (G/(G-1)) * ((n-1)/(n-k)),                    df = G-1
"""

import unittest

import numpy as np
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


def reference_fit(df: pd.DataFrame, depvar: str, treat: str, time: str):
    """Fit the DID OLS model and return (beta, se_ols, se_hc1, se_cluster) per unit id."""
    X = np.column_stack([
        np.ones(len(df)),
        df[treat].to_numpy(float),
        df[time].to_numpy(float),
        df[treat].to_numpy(float) * df[time].to_numpy(float),
    ])
    y = df[depvar].to_numpy(float)
    n, k = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    xtx_inv = np.linalg.inv(X.T @ X)
    se_ols = np.sqrt(np.diag((e @ e / (n - k)) * xtx_inv))
    meat = X.T @ ((e**2)[:, None] * X)
    se_hc1 = np.sqrt(np.diag((n / (n - k)) * (xtx_inv @ meat @ xtx_inv)))
    ids = df["id"].to_numpy()
    groups = np.unique(ids)
    scores = np.zeros((len(groups), k))
    for gi, g in enumerate(groups):
        m = ids == g
        scores[gi] = (X[m] * e[m][:, None]).sum(axis=0)
    G = len(groups)
    scale = (G / (G - 1.0)) * ((n - 1.0) / (n - k))
    se_cl = np.sqrt(np.diag(scale * (xtx_inv @ scores.T @ scores @ xtx_inv)))
    return beta, se_ols, se_hc1, se_cl


class TestDidregress(unittest.TestCase):
    def setUp(self) -> None:
        # 4 panel units x 2 periods; treated units have unit-level heterogeneity
        # so residuals are non-zero and the robust/cluster formulas are exercised.
        self.df = pd.DataFrame({
            "id":  [1, 1, 2, 2, 3, 3, 4, 4],
            "time": [0, 1, 0, 1, 0, 1, 0, 1],
            "treated": [0, 0, 0, 0, 1, 1, 1, 1],
            "y": [5, 7, 6, 9, 8, 12, 9, 14],
        })
        self.dataset = Dataset()
        self.dataset.df = self.df.copy()
        self.console = RecordingConsole()
        self.runner = create_runner()
        self.runner.dataset = self.dataset
        self.runner.console = self.console
        self.beta, self.se_ols, self.se_hc1, self.se_cl = reference_fit(
            self.df, "y", "treated", "time"
        )

    def _coef_rows(self) -> dict[str, list[float]]:
        """Parse the printed table into name -> [coef, std_err]."""
        rows: dict[str, list[float]] = {}
        for line in self.console.lines:
            name = line.split("|")[0].strip()
            if name in ("_cons", "treated", "time", "treated#c.time"):
                cells = line.split("|")[1].split()
                rows[name] = [float(cells[0]), float(cells[1])]
        return rows

    def _run(self, option: str = "") -> list[str]:
        """Run didregress appending e.g. option=' robust'; the base command
        already ends in options, so `option` must start with a space."""
        self.console.lines.clear()
        self.runner.run_source("didregress y, treat(treated) time(time)" + option)
        return list(self.console.lines)
    def test_basic_did(self):
        self._run()
        rows = self._coef_rows()
        # The interaction term's coefficient is the DID estimate
        self.assertAlmostEqual(rows["treated#c.time"][0], 2.0, places=6)
        self.assertAlmostEqual(rows["_cons"][0], self.beta[0], places=6)
        np.testing.assert_allclose(
            [rows[n][1] for n in ("_cons", "treated", "time")], self.se_ols[:3], rtol=1e-5
        )

    def test_did_intercept(self):
        # Control group baseline at time 0 is the _cons coefficient
        self._run()
        rows = self._coef_rows()
        self.assertAlmostEqual(rows["_cons"][0], 5.5, places=6)

    def test_robust_se_match_hc1(self):
        lines = self._run(" robust")
        rows = self._coef_rows()
        np.testing.assert_allclose(
            [rows[n][1] for n in ("_cons", "treated", "time", "treated#c.time")],
            self.se_hc1,
            rtol=1e-5,
        )
        self.assertTrue(any("(Robust)" in ln for ln in lines))

    def test_cluster_se_match_sandwich(self):
        lines = self._run(" cluster(id)")
        rows = self._coef_rows()
        np.testing.assert_allclose(
            [rows[n][1] for n in ("_cons", "treated", "time", "treated#c.time")],
            self.se_cl,
            rtol=1e-5,
        )
        self.assertTrue(any("adjusted for 4 clusters in id" in ln for ln in lines))

    def test_cluster_variable_required_in_data(self):
        with self.assertRaises(Exception):
            self.runner.run_source("didregress y, treat(treated) time(time) cluster(nosuchvar)")

    def test_e_results_stored(self):
        self._run()
        self.assertEqual(self.runner.state.e.scalars["N"], 8.0)
        np.testing.assert_allclose(self.runner.state.e.matrices["b"].ravel(), self.beta, rtol=1e-9)


if __name__ == "__main__":
    unittest.main()
