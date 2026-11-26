"""Tests for descriptive statistics, OLS/robust/cluster regression, logit, probit, xtreg, predict."""

import unittest

import numpy as np
import pandas as pd
from scipy import stats

from openstata.cli.main import create_runner
from openstata.core.dataset import Dataset


class TestStatsAndModels(unittest.TestCase):
    def test_regress_ols_robust_cluster_and_predict(self) -> None:
        rng = np.random.default_rng(42)
        n = 120
        cid = np.repeat(np.arange(1, 21), 6).astype(float)
        x1 = rng.normal(0, 1, n)
        x2 = rng.uniform(0, 5, n)
        g = rng.integers(1, 4, n).astype(float)
        y = 2.0 + 1.5 * x1 - 0.8 * x2 + 0.5 * (g == 2) + 1.2 * (g == 3) + rng.normal(0, 0.5, n)

        runner = create_runner()
        runner.dataset = Dataset(pd.DataFrame({"y": y, "x1": x1, "x2": x2, "g": g, "cid": cid}))

        # Standard OLS
        runner.run_source("regress y x1 x2")
        X = np.column_stack([x1, x2, np.ones(n)])
        beta_ref, *_ = np.linalg.lstsq(X, y, rcond=None)
        b_est = runner.state.e.matrices["b"].ravel()
        np.testing.assert_allclose(b_est, beta_ref, rtol=1e-10, atol=1e-10)
        self.assertEqual(int(runner.state.e.scalars["N"]), n)

        # Robust HC1
        runner.run_source("regress y x1 x2, vce(robust)")
        self.assertEqual(runner.state.e.macros["vcetype"], "Robust")

        # Cluster SEs + factor variables + predict
        runner.run_source(
            """
            regress y x1 x2 i.g, vce(cluster cid)
            predict yhat
            predict res, residuals
            """
        )
        self.assertEqual(int(runner.state.e.scalars["N_clust"]), 20)
        yhat = runner.dataset.df["yhat"].to_numpy()
        res = runner.dataset.df["res"].to_numpy()
        np.testing.assert_allclose(yhat + res, y, rtol=1e-5, atol=1e-5)

    def test_logit_probit_and_ttest_tabulate(self) -> None:
        rng = np.random.default_rng(7)
        n = 300
        x = rng.normal(0, 1, n)
        prob = 1.0 / (1.0 + np.exp(-(0.3 + 1.2 * x)))
        y = (rng.random(n) < prob).astype(float)
        grp = (x > 0).astype(float)

        runner = create_runner()
        runner.dataset = Dataset(pd.DataFrame({"y": y, "x": x, "grp": grp}))
        runner.run_source(
            """
            tabulate y grp, chi2
            ttest x, by(grp)
            logit y x
            predict p_logit, pr
            probit y x
            predict p_probit, pr
            """
        )
        self.assertGreater(runner.state.e.scalars["N"], 0)
        p_l = runner.dataset.df["p_logit"].to_numpy()
        p_p = runner.dataset.df["p_probit"].to_numpy()
        self.assertTrue(np.all((p_l > 0) & (p_l < 1)))
        self.assertTrue(np.all((p_p > 0) & (p_p < 1)))
        # Verify logit slope sign matches DGP
        runner.run_source("logit y x, or")
        self.assertGreater(float(runner.state.e.matrices["b"].ravel()[0]), 0.7)

    def test_xtset_and_xtreg_fe_re(self) -> None:
        rng = np.random.default_rng(99)
        n_id, n_t = 30, 5
        ids = np.repeat(np.arange(1, n_id + 1), n_t).astype(float)
        ts = np.tile(np.arange(1, n_t + 1), n_id).astype(float)
        alpha_i = np.repeat(rng.normal(0, 2, n_id), n_t)
        x = rng.normal(0, 1, n_id * n_t) + 0.5 * alpha_i  # correlated with FE!
        y = 2.5 * x + alpha_i + rng.normal(0, 0.3, n_id * n_t)

        runner = create_runner()
        runner.dataset = Dataset(pd.DataFrame({"id": ids, "t": ts, "y": y, "x": x}))
        runner.run_source(
            """
            xtset id t
            xtreg y x, fe
            """
        )
        b_fe = float(runner.state.e.matrices["b"].ravel()[0])
        self.assertAlmostEqual(b_fe, 2.5, delta=0.15)

        runner.run_source("xtreg y x, re")
        self.assertEqual(int(runner.state.e.scalars["N_g"]), n_id)


if __name__ == "__main__":
    unittest.main()
