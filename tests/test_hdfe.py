"""Unit tests for areg, ivregress 2sls and the control-function commands.

Reference values are recomputed independently with plain numpy:
  - areg vs an explicit LSDV (dummy-variable) regression
  - ivregress vs the closed-form A = Pz X / inverse-2SLS estimator and the
    heteroskedasticity-robust sandwich
  - cfregress/cfprobit vs ivregress point estimates (the two-step linear
    control-function estimator is algebraically identical to 2SLS).
"""

import unittest

import numpy as np
import pandas as pd

from openstata.cli.main import create_runner
from openstata.core.dataset import Dataset
from openstata.plugins.base import ConsoleWriter


class RecordingConsole(ConsoleWriter):
    def __init__(self) -> None:
        self.lines: list[str] = []

    def write(self, text: str = "") -> None:
        self.lines.append(text)

    def write_error(self, text: str) -> None:
        self.lines.append(text)


def lsdv(df: pd.DataFrame, depvar: str, exog: str, fe: str):
    """Plain dummy regression with full SEs (default VCE formulas)."""
    codes = pd.Categorical(df[fe]).codes
    D = np.eye(codes.max() + 1)[codes][:, 1:]  # drop first as base
    X = np.column_stack([np.ones(len(df)), df[exog].to_numpy(float), D])
    y = df[depvar].to_numpy(float)
    n, k = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    s2 = e @ e / (n - k)
    xtx_pinv = np.linalg.pinv(X.T @ X)
    V = s2 * xtx_pinv
    return beta[:2], np.sqrt(np.diag(V))[:2], n, k - 1  # df_r = n - (k) - (G-1) - ...


class TestAreg(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        n = 300
        g = np.repeat(np.arange(10), n // 10)
        u = rng.normal(size=10)[g]  # group effect
        x1 = rng.normal(size=n)
        y = 1 + 0.5 * x1 + 0.8 * u + rng.normal(0, 1.0, n)
        self.df = pd.DataFrame({"y": y, "x1": x1, "g": g.astype(float)})
        self.dataset = Dataset(self.df.copy())
        self.console = RecordingConsole()
        self.runner = create_runner()
        self.runner.dataset = self.dataset
        self.runner.console = self.console

    def _run(self, text: str) -> list[str]:
        self.console.lines.clear()
        self.runner.run_source(text)
        return list(self.console.lines)

    def _coefs(self) -> dict:
        e = self.runner.state.e
        return dict(zip(e.matrix_colnames["b"], e.matrices["b"].ravel()))

    def test_areg_matches_lsdv(self):
        self._run("areg y x1, absorb(g)")
        got = self._coefs()
        beta_ref, se_ref, n, _ = lsdv(self.df, "y", "x1", "g")
        # x1 (identified within groups) must equal the LSDV estimate exactly;
        # the _cons differs because the intercept is only identified up to the
        # absorbed fixed effects and Stata-style normalization differs.
        self.assertAlmostEqual(got["x1"], beta_ref[1], places=6)
        line = [ln for ln in self.console.lines if ln.split("|")[0].strip() == "x1"][0]
        se_x1 = float(line.split("|")[1].split()[1])
        self.assertAlmostEqual(se_x1, se_ref[1], places=5)

    def test_areg_df(self):
        self._run("areg y x1, absorb(g)")
        e = self.runner.state.e
        n = len(self.df)
        G = self.df.g.nunique()
        self.assertEqual(e.scalars["N"], n)
        self.assertEqual(e.scalars["df_a"], G - 1)
        self.assertEqual(e.scalars["df_r"], n - 2 - (G - 1))

    def test_areg_dfabsorb_keeps_estimates(self):
        self._run("areg y x1, absorb(g)")
        b_plain = self._coefs()["x1"]
        lines = self._run("areg y x1, absorb(g) dfabsorb")
        b_abs = self._coefs()["x1"]
        self.assertAlmostEqual(b_plain, b_abs, places=8)
        self.assertTrue(any("absorbed" in ln.lower() or "dfabsorb" in ln.lower() for ln in lines))

    def test_absorb_var_must_exist(self):
        with self.assertRaises(Exception):
            self._run("areg y x1, absorb(nosuch)")


class TestIvregress(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        n = 400
        x1 = rng.normal(size=n)
        z_inst = rng.normal(size=n)
        u = rng.normal(size=n)
        x_endog = 0.6 * z_inst + u
        y = 1 + 0.5 * x1 + 0.8 * x_endog + rng.normal(0, 0.7, n)
        self.df = pd.DataFrame({"y": y, "x1": x1, "x_endog": x_endog, "z_inst": z_inst})
        self.dataset = Dataset(self.df.copy())
        self.console = RecordingConsole()
        self.runner = create_runner()
        self.runner.dataset = self.dataset
        self.runner.console = self.console

    def _run(self, text: str) -> list[str]:
        self.console.lines.clear()
        self.runner.run_source(text)
        return list(self.console.lines)

    def _fit_manual(self, robust: bool):
        """Closed-form 2SLS. X columns follow the command's order: [x1, x_endog, _cons].
        Robust VCE uses the standard GMM sandwich A = (X'PzX)^-1, meat = (PzX)' diag(e^2) (PzX)."""
        d = self.df
        X = np.column_stack([d.x1, d.x_endog, np.ones(len(d))])
        Z = np.column_stack([d.x1, np.ones(len(d)), d.z_inst])
        y = d.y.to_numpy(float)
        PzX = Z @ np.linalg.pinv(Z.T @ Z) @ Z.T @ X
        beta = np.linalg.pinv(X.T @ PzX) @ (PzX.T @ y)
        e = y - X @ beta
        n, k = X.shape
        A = np.linalg.pinv(X.T @ PzX)
        if robust:
            meat = PzX.T @ ((e ** 2)[:, None] * PzX)
            V = (n / (n - k)) * A @ meat @ A
        else:
            V = (e @ e / (n - k)) * A
        return beta, np.sqrt(np.diag(V))

    def test_2sls_matches_closed_form(self):
        self._run("ivregress 2sls y (x_endog = z_inst) x1")
        e = self.runner.state.e
        beta_ref, _ = self._fit_manual(robust=False)
        np.testing.assert_allclose(e.matrices["b"].ravel(), beta_ref, rtol=1e-7)

    def test_2sls_robust_sandwich(self):
        self._run("ivregress 2sls y (x_endog = z_inst) x1, robust")
        e = self.runner.state.e
        _, se_ref = self._fit_manual(robust=True)
        names = list(e.matrix_colnames["b"])
        idx_x = names.index("x_endog")
        line = [ln for ln in self.console.lines if ln.split("|")[0].strip() == "x_endog"][0]
        se = float(line.split("|")[1].split()[1])
        self.assertAlmostEqual(se, se_ref[idx_x], places=5)

    def test_recovers_true_effect(self):
        self._run("ivregress 2sls y (x_endog = z_inst) x1")
        got = dict(zip(self.runner.state.e.matrix_colnames["b"], self.runner.state.e.matrices["b"].ravel()))
        self.assertAlmostEqual(got["x_endog"], 0.8, delta=0.1)
        self.assertAlmostEqual(got["x1"], 0.5, delta=0.1)

    def test_first_stage_f_reported(self):
        lines = self._run("ivregress 2sls y (x_endog = z_inst) x1")
        f_line = [ln for ln in lines if "First-stage F" in ln]
        self.assertEqual(len(f_line), 1)
        self.assertNotIn("weak", f_line[0])  # F ~= 119 > 10

    def test_weak_instrument_flagged(self):
        # z with a tiny 0.1 coefficient -> weak first stage
        rng = np.random.default_rng(9)
        n = 300
        z = rng.normal(size=n)
        u = rng.normal(size=n)
        x_endog = 0.1 * z + u
        y = 1 + 0.8 * x_endog + rng.normal(0, 0.7, n)
        d = pd.DataFrame({"y": y, "x_endog": x_endog, "z": z})
        self.dataset = Dataset(d)
        self.runner.dataset = self.dataset
        lines = self._run("ivregress 2sls y (x_endog = z)")
        self.assertTrue(any("weak" in ln for ln in lines))

    def test_anderson_rubin_reported(self):
        lines = self._run("ivregress 2sls y (x_endog = z_inst) x1")
        ar = [ln for ln in lines if "Anderson-Rubin" in ln]
        self.assertEqual(len(ar), 1)
        # true effect is 0.8, tested against 0: should reject
        self.assertLess(float(ar[0].split("p = ")[1].split()[0]), 0.001)


class TestControlFunction(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        n = 400
        x1 = rng.normal(size=n)
        z_inst = rng.normal(size=n)
        u = rng.normal(size=n)
        x_endog = 0.6 * z_inst + u
        y = 1 + 0.5 * x1 + 0.8 * x_endog + rng.normal(0, 0.7, n)
        self.df = pd.DataFrame({"y": y, "x1": x1, "x_endog": x_endog, "z_inst": z_inst})
        self.dataset = Dataset(self.df.copy())
        self.console = RecordingConsole()
        self.runner = create_runner()
        self.runner.dataset = self.dataset
        self.runner.console = self.console

    def _run(self, text: str) -> list[str]:
        self.console.lines.clear()
        self.runner.run_source(text)
        return list(self.console.lines)

    def _coefs(self) -> dict:
        e = self.runner.state.e
        return dict(zip(e.matrix_colnames["b"], e.matrices["b"].ravel()))

    def test_cfregress_matches_2sls_point_estimates(self):
        self._run("ivregress 2sls y (x_endog = z_inst) x1")
        b_2sls = self._coefs()
        self._run("cfregress y (x_endog = z_inst) x1")
        b_cf = self._coefs()
        for nm in ("_cons", "x1", "x_endog"):
            self.assertAlmostEqual(b_cf[nm], b_2sls[nm], places=6, msg=nm)

    def test_cfprobit_runs_and_notes_naive_se(self):
        rng = np.random.default_rng(13)
        n = 600
        z = rng.normal(size=n)
        u = rng.normal(size=n)
        x_endog = 0.6 * z + u
        star = -1 + 1.0 * x_endog - 0.5 * u + rng.normal(0, 1, n)
        d = pd.DataFrame({"d": (star > 0).astype(float), "x_endog": x_endog, "z": z})
        self.dataset = Dataset(d)
        self.runner.dataset = self.dataset
        lines = self._run("cfprobit d (x_endog = z)")
        got = self._coefs()
        # Unlike the linear case, the two-step control-function probit is only
        # an approximation for the structural probit, so assert signs and the
        # presence of the control-function coefficient rather than exact values.
        self.assertGreater(got["x_endog"], 0.0)
        cf_key = [k for k in got if k.endswith("_hat")][0]
        self.assertLess(got[cf_key], 0.0)
        self.assertTrue(any("naive" in ln.lower() or "do not account" in ln.lower() for ln in lines))

    def test_requires_instruments(self):
        with self.assertRaises(Exception):
            self._run("cfregress y (x_endog) x1")


if __name__ == "__main__":
    unittest.main()
