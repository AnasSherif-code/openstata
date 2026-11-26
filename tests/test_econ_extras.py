"""Unit tests for esize, lincom, mean/proportion/ratio/total and bayesboot.

Expected values are recomputed independently with plain numpy in each test,
so the tests verify the implementation rather than restating it.
"""

import re
import unittest

import numpy as np
import pandas as pd
from scipy import stats

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


class TestEconExtras(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(11)
        n = 60
        g1 = np.empty(n)
        g1[: n // 2] = 1
        g1[n // 2:] = 0
        y = 3.0 + g1 * 2.0 + rng.normal(0, 1.5, n)
        self.df = pd.DataFrame({
            "y": y,
            "g": g1,
            "x1": rng.normal(size=n),
            "bin": (rng.random(n) < 0.4).astype(float),
            "cat": rng.integers(1, 4, size=n).astype(float),
        })
        self.dataset = Dataset(self.df.copy())
        self.console = RecordingConsole()
        self.runner = create_runner()
        self.runner.dataset = self.dataset
        self.runner.console = self.console

    def _run(self, text: str) -> list[str]:
        self.console.lines.clear()
        self.runner.run_source(text)
        return list(self.console.lines)

    # ---- esize -----------------------------------------------------------

    def test_esize_cohend_matches_reference(self):
        a = self.df.loc[self.df.g == 0, "y"]
        b = self.df.loc[self.df.g == 1, "y"]
        n1, n2 = len(a), len(b)
        sp = np.sqrt(((n1 - 1) * a.std(ddof=1) ** 2 + (n2 - 1) * b.std(ddof=1) ** 2) / (n1 + n2 - 2))
        d_ref = (b.mean() - a.mean()) / sp
        lines = self._run("esize twosample y, by(g)")
        m = re.search(r"cohend\s+(-?[\d.]+)", lines[3])
        self.assertIsNotNone(m, lines)
        self.assertAlmostEqual(float(m.group(1)), d_ref, places=5)

    def test_esize_reverse_flips_sign(self):
        self._run("esize twosample y, by(g)")
        base = float(re.search(r"cohend\s+(-?[\d.]+)", self.console.lines[3]).group(1))
        lines = self._run("esize twosample y, by(g) reverse")
        rev = float(re.search(r"cohend\s+(-?[\d.]+)", lines[3]).group(1))
        self.assertAlmostEqual(base, -rev, places=9)

    def test_esize_rejects_bad_effect(self):
        with self.assertRaises(Exception):
            self._run("esize twosample y, by(g) effect(nope)")

    # ---- mean / total ----------------------------------------------------

    def test_mean_matches_pandas_with_over(self):
        lines = self._run("mean x1, over(g)")
        rows = {ln.split("|")[0].strip(): ln.split("|")[1] for ln in lines if "|" in ln and "Estimate" not in ln}
        for gv in ("0", "1"):
            x = self.df.loc[self.df.g.astype(int) == int(gv), "x1"]
            key = [k for k in rows if k.startswith(f"x1 @ {gv}.")][0]
            est = float(rows[key].split()[0])
            se = float(rows[key].split()[1])
            self.assertAlmostEqual(est, x.mean(), places=6)
            self.assertAlmostEqual(se, x.std(ddof=1) / np.sqrt(len(x)), places=6)

    def test_mean_all_vars_reported(self):
        lines = self._run("mean x1 y")
        rows = [ln.split("|")[0].strip() for ln in lines if "|" in ln and "Estimate" not in ln]
        self.assertIn("x1", rows)
        self.assertIn("y", rows)

    def test_total_matches_pandas(self):
        self._run("total y")
        row = [ln for ln in self.console.lines if "|" in ln and "Estimate" not in ln][0]
        est = float(row.split("|")[1].split()[0])
        # console shows 6 significant digits
        self.assertAlmostEqual(est, self.df.y.sum(), places=2)

    # ---- proportion ------------------------------------------------------

    def test_proportion_binary(self):
        self._run("proportion bin")
        row = [ln for ln in self.console.lines if "|" in ln and "Estimate" not in ln][0]
        est, se = row.split("|")[1].split()[:2]
        x = self.df.bin
        p = x.mean()
        self.assertAlmostEqual(float(est), p, places=6)
        self.assertAlmostEqual(float(se), np.sqrt(p * (1 - p) / len(x)), places=6)

    def test_proportion_categorical_one_row_per_value(self):
        self._run("proportion cat")
        rows = {ln.split("|")[0].strip(): ln.split("|")[1].split() for ln in self.console.lines
                if "|" in ln and "Estimate" not in ln}
        cat_rows = {k: v for k, v in rows.items() if k.startswith("cat @")}
        self.assertEqual(len(cat_rows), 3, rows)  # values 1..3
        total = 0.0
        for val in (1, 2, 3):
            key = [k for k in cat_rows if k.split()[-1] in (str(val), f"{val}.0")][0]
            share = (self.df.cat == float(val)).sum() / len(self.df)
            est, se = cat_rows[key][:2]
            self.assertAlmostEqual(float(est), share, places=6)
            self.assertAlmostEqual(float(se), np.sqrt(share * (1 - share) / len(self.df)), places=6)
            total += float(est)
        self.assertAlmostEqual(total, 1.0, places=9)

    # ---- ratio -----------------------------------------------------------

    def test_ratio_delta_method(self):
        # y is the numerator, x1 the denominator; both independent-ish
        self._run("ratio y/x1")
        row = [ln for ln in self.console.lines if "|" in ln and "Estimate" not in ln][0]
        est, se = row.split("|")[1].split()[:2]
        num, den = self.df.y.to_numpy(float), self.df.x1.to_numpy(float)
        mn, md = num.mean(), den.mean()
        R = mn / md
        vn, vd = num.var(ddof=1), den.var(ddof=1)
        covnd = np.cov(num, den, ddof=1)[0, 1]
        se_ref = np.sqrt(max(vn - 2 * R * covnd + R ** 2 * vd, 0.0) / len(num)) / abs(md)
        self.assertAlmostEqual(float(est), R, places=4)  # console shows 6 significant digits
        self.assertAlmostEqual(float(se), se_ref, places=4)

    # ---- dofsubpop -------------------------------------------------------

    def test_dofsubpop_restricts_df(self):
        # With over() and dofsubpop, the default df changes from whole-sample
        # based to subgroup based.
        lines = self._run("mean x1, over(g)")  # df here = 60 - 1 (whole sample)
        self.assertFalse(any("miss" in ln.lower() for ln in lines))
        row = [ln for ln in self.console.lines if "x1" in ln.split("|")[0]][0]
        p_val = float(row.split("|")[1].split()[3])
        # reference p-value with whole-sample df
        n_small = int((self.df.g == 0).sum())
        x = self.df.loc[self.df.g == 0, "x1"]
        t_ref = abs(x.mean()) / (x.std(ddof=1) / np.sqrt(n_small))
        p_ref = 2 * stats.t.sf(t_ref, df=59)
        self.assertAlmostEqual(p_val, p_ref, places=3)  # p printed to 3 decimals

    # ---- lincom ----------------------------------------------------------

    def test_lincom_matches_manual_quadratic_form(self):
        rng = np.random.default_rng(3)
        n = 80
        x1 = rng.normal(size=n)
        d = pd.DataFrame({"x1": x1, "y": 1.0 + 0.5 * x1 + 0.1 * rng.normal(size=n)})
        self.dataset = Dataset(d)
        self.runner.dataset = self.dataset
        self._run("regress y x1")
        e = self.runner.state.e
        beta = e.matrices["b"].ravel()
        V = np.atleast_2d(e.matrices["V"])
        # matrix colnames are [x1, _cons]
        l = np.array([3.0, -2.0])
        self._run("lincom 3*x1 - 2*_cons")
        row = [ln for ln in self.console.lines
               if "|" in ln and "Coefficient" not in ln and "lincom" not in ln.lower()][0]
        est, se = row.split("|")[1].split()[:2]
        self.assertAlmostEqual(float(est), float(l @ beta), places=5)
        self.assertAlmostEqual(float(se), float(np.sqrt(l @ V @ l)), places=5)
        # close to the population value 3*0.5 - 2*1 = -0.5
        self.assertAlmostEqual(float(est), -0.5, delta=0.1)

    def test_lincom_unknown_name_errors(self):
        with self.assertRaises(Exception):
            self._run("lincom 2*nope + _cons")

    # ---- bayesboot -------------------------------------------------------

    def test_bayesboot_covers_true_coefficient(self):
        rng = np.random.default_rng(5)
        n = 100
        x1 = rng.normal(size=n)
        d = pd.DataFrame({"x1": x1, "y": 2.0 * x1 + 1.0})  # exact line
        self.dataset = Dataset(d)
        self.runner.dataset = self.dataset
        lines = self._run("bayesboot y x1, reps(200)")
        line = [ln for ln in lines if ln.split("|")[0].strip() == "x1"][0]
        cells = line.split("|")[1].split()
        mean, sd, obs, lo, hi = (float(c) for c in cells[:5])
        self.assertAlmostEqual(obs, 2.0, places=6)  # observed estimate column
        self.assertLess(mean, 2.02)
        self.assertGreater(mean, 1.98)
        self.assertLessEqual(lo, 2.0 + 1e-9)
        self.assertGreaterEqual(hi, 2.0 - 1e-9)
        self.assertGreater(sd, 0.0)


if __name__ == "__main__":
    unittest.main()
