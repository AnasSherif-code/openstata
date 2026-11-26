"""Utility functions for the OpenStata causal inference commands.
The functions below provide the minimal building blocks needed for the
commands listed in the plan (didregress, hdidregress, xtdidregress,
teffects aipw, gencohort, estat aggregation).  They are intentionally
light‑weight and rely only on pandas, numpy and scipy, which are already
dependencies of the core project.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "panel_info",
    "make_cohort",
    "over_means",
    "expand_weights",
    "apply_weights",
    "wild_bootstrap",
    "estimate_ate",
    "estimate_att",
    "estimate_cate",
]

# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------

def panel_info(df: pd.DataFrame, timevar: str, groupvar: str) -> pd.DataFrame:
    """Return a DataFrame with ``timevar`` and ``groupvar`` as categorical.

    The function ensures the supplied columns exist, converts them to ``int``
    (or ``category``) and sorts the data by group then time.  It is used by
    ``didregress``/``hdidregress`` to identify cohorts and to compute the
    ``over()`` means.
    """
    if timevar not in df.columns:
        raise KeyError(f"time variable '{timevar}' not found in dataset")
    if groupvar not in df.columns:
        raise KeyError(f"group variable '{groupvar}' not found in dataset")
    out = df[[timevar, groupvar]].copy()
    out[timevar] = out[timevar].astype(int)
    out[groupvar] = out[groupvar].astype(int)
    out = out.sort_values([groupvar, timevar]).reset_index(drop=True)
    return out


def make_cohort(df: pd.DataFrame, timevar: str, groupvar: str, cohortvar: str = "cohort") -> pd.Series:
    """Create a cohort identifier.

    Cohort is defined as the first time period a group appears.  The result
    is a pandas Series that can be assigned back to the original DataFrame.
    """
    info = panel_info(df, timevar, groupvar)
    # Compute first observation time for each group
    first_time = info.groupby(groupvar)[timevar].min()
    # Map each observation to its cohort (the first_time value for its group)
    cohort = df[groupvar].map(first_time)
    cohort.name = cohortvar
    return cohort


def over_means(df: pd.DataFrame, timevar: str, groupvar: str, covars: list[str], *, method: str = "cohort_time") -> pd.DataFrame:
    """Compute means of ``covars`` over the dimension(s) specified by ``method``.

    * ``"cohort_time"`` – average within each (cohort, time) cell.
    * ``"cohort"`` – average within each cohort across all times.
    The returned DataFrame has a MultiIndex ``(cohort, time)`` for the
    first method and a ``cohort`` index for the second.
    """
    info = panel_info(df, timevar, groupvar)
    df_ext = df.copy()
    df_ext["cohort"] = make_cohort(df, timevar, groupvar)
    if method == "cohort_time":
        grp = df_ext.groupby(["cohort", timevar])[covars].mean()
    elif method == "cohort":
        grp = df_ext.groupby("cohort")[covars].mean()
    else:
        raise ValueError(f"unknown over method '{method}'")
    return grp

# ---------------------------------------------------------------------------
# Weight handling
# ---------------------------------------------------------------------------

def expand_weights(df: pd.DataFrame, weight_var: str | None) -> np.ndarray:
    """Return a 1‑D numpy array of weights.

    If ``weight_var`` is ``None`` a vector of ones is returned.  Missing values
    are treated as zero weight.
    """
    if weight_var is None:
        return np.ones(len(df), dtype=float)
    if weight_var not in df.columns:
        raise KeyError(f"weight variable '{weight_var}' not found")
    w = df[weight_var].astype(float).fillna(0.0).to_numpy()
    return w


def apply_weights(y: np.ndarray, X: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply weights to outcome ``y`` and design matrix ``X``.

    The function returns weighted ``y`` and ``X`` (i.e., multiplied by ``sqrt(w)``).
    This is the standard way to incorporate sampling weights in OLS‑type
    estimators.
    """
    if w.ndim != 1:
        raise ValueError("weights must be a 1‑dimensional array")
    sqrt_w = np.sqrt(w)
    return y * sqrt_w, X * sqrt_w[:, None]

# ---------------------------------------------------------------------------
# Bootstrap utilities (wild bootstrap for DID)
# ---------------------------------------------------------------------------

def wild_bootstrap(residuals: np.ndarray, n_rep: int = 999, rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate wild‑bootstrap draws of the residuals.

    Parameters
    ----------
    residuals : array‑like
        The residuals from the original regression.
    n_rep : int, default 999
        Number of bootstrap replications.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    draws : (n_rep, len(residuals)) ndarray
        Each row is a bootstrap draw of the residuals multiplied by a symmetric
        Rademacher distribution (±1 with equal probability).
    """
    if rng is None:
        rng = np.random.default_rng()
    signs = rng.choice([-1, 1], size=(n_rep, len(residuals)))
    return residuals[None, :] * signs

# ---------------------------------------------------------------------------
# Simple treatment‑effect estimators used by ``teffects aipw`` and the DID
# commands.  They operate on already‑weighted outcome and propensity predictions.
# ---------------------------------------------------------------------------

def estimate_ate(y: np.ndarray, w: np.ndarray | None = None) -> float:
    """Estimate the Average Treatment Effect (ATE).

    ``y`` should be the difference of potential outcomes (treated − control) for
    each observation.  If ``w`` is supplied, a weighted mean is returned.
    """
    if w is None:
        return float(np.mean(y))
    return float(np.average(y, weights=w))


def estimate_att(y: np.ndarray, treated: np.ndarray, w: np.ndarray | None = None) -> float:
    """Estimate the Average Treatment Effect on the Treated (ATT).

    ``treated`` is a binary array (1 for treated, 0 for control).  The function
    averages ``y`` over the treated observations, optionally weighted.
    """
    mask = treated == 1
    y_t = y[mask]
    if w is None:
        return float(np.mean(y_t))
    w_t = w[mask]
    return float(np.average(y_t, weights=w_t))


def estimate_cate(y: np.ndarray, subgroup: np.ndarray, w: np.ndarray | None = None) -> pd.Series:
    """Estimate Conditional Average Treatment Effects (CATE).

    ``subgroup`` should contain categorical identifiers (e.g., cohort ids).
    Returns a pandas Series indexed by the unique subgroup values.
    """
    df = pd.DataFrame({"y": y, "g": subgroup})
    if w is not None:
        df["w"] = w
        func = lambda g: np.average(g["y"], weights=g["w"])
    else:
        func = lambda g: g["y"].mean()
    return df.groupby("g").apply(func)

# ---------------------------------------------------------------------------
# End of module
# ---------------------------------------------------------------------------
