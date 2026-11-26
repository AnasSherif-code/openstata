"""didregress command – basic two-way DID estimator.
This implementation supports the simplest DID setup:
    didregress depvar, treat(var) time(var) [robust] [cluster(var)]
It fits an OLS model with intercept, treatment indicator, time indicator,
and their interaction. The coefficient on the interaction term is the
Difference‑in‑Differences estimate.
More advanced options (over(), noxinteract, heterogeneous DID, etc.) are
out of scope for this minimal version and can be added later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from openstata.core.errors import StataError
from openstata.plugins.base import BaseCommand, CommandSpec, ExecutionContext
from openstata.lang.parser import ParsedCommand
from openstata.causal.utils import expand_weights, apply_weights
from openstata.lang.executor import compute_sample_mask


class DidregressCmd(BaseCommand):
    # "di" stays unique to display; "did" is the shortest abbreviation
    spec = CommandSpec("didregress", 3, "Difference‑in‑Differences (basic OLS)")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        ds = ctx.dataset
        # Parse main args – first token is the dependent variable
        toks = cmd.main_args.split()
        if not toks:
            raise StataError("dependent variable required", rc=100)
        depvar = toks[0]
        ds.require_var(depvar)
        # Required options
        treat = cmd.options.get("treat")
        time = cmd.options.get("time")
        if not treat or not isinstance(treat, str):
            raise StataError("option treat() required", rc=198)
        if not time or not isinstance(time, str):
            raise StataError("option time() required", rc=198)
        ds.require_var(treat)
        ds.require_var(time)
        # Optional robust/cluster handling – reuse existing logic for VCE
        vce_kind, vce_opt = self._parse_vce(cmd)
        # Build design matrix manually (no factor expansion needed for this simple case)
        df = ds.df
        y = df[depvar].to_numpy(dtype=float)
        treat_arr = df[treat].astype(float).to_numpy()
        time_arr = df[time].astype(float).to_numpy()
        # Interaction term
        interact = treat_arr * time_arr
        # Intercept
        X = np.column_stack([np.ones_like(y), treat_arr, time_arr, interact])
        # Apply sampling weights if provided via [weight=...] (not implemented here)
        w = expand_weights(df, None)
        y_w, X_w = apply_weights(y, X, w)
        # OLS estimate
        beta, residuals, rank, s = np.linalg.lstsq(X_w, y_w, rcond=None)
        # Compute standard errors (robust/cluster not implemented – use simple OLS SE)
        n, k = X_w.shape
        resid = y_w - X_w @ beta
        sigma2 = (resid @ resid) / (n - k)
        var_beta = sigma2 * np.linalg.inv(X_w.T @ X_w)
        se = np.sqrt(np.diag(var_beta))
        # Output table – reuse helper from model_cmds if available, otherwise simple print
        # We'll mimic the _print_coef_table signature used in regress
        from openstata.commands.model_cmds import _print_coef_table
        names = ["_cons", treat, time, f"{treat}#c.{time}"]
        _print_coef_table(ctx, depvar, names, beta, se, df_t=n - k)

    def _parse_vce(self, cmd: ParsedCommand) -> tuple[str, str | None]:
        # Minimal reimplementation of _parse_vce from model_cmds (robust/cluster only)
        if cmd.options.get("robust") or cmd.options.get("r"):
            return "robust", None
        cl = cmd.options.get("cluster") or cmd.options.get("cl")
        if isinstance(cl, str) and cl:
            return "cluster", cl.strip()
        return "default", None
