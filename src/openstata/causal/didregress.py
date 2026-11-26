"""didregress command – basic two-way DID estimator.
This implementation supports the simplest DID setup:
    didregress depvar, treat(var) time(var) [robust] [cluster(var)]
It fits an OLS model with intercept, treatment indicator, time indicator,
and their interaction. The coefficient on the interaction term is the
Difference‑in‑Differences estimate.
Variance estimation mirrors `regress`: ordinary OLS standard errors by
default, HC1 robust standard errors with `robust`, and cluster-robust
standard errors with `cluster(var)`.
More advanced options (over(), noxinteract, heterogeneous DID, etc.) are
out of scope for this minimal version and can be added later.
"""

from __future__ import annotations

import numpy as np
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
        vce_kind, cl_var = self._parse_vce(cmd)
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
        beta, _, rank, _ = np.linalg.lstsq(X_w, y_w, rcond=None)
        n, k = X_w.shape
        if n <= k:
            raise StataError("insufficient observations", rc=2001)
        resid = y_w - X_w @ beta
        df_r = n - rank
        rss = float(resid @ resid)

        xtx_inv = np.linalg.pinv(X_w.T @ X_w)
        n_clust = None
        if vce_kind == "robust":
            vcetype = "Robust"
            meat = X_w.T @ ((resid**2)[:, None] * X_w)
            V = (n / df_r) * (xtx_inv @ meat @ xtx_inv)
            df_t = df_r
        elif vce_kind == "cluster":
            vcetype = "Robust"
            ds.require_var(cl_var)
            clusters = df[cl_var].to_numpy()
            uniq_c, inv_idx = np.unique(clusters, return_inverse=True)
            G = len(uniq_c)
            if G < 2:
                raise StataError("cluster variable must have at least 2 clusters", rc=198)
            scores = X_w * resid[:, None]
            cluster_sums = np.zeros((G, k), dtype=np.float64)
            np.add.at(cluster_sums, inv_idx, scores)
            meat = cluster_sums.T @ cluster_sums
            scale = (G / (G - 1.0)) * ((n - 1.0) / df_r)
            V = scale * (xtx_inv @ meat @ xtx_inv)
            df_t = G - 1
            n_clust = G
        else:
            vcetype = "OLS"
            V = (rss / df_r) * xtx_inv
            df_t = df_r
        se = np.sqrt(np.maximum(0.0, np.diag(V)))

        # Output header + coefficient table, same layout as `regress`
        from openstata.commands.model_cmds import _print_coef_table
        names = ["_cons", treat, time, f"{treat}#c.{time}"]
        ctx.console.write(
            f"Difference-in-Differences regression ({vcetype})"
            f"{' ' * max(1, 30 - len(vcetype))}Number of obs     = {n:>10,}"
        )
        if n_clust is not None:
            ctx.console.write(
                f"                                (Std. err. adjusted for {n_clust} clusters in {cl_var})"
            )
        _print_coef_table(ctx, depvar, names, beta, se, df_t=df_t)
        # e() results so post-estimation commands can see the model
        ctx.state.e.clear()
        ctx.state.e.scalars.update(
            {
                "N": float(n),
                "df_r": float(df_r),
                "df_m": float(k - 1),
            }
        )
        ctx.state.e.matrices["b"] = beta.reshape(1, -1)
        ctx.state.e.matrices["V"] = V.copy()

    def _parse_vce(self, cmd: ParsedCommand) -> tuple[str, str | None]:
        # Minimal reimplementation of _parse_vce from model_cmds (robust/cluster only)
        if cmd.options.get("robust") or cmd.options.get("r"):
            return "robust", None
        cl = cmd.options.get("cluster") or cmd.options.get("cl")
        if isinstance(cl, str) and cl:
            return "cluster", cl.strip()
        return "default", None
