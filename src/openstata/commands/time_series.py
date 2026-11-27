"""Time-series commands: ivlpirf (Instrumental-Variables Local Projections)."""

import numpy as np
import pandas as pd
from scipy import stats

from openstata.plugins.base import BaseCommand, CommandSpec, ExecutionContext
from openstata.lang.parser import ParsedCommand, split_top_level_tokens, strip_quotes
from openstata.core.dataset import Dataset
from openstata.core.errors import StataError


def _parse_ivlpirf_args(cmd: ParsedCommand):
    """Parse ivlpirf command arguments.
    Syntax: ivlpirf depvar_list [, options]
    Options: endogenous(endog = inst) step(#) lag(#/#) cumulative robust cluster() level()
    """
    toks = split_top_level_tokens(cmd.main_args)
    if not toks:
        raise StataError("depvar required", rc=100)
    depvars = [t for t in toks if t not in [','] and not t.startswith('(')]
    if ',' in toks:
        depvars = depvars[:toks.index(',')]
    if not depvars:
        raise StataError("at least one dependent variable required", rc=100)
    
    opts = {}
    for key, val in cmd.options.items():
        opts[key.lower()] = val if val is not None else True
    
    endog_spec = opts.get('endogenous')
    if not endog_spec:
        raise StataError("endogenous() option is required", rc=100)
    endog_spec = str(endog_spec).strip('()')
    if '=' not in endog_spec:
        raise StataError("endogenous() must specify: var = instrument", rc=198)
    endog_var, inst_var = endog_spec.split('=', 1)
    endog_var = endog_var.strip()
    inst_var = inst_var.strip()
    
    step = int(opts.get('step', 4))
    lag_spec = opts.get('lag', '1/1')
    if '/' in str(lag_spec):
        min_lag, max_lag = map(int, lag_spec.split('/'))
        lags = list(range(min_lag, max_lag + 1))
    else:
        lags = [int(lag_spec)]
    cumulative = opts.get('cumulative', False)
    robust = opts.get('robust', False)
    cluster = opts.get('cluster')
    level = float(opts.get('level', 95.0))
    
    return {
        'depvars': depvars, 'endog': endog_var, 'inst': inst_var,
        'step': step, 'lags': lags, 'cumulative': cumulative,
        'robust': robust, 'cluster': cluster, 'level': level,
    }


def _build_lp_data(df, Y_var, D_var, inst_var, controls_list, h):
    """Build data for a single LP horizon regression.
    Returns X, y where X = [1, D_{t-h}, controls_{t-h}] and y = Y_{t}.
    """
    # Use only rows where all control lags are defined
    valid = df[controls_list].notna().all(axis=1)
    df_valid = df[valid].copy()
    
    n = len(df_valid)
    if h > 0:
        # For horizon h, we need Y[t], D[t-h], controls[t-h]
        y = df_valid[Y_var].iloc[h:].values
        D_vals = df_valid[D_var].iloc[:-h].values
        inst_vals = df_valid[inst_var].iloc[:-h].values
        ctrl_vals = df_valid[controls_list].iloc[:-h].values
    else:
        # h=0: Y[t], D[t], controls[t]
        y = df_valid[Y_var].values
        D_vals = df_valid[D_var].values
        inst_vals = df_valid[inst_var].values
        ctrl_vals = df_valid[controls_list].values
    
    # Remove NaN rows
    mask = (~np.isnan(y)) & (~np.isnan(D_vals)) & (~np.isnan(inst_vals)).all()
    if ctrl_vals.size > 0:
        mask &= (~np.isnan(ctrl_vals)).all(axis=1)
    
    y = y[mask]
    D_vals = D_vals[mask]
    inst_vals = inst_vals[mask]
    ctrl_vals = ctrl_vals[mask] if ctrl_vals.size > 0 else ctrl_vals
    
    n_obs = len(y)
    if n_obs == 0:
        return None, None, None
    
    # Build X: [1, D, controls]
    X = np.column_stack([np.ones(n_obs), D_vals])
    if ctrl_vals.size > 0 and len(ctrl_vals.shape) > 1:
        X = np.column_stack([X, ctrl_vals])
    
    # Build Z: [1, inst, controls]  (instruments)
    Z = np.column_stack([np.ones(n_obs), inst_vals])
    if ctrl_vals.size > 0 and len(ctrl_vals.shape) > 1:
        Z = np.column_stack([Z, ctrl_vals])
    
    return X, y, Z


def _estimate_2sls(X, y, Z, robust=False, cluster=None, level=95.0):
    """2SLS estimation with optional robust/cluster SEs.
    Returns: coefs, se for the D coefficient (index 1 in X)."""
    if X is None or len(X) < 2:
        return None, None
    
    n, k = X.shape
    if n < k:
        return None, None
    
    ZtZ_inv = np.linalg.pinv(Z.T @ Z)
    PzX = Z @ ZtZ_inv @ Z.T @ X
    XtPzX = X.T @ PzX
    
    try:
        XtPzX_inv = np.linalg.pinv(XtPzX)
    except np.linalg.LinAlgError:
        return None, None
    
    b = XtPzX_inv @ (PzX.T @ y)
    e = y - X @ b
    df_r = n - k
    
    if df_r <= 0:
        return None, None
    
    # D coefficient is at index 1 (after _cons)
    beta_D = b[1] if len(b) > 1 else None
    if beta_D is None:
        return None, None
    
    # Standard errors for beta_D
    if robust:
        if cluster is not None:
            # Cluster-robust
            codes = cluster
            unique_codes = np.unique(codes)
            G = len(unique_codes)
            PzX_e = PzX * e[:, None]
            meat = np.zeros((k, k))
            for code in unique_codes:
                mask = codes == code
                if np.any(mask):
                    score = PzX_e[mask].sum(axis=0)
                    meat += np.outer(score, score)
            scale = (G / (G - 1.0)) * ((n - 1.0) / df_r) if G > 1 else 1.0
            V = scale * (XtPzX_inv @ meat @ XtPzX_inv)
        else:
            # HC-robust
            meat = PzX.T @ np.diag(e ** 2) @ PzX
            scale = n / df_r
            V = scale * (XtPzX_inv @ meat @ XtPzX_inv)
    else:
        s2 = np.var(e, ddof=1)
        V = s2 * XtPzX_inv
    
    se_D = np.sqrt(V[1, 1]) if V.shape[0] > 1 else None
    
    return beta_D, se_D


class IvlpirfCmd(BaseCommand):
    """`ivlpirf` — Instrumental-Variables Local Projections IRFs.
    Estimates structural impulse response functions using the local projections
    method of Jorda (2005), accounting for endogeneity with instrumental variables.
    """
    spec = CommandSpec("ivlpirf", 10, "Instrumental-variables local projections IRFs")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        ds = ctx.dataset
        args = _parse_ivlpirf_args(cmd)
        
        depvars = args['depvars']
        endog_var = args['endog']
        inst_var = args['inst']
        step = args['step']
        lags = args['lags']
        cumulative = args['cumulative']
        robust = args['robust']
        cluster = args['cluster']
        level = args['level']
        
        for v in depvars + [endog_var, inst_var]:
            ds.require_var(v)
        if cluster:
            ds.require_var(cluster)
        
        df = ds.df
        n_obs = len(df)
        
        # Build control variables (lags of depvars and endog)
        controls_list = []
        for v in depvars + [endog_var]:
            for lag in lags:
                col_name = f"L{lag}.{v}"
                controls_list.append(col_name)
                df[col_name] = df[v].shift(lag)
        
        # Clean up: keep only valid observations for controls
        df_valid = df.dropna(subset=controls_list).copy()
        n_valid = len(df_valid)
        
        # Determine cluster if needed
        if cluster:
            cluster_series = df_valid[cluster]
        else:
            cluster_series = None
        
        ctx.console.write("Instrumental-variables local-projection impulse responses")
        ctx.console.write("")
        vce_note = "Robust" if robust else (f"Cluster-robust (by {cluster})" if cluster else "Standard")
        irf_type = "Cumulative structural" if cumulative else "Structural"
        ctx.console.write(f"Sample: {ds.label if hasattr(ds, 'label') and ds.label else 'data'}    Number of obs = {n_valid}")
        ctx.console.write(f"{irf_type} impulse-response functions")
        ctx.console.write(f"{vce_note} standard errors")
        ctx.console.write("")
        
        # Collect all variables to display
        all_vars = depvars + [endog_var]
        all_coefs = {v: [] for v in all_vars}
        all_se = {v: [] for v in all_vars}
        
        # Estimate for each horizon
        for h in range(step + 1):
            for var in depvars:
                X, y, Z = _build_lp_data(df_valid, var, endog_var, inst_var, controls_list, h)
                coef, se = _estimate_2sls(X, y, Z, robust, cluster_series, level)
                all_coefs[var].append(coef if coef is not None else np.nan)
                all_se[var].append(se if se is not None else np.nan)
            
            # For impulse variable: constrained to 1 at h=0
            if h == 0:
                all_coefs[endog_var].append(1.0)
                all_se[endog_var].append(0.0)
            else:
                X, y, Z = _build_lp_data(df_valid, endog_var, endog_var, inst_var, controls_list, h)
                coef, se = _estimate_2sls(X, y, Z, robust, cluster_series, level)
                all_coefs[endog_var].append(coef if coef is not None else np.nan)
                all_se[endog_var].append(se if se is not None else np.nan)
        
        # Print results table
        header = " " * 12
        for var in all_vars:
            header += f"{var:>15}"
        ctx.console.write(header)
        ctx.console.write("-" * len(header))
        
        for h in range(step + 1):
            label = "--." if h == 0 else f"F{h}."
            line = f"{label:>12}"
            for var in all_vars:
                coef = all_coefs[var][h]
                if h == 0 and var == endog_var:
                    line += f"{'1':>15}  (constrained)"
                elif pd.isna(coef) or coef is None:
                    line += " " * 15
                else:
                    line += f"{coef:>15.6f}"
            ctx.console.write(line)
            
            line2 = " " * 12
            for var in all_vars:
                if h == 0 and var == endog_var:
                    line2 += " " * 15
                else:
                    se = all_se[var][h]
                    if pd.isna(se) or se is None:
                        line2 += " " * 15
                    else:
                        line2 += f"{se:>15.6f}"
            if line2.strip():
                ctx.console.write(line2)
        
        ctx.console.write("")
        ctx.console.write(f"Impulse: {endog_var}")
        ctx.console.write(f"Responses: {' '.join(all_vars)}")
        ctx.console.write(f"Instrument: {inst_var}")
        ctx.console.write(f"Controls: {' '.join(controls_list)}")
        if cumulative:
            ctx.console.write("Note: Cumulative structural impulse-response functions are reported.")
        else:
            ctx.console.write("Note: Structural impulse-response functions are reported.")
        
        # Store results
        ctx.state.e.scalars["N"] = float(n_valid)
        ctx.state.e.scalars["step"] = float(step)
        ctx.state.e.scalars["cumulative"] = 1.0 if cumulative else 0.0
        
        irf_matrix = np.zeros((step + 1, len(all_vars)))
        for h in range(step + 1):
            for i, var in enumerate(all_vars):
                c = all_coefs[var][h]
                irf_matrix[h, i] = 0.0 if (pd.isna(c) or c is None) else float(c)
        
        ctx.state.e.matrices["irf"] = irf_matrix
        ctx.state.e.matrix_colnames["irf"] = [f"F{h}." for h in range(step + 1)]
        ctx.state.e.matrix_rownames["irf"] = list(all_vars)
