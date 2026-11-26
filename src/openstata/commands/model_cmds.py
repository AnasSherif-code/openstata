"""Econometric estimation and post-estimation commands: regress, logit, probit, xtset, xtreg, predict."""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
from scipy import optimize, stats

from openstata.core.dataset import Dataset
from openstata.core.errors import StataError
from openstata.lang.executor import compute_sample_mask
from openstata.lang.expr import eval_expression
from openstata.lang.parser import ParsedCommand, expand_varlist, split_top_level_tokens
from openstata.plugins.base import BaseCommand, CommandSpec, ExecutionContext


def _parse_vce(cmd: ParsedCommand) -> tuple[str, str | None]:
    if cmd.options.get("robust") or cmd.options.get("r"):
        return "robust", None
    cl = cmd.options.get("cluster") or cmd.options.get("cl")
    if isinstance(cl, str) and cl:
        return "cluster", cl.strip()
    vce = cmd.options.get("vce")
    if isinstance(vce, str):
        toks = vce.split()
        kind = toks[0].lower()
        if kind in ("r", "rob", "robust"):
            return "robust", None
        if kind in ("cl", "clus", "cluster") and len(toks) >= 2:
            return "cluster", toks[1]
        if kind in ("ols", "oim"):
            return "oim", None
        raise StataError(f"vce({vce}) not supported", rc=198)
    return "default", None


def _expand_factor_term(term: str, df: pd.DataFrame, mask: np.ndarray) -> tuple[list[np.ndarray], list[str], list[str]]:
    """Expand a single term like x, i.rep78, ib3.rep78, c.x, i.a#i.b, or a##b."""
    if "##" in term:
        parts = term.split("##")
        if len(parts) != 2:
            raise StataError(f"unsupported multi-way ## term '{term}'", rc=198)
        a_cols, a_names, a_raw = _expand_factor_term(parts[0], df, mask)
        b_cols, b_names, b_raw = _expand_factor_term(parts[1], df, mask)
        cols = list(a_cols) + list(b_cols)
        names = list(a_names) + list(b_names)
        for ca, na in zip(a_cols, a_names):
            for cb, nb in zip(b_cols, b_names):
                cols.append(ca * cb)
                names.append(f"{na}#{nb}")
        return cols, names, list(dict.fromkeys(a_raw + b_raw))

    if "#" in term:
        parts = term.split("#")
        if len(parts) != 2:
            raise StataError(f"unsupported multi-way # term '{term}'", rc=198)
        a_cols, a_names, a_raw = _expand_factor_term(parts[0], df, mask)
        b_cols, b_names, b_raw = _expand_factor_term(parts[1], df, mask)
        cols, names = [], []
        for ca, na in zip(a_cols, a_names):
            for cb, nb in zip(b_cols, b_names):
                cols.append(ca * cb)
                names.append(f"{na}#{nb}")
        return cols, names, list(dict.fromkeys(a_raw + b_raw))

    m_fac = re.match(r"^i(?:b(\d+))?\.([A-Za-z_][A-Za-z0-9_]*)$", term, flags=re.IGNORECASE)
    if m_fac:
        base_str, vname = m_fac.group(1), m_fac.group(2)
        if vname not in df.columns:
            raise StataError(f"variable {vname} not found", rc=111)
        s = df[vname]
        levels = sorted(s.loc[mask & s.notna()].unique())
        if not levels:
            return [], [], [vname]
        base_val = float(base_str) if base_str is not None else levels[0]
        cols = []
        names = []
        for lv in levels:
            if lv == base_val:
                continue
            cols.append((s == lv).to_numpy(dtype=np.float64))
            lbl = str(int(lv)) if isinstance(lv, (int, float)) and float(int(lv)) == lv else str(lv)
            names.append(f"{lbl}.{vname}")
        return cols, names, [vname]

    if term.lower().startswith("c."):
        vname = term[2:]
        if vname not in df.columns:
            raise StataError(f"variable {vname} not found", rc=111)
        return [df[vname].to_numpy(dtype=np.float64)], [vname], [vname]

    if term not in df.columns:
        raise StataError(f"variable {term} not found", rc=111)
    return [df[term].to_numpy(dtype=np.float64)], [term], [term]


def build_design_matrix(
    depvar: str,
    rhs_terms: list[str],
    ds: Dataset,
    mask: np.ndarray,
    noconstant: bool = False,
    extra_vars: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Construct (y, X, col_names, valid_sample_mask) respecting factor notation."""
    ds.require_var(depvar)
    if ds.is_string(depvar):
        raise StataError(f"depvar {depvar} may not be string", rc=109)

    raw_vars = [depvar] + (extra_vars or [])
    expanded_rhs: list[str] = []
    for t in rhs_terms:
        if any(sym in t for sym in (".", "#")):
            expanded_rhs.append(t)
        else:
            expanded_rhs.extend(expand_varlist([t], ds))

    # First pass to identify all raw variables for missingness filtering
    valid = mask & (~ds.get_missing_mask(depvar))
    for ev in extra_vars or []:
        ds.require_var(ev)
        valid &= ~ds.get_missing_mask(ev)
    for t in expanded_rhs:
        _, _, rvars = _expand_factor_term(t, ds.df, valid)
        for rv in rvars:
            valid &= ~ds.get_missing_mask(rv)
            if rv not in raw_vars:
                raw_vars.append(rv)

    n = int(np.sum(valid))
    if n == 0:
        raise StataError("no observations", rc=2000)

    y = ds.df.loc[valid, depvar].to_numpy(dtype=np.float64)
    cols: list[np.ndarray] = []
    names: list[str] = []
    for t in expanded_rhs:
        t_cols, t_names, _ = _expand_factor_term(t, ds.df, valid)
        for c, nm in zip(t_cols, t_names):
            cols.append(c[valid])
            names.append(nm)

    if not noconstant:
        cols.append(np.ones(n, dtype=np.float64))
        names.append("_cons")

    X = np.column_stack(cols) if cols else np.empty((n, 0), dtype=np.float64)
    return y, X, names, valid


def _print_coef_table(
    ctx: ExecutionContext,
    depvar: str,
    names: list[str],
    beta: np.ndarray,
    se: np.ndarray,
    df_t: int | None,
    level: float = 95.0,
    stat_label: str = "t",
    exponentiate: bool = False,
    coef_header: str = "Coefficient",
) -> None:
    alpha = 1.0 - level / 100.0
    crit = float(stats.t.ppf(1.0 - alpha / 2.0, df_t)) if df_t is not None else float(stats.norm.ppf(1.0 - alpha / 2.0))
    ctx.console.write("-" * 78)
    ctx.console.write(
        f"{depvar:>12} | {coef_header:>11}  {'Std. err.':>10}  {stat_label:>7}  {'P>|' + stat_label + '|':>7}  [{int(level)}% conf. interval]"
    )
    ctx.console.write("-" * 78)
    for nm, b, s in zip(names, beta, se):
        if s <= 0 or np.isnan(s):
            stat_v = np.nan
            p_v = np.nan
            lo = np.nan
            hi = np.nan
        else:
            stat_v = b / s
            p_v = float(2.0 * stats.t.sf(abs(stat_v), df_t)) if df_t is not None else float(2.0 * stats.norm.sf(abs(stat_v)))
            lo = b - crit * s
            hi = b + crit * s

        disp_b = np.exp(b) if exponentiate else b
        disp_s = (np.exp(b) * s) if exponentiate else s
        disp_lo = np.exp(lo) if exponentiate else lo
        disp_hi = np.exp(hi) if exponentiate else hi
        ctx.console.write(
            f"{nm:>12} | {disp_b:>11.6g}  {disp_s:>10.6g}  {stat_v:>7.2f}  {p_v:>7.3f}  {disp_lo:>10.6g}  {disp_hi:>10.6g}"
        )
    ctx.console.write("-" * 78)


class RegressCmd(BaseCommand):
    spec = CommandSpec("regress", 3, "Linear regression (OLS, robust, clustered SEs)")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        ds = ctx.dataset
        toks = split_top_level_tokens(cmd.main_args)
        if not toks:
            raise StataError("depvar required", rc=100)
        depvar = expand_varlist([toks[0]], ds)[0]
        rhs_terms = toks[1:]
        noconst = bool(cmd.options.get("noconstant") or cmd.options.get("nocons"))
        vce_kind, cl_var = _parse_vce(cmd)
        level = float(cmd.options.get("level", 95.0)) if isinstance(cmd.options.get("level"), str) else 95.0

        mask = compute_sample_mask(cmd, ds, ctx.state)
        extra = [cl_var] if cl_var else None
        y, X, names, valid = build_design_matrix(depvar, rhs_terms, ds, mask, noconstant=noconst, extra_vars=extra)

        n, k = X.shape
        if n <= k:
            raise StataError("insufficient observations", rc=2001)

        # Analytic weights if provided
        if cmd.weight is not None:
            wtype, wexpr = cmd.weight
            w_all = np.asarray(eval_expression(wexpr, ds, ctx.state).values, dtype=np.float64)[valid]
            if np.any(w_all <= 0):
                raise StataError("weights must be positive", rc=402)
            sw = np.sqrt(w_all / np.mean(w_all)) if wtype == "aw" else np.sqrt(w_all)
            Xw = X * sw[:, None]
            yw = y * sw
        else:
            Xw, yw = X, y

        beta, _, rank, _ = np.linalg.lstsq(Xw, yw, rcond=None)
        resid = yw - Xw @ beta
        df_r = n - rank
        df_m = rank - (0 if noconst else 1)

        xtx_inv = np.linalg.pinv(Xw.T @ Xw)
        rss = float(np.sum(resid**2))
        y_mean = 0.0 if noconst else float(np.mean(yw))
        tss = float(np.sum((yw - y_mean) ** 2))
        mss = tss - rss
        r2 = mss / tss if tss > 0 else 0.0
        r2_a = 1.0 - (1.0 - r2) * ((n - (0 if noconst else 1)) / df_r) if df_r > 0 else np.nan
        rmse = float(np.sqrt(rss / df_r))

        vcetype = "OLS"
        n_clust = None
        if vce_kind == "robust":
            vcetype = "Robust"
            meat = Xw.T @ ((resid**2)[:, None] * Xw)
            V = (n / df_r) * (xtx_inv @ meat @ xtx_inv)
            df_t = df_r
        elif vce_kind == "cluster":
            vcetype = "Robust"
            clusters = ds.df.loc[valid, cl_var].to_numpy()
            uniq_c, inv_idx = np.unique(clusters, return_inverse=True)
            G = len(uniq_c)
            if G < 2:
                raise StataError("cluster variable must have at least 2 clusters", rc=198)
            scores = Xw * resid[:, None]
            cluster_sums = np.zeros((G, k), dtype=np.float64)
            np.add.at(cluster_sums, inv_idx, scores)
            meat = cluster_sums.T @ cluster_sums
            scale = (G / (G - 1.0)) * ((n - 1.0) / df_r)
            V = scale * (xtx_inv @ meat @ xtx_inv)
            df_t = G - 1
            n_clust = G
        else:
            V = (rss / df_r) * xtx_inv
            df_t = df_r

        se = np.sqrt(np.maximum(0.0, np.diag(V)))
        # F-stat
        if df_m > 0:
            if vce_kind == "default":
                f_stat = (mss / df_m) / (rss / df_r)
            else:
                idx_nc = [i for i, nm in enumerate(names) if nm != "_cons"]
                b_sub = beta[idx_nc]
                V_sub = V[np.ix_(idx_nc, idx_nc)]
                f_stat = float(b_sub.T @ np.linalg.pinv(V_sub) @ b_sub) / df_m
            p_f = float(stats.f.sf(f_stat, df_m, df_t))
        else:
            f_stat, p_f = np.nan, np.nan

        ctx.console.write(f"Linear regression ({vcetype})              Number of obs     = {n:>10,}")
        ctx.console.write(f"                                        F({df_m}, {df_t})       = {f_stat:>10.2f}")
        ctx.console.write(f"                                        Prob > F          = {p_f:>10.4f}")
        ctx.console.write(f"                                        R-squared         = {r2:>10.4f}")
        if vce_kind == "default":
            ctx.console.write(f"                                        Adj R-squared     = {r2_a:>10.4f}")
        ctx.console.write(f"                                        Root MSE          = {rmse:>10.5g}")
        if n_clust is not None:
            ctx.console.write(f"                                (Std. err. adjusted for {n_clust} clusters in {cl_var})")

        _print_coef_table(ctx, depvar, names, beta, se, df_t=df_t, level=level)

        ctx.state.e.clear()
        ctx.state.e.scalars.update(
            {
                "N": float(n),
                "df_m": float(df_m),
                "df_r": float(df_r),
                "F": float(f_stat),
                "r2": float(r2),
                "r2_a": float(r2_a),
                "rmse": float(rmse),
                "mss": float(mss),
                "rss": float(rss),
            }
        )
        if n_clust is not None:
            ctx.state.e.scalars["N_clust"] = float(n_clust)
            ctx.state.e.macros["clustvar"] = str(cl_var)
        ctx.state.e.macros.update(
            {
                "cmd": "regress",
                "depvar": depvar,
                "indepvars": " ".join(names),
                "vce": vce_kind,
                "vcetype": vcetype,
                "predict": "xb",
                "noconstant": "1" if noconst else "0",
            }
        )
        ctx.state.e.matrices["b"] = beta.reshape(1, -1)
        ctx.state.e.matrices["V"] = V
        ctx.state.e.matrix_colnames["b"] = names
        ctx.state.e.matrix_colnames["V"] = names
        ctx.state.e.matrix_rownames["V"] = names
        ctx.state.e.matrices["_sample"] = valid
        ctx.state.e.macros["_rhs_terms"] = " ".join(rhs_terms)


class BinaryChoiceCmd(BaseCommand):
    """Shared implementation for `logit` and `probit` MLE estimators."""

    def __init__(self, family: str) -> None:
        self.family = family
        self.spec = CommandSpec(family, 4, f"{family.capitalize()} binary response regression (MLE)")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        ds = ctx.dataset
        toks = split_top_level_tokens(cmd.main_args)
        if not toks:
            raise StataError("depvar required", rc=100)
        depvar = expand_varlist([toks[0]], ds)[0]
        rhs_terms = toks[1:]
        noconst = bool(cmd.options.get("noconstant") or cmd.options.get("nocons"))
        vce_kind, cl_var = _parse_vce(cmd)
        show_or = bool(cmd.options.get("or")) and self.family == "logit"
        level = float(cmd.options.get("level", 95.0)) if isinstance(cmd.options.get("level"), str) else 95.0

        mask = compute_sample_mask(cmd, ds, ctx.state)
        extra = [cl_var] if cl_var else None
        y_raw, X, names, valid = build_design_matrix(depvar, rhs_terms, ds, mask, noconstant=noconst, extra_vars=extra)
        y = (y_raw != 0).astype(np.float64)
        n, k = X.shape
        if np.all(y == 0) or np.all(y == 1):
            raise StataError("outcome does not vary; remember Stata drops perfect predictors", rc=2000)

        def _neg_ll_and_grad(beta: np.ndarray):
            eta = np.clip(X @ beta, -30.0, 30.0)
            if self.family == "logit":
                p = 1.0 / (1.0 + np.exp(-eta))
                p = np.clip(p, 1e-12, 1.0 - 1e-12)
                ll = float(np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
                grad = X.T @ (p - y)
                w = p * (1.0 - p)
            else:
                p = np.clip(stats.norm.cdf(eta), 1e-12, 1.0 - 1e-12)
                phi = stats.norm.pdf(eta)
                ll = float(np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
                r_score = (y - p) * phi / (p * (1.0 - p))
                grad = -(X.T @ r_score)
                w = (phi**2) / (p * (1.0 - p))
            return -ll, grad, w, p

        opt = optimize.minimize(
            lambda b: _neg_ll_and_grad(b)[:2],
            np.zeros(k, dtype=np.float64),
            jac=True,
            method="BFGS",
            options={"gtol": 1e-8, "maxiter": int(ctx.state.settings.get("maxiter", 300))},
        )
        beta = opt.x
        neg_ll, _, w, p_hat = _neg_ll_and_grad(beta)
        ll = -neg_ll
        p0 = np.clip(float(np.mean(y)), 1e-12, 1.0 - 1e-12)
        ll0 = float(np.sum(y * np.log(p0) + (1.0 - y) * np.log(1.0 - p0)))
        r2_p = 1.0 - ll / ll0 if ll0 != 0 else 0.0

        info_inv = np.linalg.pinv(X.T @ (w[:, None] * X))
        if self.family == "logit":
            scores = X * (y - p_hat)[:, None]
        else:
            eta = np.clip(X @ beta, -30.0, 30.0)
            phi = stats.norm.pdf(eta)
            scores = X * (((y - p_hat) * phi / (p_hat * (1.0 - p_hat)))[:, None])

        if vce_kind == "robust":
            V = info_inv @ (scores.T @ scores) @ info_inv
            vcetype = "Robust"
        elif vce_kind == "cluster":
            clusters = ds.df.loc[valid, cl_var].to_numpy()
            uniq_c, inv_idx = np.unique(clusters, return_inverse=True)
            G = len(uniq_c)
            csums = np.zeros((G, k), dtype=np.float64)
            np.add.at(csums, inv_idx, scores)
            V = (G / (G - 1.0)) * (info_inv @ (csums.T @ csums) @ info_inv)
            vcetype = "Robust"
        else:
            V = info_inv
            vcetype = "OIM"

        se = np.sqrt(np.maximum(0.0, np.diag(V)))
        idx_nc = [i for i, nm in enumerate(names) if nm != "_cons"]
        df_m = len(idx_nc)
        if df_m > 0:
            if vce_kind == "default":
                chi2_stat = 2.0 * (ll - ll0)
                chi2_lbl = f"LR chi2({df_m})"
            else:
                b_sub = beta[idx_nc]
                V_sub = V[np.ix_(idx_nc, idx_nc)]
                chi2_stat = float(b_sub.T @ np.linalg.pinv(V_sub) @ b_sub)
                chi2_lbl = f"Wald chi2({df_m})"
            p_chi2 = float(stats.chi2.sf(max(0.0, chi2_stat), df_m))
        else:
            chi2_stat, p_chi2, chi2_lbl = 0.0, 1.0, "chi2(0)"

        title = "Logistic regression" if self.family == "logit" else "Probit regression"
        ctx.console.write(f"{title:<36} Number of obs   = {n:>10,}")
        ctx.console.write(f"                                     {chi2_lbl:<15} = {chi2_stat:>10.2f}")
        ctx.console.write(f"                                     Prob > chi2     = {p_chi2:>10.4f}")
        ctx.console.write(f"Log likelihood = {ll:<19.6f} Pseudo R2       = {r2_p:>10.4f}")
        _print_coef_table(
            ctx,
            depvar,
            names,
            beta,
            se,
            df_t=None,
            level=level,
            stat_label="z",
            exponentiate=show_or,
            coef_header="Odds ratio" if show_or else "Coefficient",
        )

        ctx.state.e.clear()
        ctx.state.e.scalars.update(
            {
                "N": float(n),
                "ll": float(ll),
                "ll_0": float(ll0),
                "r2_p": float(r2_p),
                "chi2": float(chi2_stat),
                "df_m": float(df_m),
                "p": float(p_chi2),
            }
        )
        ctx.state.e.macros.update(
            {
                "cmd": self.family,
                "depvar": depvar,
                "indepvars": " ".join(names),
                "vce": vce_kind,
                "vcetype": vcetype,
                "noconstant": "1" if noconst else "0",
                "_rhs_terms": " ".join(rhs_terms),
            }
        )
        ctx.state.e.matrices["b"] = beta.reshape(1, -1)
        ctx.state.e.matrices["V"] = V
        ctx.state.e.matrix_colnames["b"] = names
        ctx.state.e.matrix_colnames["V"] = names
        ctx.state.e.matrix_rownames["V"] = names
        ctx.state.e.matrices["_sample"] = valid


class XtsetCmd(BaseCommand):
    spec = CommandSpec("xtset", 5, "Declare panel dataset structure (panelvar [timevar])")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        toks = split_top_level_tokens(cmd.main_args)
        if not toks:
            if ctx.dataset.panel_var:
                ctx.console.write(f"Panel variable: {ctx.dataset.panel_var}")
                if ctx.dataset.time_var:
                    ctx.console.write(f" Time variable: {ctx.dataset.time_var}")
                return
            raise StataError("panelvar required", rc=198)
        vlist = expand_varlist(toks, ctx.dataset)
        ctx.dataset.panel_var = vlist[0]
        ctx.dataset.time_var = vlist[1] if len(vlist) >= 2 else None
        ctx.dataset.sort_rows(vlist)
        ctx.console.write(f"Panel variable: {ctx.dataset.panel_var}")
        if ctx.dataset.time_var:
            ctx.console.write(f" Time variable: {ctx.dataset.time_var}")


class XtregCmd(BaseCommand):
    spec = CommandSpec("xtreg", 5, "Fixed-effects or random-effects panel regression")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        ds = ctx.dataset
        if not ds.panel_var:
            raise StataError("must xtset data before xtreg", rc=459)
        pvar = ds.panel_var
        toks = split_top_level_tokens(cmd.main_args)
        if not toks:
            raise StataError("depvar required", rc=100)
        depvar = expand_varlist([toks[0]], ds)[0]
        rhs_terms = toks[1:]
        use_re = bool(cmd.options.get("re"))
        use_fe = bool(cmd.options.get("fe")) or not use_re
        level = float(cmd.options.get("level", 95.0)) if isinstance(cmd.options.get("level"), str) else 95.0

        mask = compute_sample_mask(cmd, ds, ctx.state)
        y, X_full, names_full, valid = build_design_matrix(
            depvar, rhs_terms, ds, mask, noconstant=False, extra_vars=[pvar]
        )
        # Separate _cons for within demeaning
        idx_nc = [i for i, nm in enumerate(names_full) if nm != "_cons"]
        X = X_full[:, idx_nc]
        names = [names_full[i] for i in idx_nc]
        groups = ds.df.loc[valid, pvar].to_numpy()
        uniq_g, inv_g = np.unique(groups, return_inverse=True)
        G = len(uniq_g)
        n, k = X.shape
        if G < 2:
            raise StataError("insufficient panel groups", rc=2001)

        # Compute group means
        g_counts = np.bincount(inv_g).astype(np.float64)
        y_gmean = np.bincount(inv_g, weights=y) / g_counts
        X_gmean = np.column_stack([np.bincount(inv_g, weights=X[:, j]) / g_counts for j in range(k)])

        if use_fe:
            y_w = y - y_gmean[inv_g]
            X_w = X - X_gmean[inv_g]
            beta, _, rank, _ = np.linalg.lstsq(X_w, y_w, rcond=None)
            resid_w = y_w - X_w @ beta
            df_r = n - G - rank
            if df_r <= 0:
                raise StataError("insufficient degrees of freedom", rc=2001)
            rss_w = float(np.sum(resid_w**2))
            tss_w = float(np.sum(y_w**2))
            r2_w = 1.0 - rss_w / tss_w if tss_w > 0 else 0.0
            xtx_inv = np.linalg.pinv(X_w.T @ X_w)
            V = (rss_w / df_r) * xtx_inv

            # Add intercept back as average fixed effect
            cons = float(np.mean(y) - np.mean(X, axis=0) @ beta)
            beta_out = np.append(beta, cons)
            V_out = np.zeros((k + 1, k + 1), dtype=np.float64)
            V_out[:k, :k] = V
            V_out[k, k] = rss_w / (n * df_r)
            names_out = names + ["_cons"]
            se_out = np.sqrt(np.maximum(0.0, np.diag(V_out)))
            f_stat = ((tss_w - rss_w) / max(1, rank)) / (rss_w / df_r)
            p_f = float(stats.f.sf(f_stat, max(1, rank), df_r))

            ctx.console.write(f"Fixed-effects (within) regression        Number of obs     = {n:>10,}")
            ctx.console.write(f"Group variable: {pvar:<24} Number of groups  = {G:>10,}")
            ctx.console.write(f"R-sq: within  = {r2_w:<8.4f}                 F({rank}, {df_r})       = {f_stat:>10.2f}")
            ctx.console.write(f"                                         Prob > F          = {p_f:>10.4f}")
            _print_coef_table(ctx, depvar, names_out, beta_out, se_out, df_t=df_r, level=level)
            df_m = rank
            r2_stored = r2_w
        else:
            # Swamy-Arora Random Effects GLS
            y_w = y - y_gmean[inv_g]
            X_w = X - X_gmean[inv_g]
            b_fe, _, rank_w, _ = np.linalg.lstsq(X_w, y_w, rcond=None)
            sigma_e2 = float(np.sum((y_w - X_w @ b_fe) ** 2)) / max(1, n - G - rank_w)

            X_b = np.column_stack([X_gmean, np.ones(G, dtype=np.float64)])
            b_be, _, rank_b, _ = np.linalg.lstsq(X_b, y_gmean, rcond=None)
            rss_b = float(np.sum((y_gmean - X_b @ b_be) ** 2))
            t_bar_h = G / float(np.sum(1.0 / g_counts))
            sigma_u2 = max(0.0, (rss_b / max(1, G - rank_b)) - sigma_e2 / t_bar_h)

            theta_g = 1.0 - np.sqrt(sigma_e2 / (g_counts * sigma_u2 + sigma_e2))
            theta_i = theta_g[inv_g]
            y_re = y - theta_i * y_gmean[inv_g]
            X_re = np.column_stack([X - theta_i[:, None] * X_gmean[inv_g], 1.0 - theta_i])
            beta_out, _, rank, _ = np.linalg.lstsq(X_re, y_re, rcond=None)
            resid_re = y_re - X_re @ beta_out
            sigma_re2 = float(np.sum(resid_re**2)) / max(1, n - rank)
            V_out = sigma_re2 * np.linalg.pinv(X_re.T @ X_re)
            names_out = names + ["_cons"]
            se_out = np.sqrt(np.maximum(0.0, np.diag(V_out)))
            chi2 = float(beta_out[:-1].T @ np.linalg.pinv(V_out[:-1, :-1]) @ beta_out[:-1])
            p_c = float(stats.chi2.sf(chi2, k))
            ctx.console.write(f"Random-effects GLS regression            Number of obs     = {n:>10,}")
            ctx.console.write(f"Group variable: {pvar:<24} Number of groups  = {G:>10,}")
            ctx.console.write(f"                                         Wald chi2({k})     = {chi2:>10.2f}")
            ctx.console.write(f"                                         Prob > chi2       = {p_c:>10.4f}")
            _print_coef_table(ctx, depvar, names_out, beta_out, se_out, df_t=None, level=level, stat_label="z")
            df_m = k
            r2_stored = 0.0

        ctx.state.e.clear()
        ctx.state.e.scalars.update({"N": float(n), "N_g": float(G), "df_m": float(df_m), "r2_w": float(r2_stored)})
        ctx.state.e.macros.update(
            {
                "cmd": "xtreg",
                "depvar": depvar,
                "indepvars": " ".join(names_out),
                "ivar": pvar,
                "noconstant": "0",
                "_rhs_terms": " ".join(rhs_terms),
            }
        )
        ctx.state.e.matrices["b"] = beta_out.reshape(1, -1)
        ctx.state.e.matrices["V"] = V_out
        ctx.state.e.matrix_colnames["b"] = names_out
        ctx.state.e.matrices["_sample"] = valid


class PredictCmd(BaseCommand):
    spec = CommandSpec("predict", 4, "Generate post-estimation predictions or residuals")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        if "cmd" not in ctx.state.e.macros or "b" not in ctx.state.e.matrices:
            raise StataError("last estimates not found", rc=301)
        newvar = cmd.main_args.strip()
        if not newvar:
            raise StataError("newvar required for predict", rc=100)
        if newvar in ctx.dataset.df.columns:
            raise StataError(f"variable {newvar} already defined", rc=110)

        ds = ctx.dataset
        est_cmd = ctx.state.e.macros["cmd"]
        depvar = ctx.state.e.macros["depvar"]
        rhs_terms = ctx.state.e.macros.get("_rhs_terms", "").split()
        noconst = ctx.state.e.macros.get("noconstant") == "1"
        beta = ctx.state.e.matrices["b"].ravel()
        V = ctx.state.e.matrices["V"]

        mask = compute_sample_mask(cmd, ds, ctx.state)
        # Build X on current dataset rows passing `if`/`in` and non-missing RHS
        valid = mask.copy()
        cols: list[np.ndarray] = []
        for t in rhs_terms:
            _, _, rvars = _expand_factor_term(t, ds.df, valid)
            for rv in rvars:
                valid &= ~ds.get_missing_mask(rv)
        for t in rhs_terms:
            t_cols, _, _ = _expand_factor_term(t, ds.df, valid)
            for c in t_cols:
                cols.append(c[valid])
        if not noconst:
            cols.append(np.ones(int(np.sum(valid)), dtype=np.float64))

        X = np.column_stack(cols) if cols else np.empty((int(np.sum(valid)), 0), dtype=np.float64)
        if X.shape[1] != len(beta):
            raise StataError("prediction design matrix dimension mismatch", rc=503)

        xb = X @ beta
        out = np.full(ds.nobs, np.nan, dtype=np.float64)

        if cmd.options.get("residuals") or cmd.options.get("resid"):
            y_vals, y_mc = ds.get_numeric(depvar)
            res_mask = valid & (y_mc == 0)
            # Re-index xb onto res_mask subset
            sub_idx = np.flatnonzero(valid)
            keep_sub = y_mc[sub_idx] == 0
            out[sub_idx[keep_sub]] = y_vals[sub_idx[keep_sub]] - xb[keep_sub]
        elif cmd.options.get("stdp"):
            var_xb = np.sum((X @ V) * X, axis=1)
            out[valid] = np.sqrt(np.maximum(0.0, var_xb))
        elif cmd.options.get("xb") or (est_cmd in ("regress", "xtreg") and not cmd.options.get("pr")):
            out[valid] = xb
        elif est_cmd == "logit":
            out[valid] = 1.0 / (1.0 + np.exp(-np.clip(xb, -30.0, 30.0)))
        elif est_cmd == "probit":
            out[valid] = stats.norm.cdf(xb)
        else:
            out[valid] = xb

        ds.add_numeric_var(newvar, out, stata_type="float")
