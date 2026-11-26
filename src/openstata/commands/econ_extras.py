"""esize twosample, lincom, Bayesian bootstrap, and mean/proportion/ratio/total."""

from __future__ import annotations

import re

import numpy as np
from scipy import stats

from openstata.core.errors import StataError
from openstata.commands.model_cmds import _print_coef_table
from openstata.lang.executor import compute_sample_mask
from openstata.lang.parser import ParsedCommand, split_top_level_tokens, strip_quotes
from openstata.plugins.base import BaseCommand, CommandSpec, ExecutionContext

EFFECT_CHOICES = ("cohend", "hedgesg", "glassdelta", "glassdelta2", "pbcorr")


class EsizeCmd(BaseCommand):
    """`esize twosample` — effect sizes for two independent samples."""

    spec = CommandSpec("esize", 5, "Effect sizes for two-sample comparisons")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        ds = ctx.dataset
        toks = split_top_level_tokens(cmd.main_args)
        if len(toks) < 2 or toks[0].lower() != "twosample":
            raise StataError('syntax: esize twosample varname, by(groupvar) | varname1 == varname2', rc=198)
        varname = toks[1]
        ds.require_var(varname)
        level = float(cmd.options.get("level", 95.0)) if isinstance(cmd.options.get("level"), str) else 95.0
        effect = str(cmd.options.get("effect", "cohend")).lower()
        if effect not in EFFECT_CHOICES:
            raise StataError(f"effect() must be one of {', '.join(EFFECT_CHOICES)}", rc=198)
        reverse = bool(cmd.options.get("reverse"))
        by_var = cmd.options.get("by")

        if isinstance(by_var, str) and by_var:
            ds.require_var(by_var)
            mask = compute_sample_mask(cmd, ds, ctx.state) & ~ds.get_missing_mask(varname)
            groups = sorted(
                ds.df.loc[mask & ds.df[by_var].notna(), by_var].unique().tolist()
            )
            if len(groups) != 2:
                raise StataError(f"by() variable must define exactly 2 nonmissing groups; found {len(groups)}", rc=198)
            v = ds.df.loc[mask]
            g1 = v.loc[v[by_var] == groups[0], varname].astype(float).values
            g2 = v.loc[v[by_var] == groups[1], varname].astype(float).values
            lbl = f"comparison of {varname} by {by_var}: {groups[0]} vs {groups[1]}"
        elif len(toks) >= 4 and toks[2] == "==":
            ds.require_var(toks[3])
            mask = compute_sample_mask(cmd, ds, ctx.state) & ~ds.get_missing_mask(varname) & ~ds.get_missing_mask(toks[3])
            a = ds.df.loc[mask, varname].astype(float).values
            b = ds.df.loc[mask, toks[3]].astype(float).values
            g1 = a[~np.isnan(a)]
            g2 = b[~np.isnan(b)]
            lbl = f"comparison of {varname} vs {toks[3]}"
        else:
            raise StataError('specify by(groupvar) or "varname1 == varname2"', rc=198)

        if reverse:
            g1, g2 = g2, g1
        n1, n2 = len(g1), len(g2)
        if n1 < 2 or n2 < 2:
            raise StataError("each group must have at least 2 observations", rc=2001)
        m1, m2 = float(np.mean(g1)), float(np.mean(g2))
        s1, s2 = float(np.std(g1, ddof=1)), float(np.std(g2, ddof=1))
        dfree = n1 + n2 - 2
        sp = float(np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / dfree))
        if sp == 0:
            raise StataError("pooled standard deviation is zero", rc=2001)

        est, se, df_ci = self._effect(effect, m1, m2, s1, s2, s2, sp, n1, n2, g1, g2)
        # Effect sizes are compared to 0 (standard normal / t reference)
        z = est / se if se > 0 else np.nan
        p = float(2 * stats.norm.sf(abs(z))) if se > 0 else np.nan
        crit = float(stats.norm.ppf(1 - (1 - level / 100) / 2))
        lo, hi = est - crit * se, est + crit * se

        ctx.console.write(f"Two-sample effect size ({lbl})")
        ctx.console.write(f"Effect size                Estimate             z         S.E.   [{int(level)}% conf. interval]")
        ctx.console.write("-" * 78)
        ctx.console.write(
            f"{effect:>16}" f"{est:>16.6f}" f"{z:>16.4f}" f"{se:>16.4f}" f"   {lo:>10.4f}   {hi:>10.4f}"
        )
        ctx.console.write("-" * 78)
        ctx.state.r.clear()
        ctx.state.r.scalars.update({"N": float(n1 + n2), "N_1": float(n1), "N_2": float(n2),
                                    "est": float(est), "se": float(se), "reverse": 1.0 if reverse else 0.0})
        ctx.state.r.macros["effect"] = effect

    def _effect(self, effect, m1, m2, s1, s2, s_control, sp, n1, n2, g1, g2):
        if effect == "cohend":
            est = (m2 - m1) / sp
            se = float(np.sqrt((n1 + n2) / (n1 * n2) + est**2 / (2 * (n1 + n2))))
        elif effect in ("hedgesg", "glassdelta", "glassdelta2"):
            if effect == "hedgesg":
                d = (m2 - m1) / sp
                J = 1 - 3 / (4 * (n1 + n2) - 9)
                est = d * J
                se = float(np.sqrt((n1 + n2) / (n1 * n2) + est**2 / (2 * (n1 + n2))))
            else:
                # glassdelta uses the second group's sd (control), glassdelta2 the first's
                s_ref = s2 if effect == "glassdelta" else s1
                if s_ref == 0:
                    raise StataError("reference-group standard deviation is zero", rc=2001)
                est = (m2 - m1) / s_ref
                se = float(np.sqrt((n1 + n2) / (n1 * n2) + est**2 / (2 * (n1 + n2))))
        else:  # pbcorr: point-biserial correlation of pooled values vs group indicator
            combined = np.concatenate([g1, g2])
            codes = np.concatenate([np.zeros(len(g1)), np.ones(len(g2))])
            r = float(np.corrcoef(combined, codes)[0, 1])
            est = r
            n = n1 + n2
            # Fisher-z SE
            se = float(np.sqrt(1 / max(n - 3, 1)))
        return est, se, None


class LincomCmd(BaseCommand):
    """`lincom` — linear combinations of estimated coefficients."""

    spec = CommandSpec("lincom", 6, "Point estimates, SEs for linear combinations of coefficients")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        e = ctx.state.e
        if "b" not in e.matrices or "V" not in e.matrices:
            raise StataError("last estimates not found", rc=301)
        names = e.matrix_colnames.get("b") or [f"c{i + 1}" for i in range(e.matrices["b"].size)]
        expr = cmd.main_args.strip()
        if not expr:
            raise StataError("expression required, e.g. lincom treated - time", rc=100)
        # Replace coefficient names (longest first) with safe placeholders
        ordered = sorted(names, key=len, reverse=True)
        pattern = "|".join(re.escape(nm) for nm in ordered)
        tokens: list[str] = []
        idx = 0
        while idx < len(expr):
            m = re.match(pattern, expr[idx:]) if pattern else None
            if m and (idx == 0 or not (expr[idx - 1].isalnum() or expr[idx - 1] in "_.#")):
                tokens.append(f"w{names.index(m.group(0))}")
                idx += len(m.group(0))
            else:
                tokens.append(expr[idx])
                idx += 1
        arithmetic = "".join(tokens)
        if not re.fullmatch(r"[w0-9\+\-\*\/\(\)\. ]*", arithmetic):
            raise StataError(f"invalid coefficient expression '{cmd.main_args}'", rc=198)
        k = len(names)
        l = np.zeros(k, dtype=np.float64)
        env = {f"w{i}": np.eye(k)[i] for i in range(k)}
        try:
            result = eval(arithmetic, {"__builtins__": {}}, dict(env))  # noqa: S307 (sandboxed tokens)
        except ZeroDivisionError:
            raise StataError("division by zero in coefficient expression", rc=198) from None
        except Exception as exc:
            raise StataError(f"invalid coefficient expression: {exc}", rc=198) from exc
        l = np.asarray(result, dtype=np.float64)
        if l.shape != (k,):
            raise StataError("invalid coefficient expression", rc=198)

        beta = e.matrices["b"].ravel()
        V = e.matrices["V"]
        est = float(l @ beta)
        var = float(l @ V @ l)
        se = float(np.sqrt(max(var, 0.0)))
        df_r = e.scalars.get("df_r")
        level = float(cmd.options.get("level", 95.0)) if isinstance(cmd.options.get("level"), str) else 95.0
        crit = (
            float(stats.t.ppf(1 - (1 - level / 100) / 2, df_r))
            if df_r is not None
            else float(stats.norm.ppf(1 - (1 - level / 100) / 2))
        )
        # coef option: report the raw combination; otherwise after logit show ORs
        exponentiate = (e.macros.get("cmd") == "logit") and not cmd.options.get("coef")
        disp_est, disp_lo, disp_hi = est, est - crit * se, est + crit * se
        if exponentiate:
            disp_est, disp_lo, disp_hi = np.exp(est), np.exp(est - crit * se), np.exp(est + crit * se)
        t = est / se if se > 0 else np.nan
        p = (
            float(2 * stats.t.sf(abs(t), df_r)) if df_r is not None and se > 0
            else float(2 * stats.norm.sf(abs(t))) if se > 0 else np.nan
        )
        label = "Odds ratio" if exponentiate else "(1)  " + cmd.main_args.strip()
        ctx.console.write(f"{label:>12} | {disp_est:>11.6g}" f"{(disp_est * se) if exponentiate else se:>10.6g}"
                          f"{t:>9.2f}  {p:>7.3f}   {disp_lo:>10.6g}   {disp_hi:>10.6g}")
        ctx.state.r.clear()
        ctx.state.r.scalars.update({"estimate": float(est), "se": float(se), "t": float(t), "pvalue": float(p)})
        if exponentiate:
            ctx.state.r.scalars["or"] = float(disp_est)


class BayesbootCmd(BaseCommand):
    """`bayesboot` — Bayesian bootstrap (Rubin) with Dirichlet replicate weights."""

    spec = CommandSpec("bayesboot", 9, "Bayesian bootstrap for linear regression")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        from openstata.commands.model_cmds import build_design_matrix

        ds = ctx.dataset
        toks = split_top_level_tokens(cmd.main_args)
        if not toks:
            raise StataError("depvar required", rc=100)
        depvar = toks[0]
        rhs_terms = toks[1:]
        reps = int(str(cmd.options.get("reps", "500")))
        if reps < 2:
            raise StataError("reps() must be at least 2", rc=198)
        level = float(cmd.options.get("level", 95.0)) if isinstance(cmd.options.get("level"), str) else 95.0
        seed = int(str(cmd.options.get("seed"))) if isinstance(cmd.options.get("seed"), str) else None
        rng = np.random.default_rng(seed)

        mask = compute_sample_mask(cmd, ds, ctx.state)
        y, X, names, valid = build_design_matrix(depvar, rhs_terms, ds, mask)
        n, k = X.shape
        betas = np.empty((reps, k), dtype=np.float64)
        for r in range(reps):
            # Dirichlet(1,...,1) via normalized exponentials - the replicate weights
            w = rng.exponential(scale=1.0, size=n)
            w /= w.sum()
            sw = np.sqrt(w)
            betas[r] = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]

        alpha = 1 - level / 100
        lo_q, hi_q = 100 * (alpha / 2), 100 * (1 - alpha / 2)
        ctx.console.write(f"Bayesian bootstrap                     Number of obs   = {n:>10}")
        ctx.console.write(f"Replications                           = {reps:>10}")
        ctx.console.write("-" * 78)
        ctx.console.write(f"{'':>12} | {'Mean':>11}  {'Std. dev.':>10}  {'[Observed]':>10}" f"  [{int(level)}% percentile interval]")
        ctx.console.write("-" * 78)
        observed = np.linalg.lstsq(X, y, rcond=None)[0]
        for i, nm in enumerate(names):
            mean_i = float(betas[:, i].mean())
            sd_i = float(betas[:, i].std(ddof=1))
            lo_i = float(np.percentile(betas[:, i], lo_q))
            hi_i = float(np.percentile(betas[:, i], hi_q))
            ctx.console.write(
                f"{nm:>12} | {mean_i:>11.6g}  {sd_i:>10.6g}  {observed[i]:>10.6g}   {lo_i:>10.6g}  {hi_i:>10.6g}"
            )
        ctx.console.write("-" * 78)
        ctx.state.r.clear()
        ctx.state.r.scalars.update({"N": float(n), "reps": float(reps)})
        ctx.state.r.matrices["mean"] = betas.mean(axis=0).reshape(1, -1)
        ctx.state.r.matrices["sd"] = betas.std(axis=0, ddof=1).reshape(1, -1)
        ctx.state.r.matrix_colnames["mean"] = names
        ctx.state.r.matrix_colnames["sd"] = names


class _GroupSummary(BaseCommand):
    """Shared engine for mean/proportion/ratio/total with over() and dofsubpop."""

    family = "mean"
    spec = CommandSpec("mean", 4, "Estimation of means/proportions/ratios/totals")

    def _parse(self, cmd: ParsedCommand, ctx: ExecutionContext) -> tuple[list[str], dict, float]:
        ds = ctx.dataset
        toks = split_top_level_tokens(cmd.main_args)
        over_var = cmd.options.get("over")
        over_var = strip_quotes(str(over_var)) if isinstance(over_var, str) and over_var else None
        dofsubpop = bool(cmd.options.get("dofsubpop"))
        level = float(cmd.options.get("level", 95.0)) if isinstance(cmd.options.get("level"), str) else 95.0
        varlist: list[str] = []
        if self.family == "ratio":
            # Accept both "num / den" and "num/den"
            joined = "".join(toks)
            if len(joined.split("/")) == 2 and all(joined.split("/")):
                varlist = [p.strip() for p in joined.split("/")]
            else:
                raise StataError("syntax: ratio numvar / denvar", rc=198)
        else:
            if not toks:
                raise StataError("varlist required", rc=100)
            varlist = toks
        for v in varlist:
            ds.require_var(v)
        return varlist, {"over": over_var, "dofsubpop": dofsubpop}, level

    def _compute_rows(self, ds, varlist, over_var, dofsubpop, level):
        """Yield rows (label, est, se, df) per variable (or variable x group)."""
        family = self.family
        rows_out: list[tuple[str, float, float, int]] = []
        if over_var:
            vals = ds.df[over_var].dropna().unique().tolist()
            groups = [(g, (ds.df[over_var] == g).fillna(False).to_numpy()) for g in sorted(vals, key=str)]
        else:
            groups = [(None, None)]

        # Whole-sample validity for the default df (unless dofsubpop)
        n_base_mask = ~ds.get_missing_mask(varlist[0])
        for v in varlist[1:]:
            n_base_mask &= ~ds.get_missing_mask(v)
        n_base = int(n_base_mask.sum())

        for glabel, gmask in groups:
            sub = n_base_mask.copy()
            if gmask is not None:
                sub &= gmask
            n_g = int(sub.sum())
            df = (n_g if dofsubpop else n_base) - 1
            vars_here = varlist if family != "ratio" else [varlist[0]]
            for v in vars_here:
                label = f"{varlist[0]}/{varlist[1]}" if family == "ratio" else v
                if over_var and glabel is not None:
                    label += f" @ {glabel}"
                if df < 1:
                    rows_out.append((label, np.nan, np.nan, df))
                    continue
                if family == "mean":
                    x = ds.df.loc[sub, v].to_numpy(float)
                    est = float(x.mean())
                    se = float(x.std(ddof=1) / np.sqrt(n_g))
                elif family == "total":
                    x = ds.df.loc[sub, v].to_numpy(float)
                    est = float(x.sum())
                    se = float(np.sqrt(n_g) * x.std(ddof=1))
                elif family == "proportion":
                    x_all = ds.df.loc[sub, v]
                    uniq_vals = np.unique(x_all.dropna().to_numpy())
                    if len(uniq_vals) == 2 and set(np.unique(x_all.astype(float))) <= {0.0, 1.0}:
                        # Binary indicator: single proportion of 1s
                        est = float(x_all.mean())
                        se = float(np.sqrt(max(est * (1 - est), 0.0) / n_g))
                    else:
                        # Categorical: one row per value, Stata-style
                        for val in uniq_vals:
                            mval = sub & (ds.df[v] == val).fillna(False).to_numpy()
                            p_v = float(mval.sum()) / n_g
                            se_v = float(np.sqrt(max(p_v * (1 - p_v), 0.0) / n_g))
                            if over_var and glabel is not None:
                                vlabel = f"{v} @ {val} @ {glabel}"
                            else:
                                vlabel = f"{v} @ {val}"
                            rows_out.append((vlabel, p_v, se_v, df))
                        continue
                else:  # ratio of means, delta-method SE
                    num = ds.df.loc[sub, varlist[0]].to_numpy(float)
                    den = ds.df.loc[sub, varlist[1]].to_numpy(float)
                    mn, md = float(num.mean()), float(den.mean())
                    R = mn / md if md != 0 else np.nan
                    vn = float(num.var(ddof=1))
                    vd = float(den.var(ddof=1))
                    covnd = float(np.cov(num, den, ddof=1)[0, 1])
                    est = R
                    se = (
                        float(np.sqrt(max(vn - 2 * R * covnd + R**2 * vd, 0.0) / n_g)) / (abs(md) if md != 0 else 1.0)
                    )
                rows_out.append((label, est, se, df))
        return rows_out

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        ds = ctx.dataset
        varlist, opts, level = self._parse(cmd, ctx)
        rows = self._compute_rows(ds, varlist, opts["over"], opts["dofsubpop"], level)
        crit_cache: dict[int, float] = {}
        ctx.console.write(f"{self.family.capitalize() + ' estimation'}" + ("" if not opts["over"] else f"  over: {opts['over']}"))
        ctx.console.write("-" * 78)
        ctx.console.write(f"{'':>20} | {'Estimate':>11}  {'Std. err.':>10}  {'t':>7}  {'P>|t|':>7}  [{int(level)}% conf. interval]")
        ctx.console.write("-" * 78)
        for label, est, se, df in rows:
            if df < 1 or np.isnan(est):
                ctx.console.write(f"{label:>20} | {'.':>11}")
                continue
            if df not in crit_cache:
                crit_cache[df] = float(stats.t.ppf(1 - (1 - level / 100) / 2, df))
            crit = crit_cache[df]
            t = est / se if se > 0 else np.nan
            p = float(2 * stats.t.sf(abs(t), df)) if se > 0 else np.nan
            ctx.console.write(
                f"{label:>20} | {est:>11.6g}  {se:>10.6g}  {t:>7.2f}  {p:>7.3f}   {est - crit * se:>10.6g}   {est + crit * se:>10.6g}"
            )
        ctx.console.write("-" * 78)
        ctx.state.r.clear()
        for i, (label, est, se, df) in enumerate(rows, start=1):
            ctx.state.r.scalars[f"est{i}"] = float(est)
            ctx.state.r.scalars[f"se{i}"] = float(se)
        ctx.state.r.macros["cmd"] = self.family


class MeanCmd(_GroupSummary):
    family = "mean"
    spec = CommandSpec("mean", 4, "Estimate means, optionally by over() groups")


class ProportionCmd(_GroupSummary):
    family = "proportion"
    spec = CommandSpec("proportion", 4, "Estimate proportions of binary indicators")


class RatioCmd(_GroupSummary):
    family = "ratio"
    spec = CommandSpec("ratio", 5, "Estimate the ratio of two means (delta method)")


class TotalCmd(_GroupSummary):
    family = "total"
    spec = CommandSpec("total", 5, "Estimate totals, optionally by over() groups")
