"""Descriptive statistics commands: summarize, tabulate, correlate, ttest."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from openstata.core.errors import StataError
from openstata.lang.executor import compute_sample_mask
from openstata.lang.expr import eval_expression
from openstata.lang.parser import ParsedCommand, expand_varlist, split_top_level_tokens
from openstata.plugins.base import BaseCommand, CommandSpec, ExecutionContext


def _get_weights(cmd: ParsedCommand, ctx: ExecutionContext, mask: np.ndarray) -> np.ndarray | None:
    if cmd.weight is None:
        return None
    wtype, wexpr = cmd.weight
    res = eval_expression(wexpr, ctx.dataset, ctx.state)
    if res.is_string:
        raise StataError("weight expression must be numeric", rc=109)
    w = np.asarray(res.values, dtype=np.float64)
    w = np.where(np.isnan(w) | (w <= 0), 0.0, w)
    return w


class SummarizeCmd(BaseCommand):
    spec = CommandSpec("summarize", 2, "Summary statistics (mean, sd, min, max, percentiles)")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        ds = ctx.dataset
        vlist = expand_varlist(cmd.main_args, ds) if cmd.main_args.strip() else ds.columns
        mask = compute_sample_mask(cmd, ds, ctx.state)
        weights = _get_weights(cmd, ctx, mask)
        detail = bool(cmd.options.get("detail") or cmd.options.get("d"))

        if not detail:
            ctx.console.write(f"    {'Variable':<14} | {'Obs':>8}  {'Mean':>11}  {'Std. dev.':>11}  {'Min':>11}  {'Max':>11}")
            ctx.console.write("-" * 75)

        ctx.state.r.clear()
        for v in vlist:
            if ds.is_string(v):
                if not detail:
                    ctx.console.write(f"    {v:<14} | {'0':>8}")
                continue
            vals, mc = ds.get_numeric(v)
            valid = mask & (mc == 0)
            if weights is not None:
                valid &= weights > 0
            x = vals[valid]
            n = len(x)
            if n == 0:
                if not detail:
                    ctx.console.write(f"    {v:<14} | {'0':>8}")
                ctx.state.r.scalars["N"] = 0.0
                continue

            if weights is not None:
                w = weights[valid]
                wsum = float(np.sum(w))
                mean = float(np.sum(w * x) / wsum)
                var = float(np.sum(w * (x - mean) ** 2) / (wsum * (n - 1) / n)) if n > 1 else 0.0
                sd = float(np.sqrt(max(0.0, var)))
                total = float(np.sum(w * x))
            else:
                mean = float(np.mean(x))
                var = float(np.var(x, ddof=1)) if n > 1 else 0.0
                sd = float(np.sqrt(var))
                total = float(np.sum(x))

            vmin, vmax = float(np.min(x)), float(np.max(x))
            ctx.state.r.scalars["N"] = float(n)
            ctx.state.r.scalars["mean"] = mean
            ctx.state.r.scalars["Var"] = var
            ctx.state.r.scalars["sd"] = sd
            ctx.state.r.scalars["min"] = vmin
            ctx.state.r.scalars["max"] = vmax
            ctx.state.r.scalars["sum"] = total
            ctx.state.r.scalars["sum_w"] = float(np.sum(weights[valid])) if weights is not None else float(n)

            if not detail:
                ctx.console.write(
                    f"    {v:<14} | {n:>8,}  {mean:>11.6g}  {sd:>11.6g}  {vmin:>11.6g}  {vmax:>11.6g}"
                )
            else:
                pcts = np.percentile(x, [1, 5, 10, 25, 50, 75, 90, 95, 99])
                for pname, pval in zip(["p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"], pcts):
                    ctx.state.r.scalars[pname] = float(pval)
                skew = float(stats.skew(x, bias=True)) if n > 2 and sd > 0 else np.nan
                kurt = float(stats.kurtosis(x, fisher=False, bias=True)) if n > 3 and sd > 0 else np.nan
                ctx.state.r.scalars["skewness"] = skew
                ctx.state.r.scalars["kurtosis"] = kurt
                ctx.console.write(f"                             {v}")
                ctx.console.write("-" * 62)
                ctx.console.write(f"      Percentiles      Smallest")
                ctx.console.write(f"  1%   {pcts[0]:>10.5g}    {vmin:>10.5g}       Obs        {n:>10,}")
                ctx.console.write(f"  5%   {pcts[1]:>10.5g}                     Mean       {mean:>10.5g}")
                ctx.console.write(f" 10%   {pcts[2]:>10.5g}                     Std. dev.  {sd:>10.5g}")
                ctx.console.write(f" 25%   {pcts[3]:>10.5g}                     Variance   {var:>10.5g}")
                ctx.console.write(f" 50%   {pcts[4]:>10.5g}                     Skewness   {skew:>10.5g}")
                ctx.console.write(f" 75%   {pcts[5]:>10.5g}                     Kurtosis   {kurt:>10.5g}")
                ctx.console.write(f" 90%   {pcts[6]:>10.5g}")
                ctx.console.write(f" 95%   {pcts[7]:>10.5g}")
                ctx.console.write(f" 99%   {pcts[8]:>10.5g}    {vmax:>10.5g}")


class TabulateCmd(BaseCommand):
    spec = CommandSpec("tabulate", 2, "One-way and two-way frequency tables")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        ds = ctx.dataset
        vlist = expand_varlist(cmd.main_args, ds)
        if len(vlist) not in (1, 2):
            raise StataError("tabulate accepts 1 or 2 variables", rc=198)
        mask = compute_sample_mask(cmd, ds, ctx.state)
        show_missing = bool(cmd.options.get("missing") or cmd.options.get("m"))

        ctx.state.r.clear()
        if len(vlist) == 1:
            v = vlist[0]
            sub = ds.df.loc[mask, v]
            if not show_missing:
                sub = sub[~ds.get_missing_mask(v)[mask]]
            counts = sub.value_counts(dropna=False).sort_index()
            total = int(counts.sum())
            ctx.console.write(f"{v:<16} | {'Freq.':>10} {'Percent':>10} {'Cum.':>10}")
            ctx.console.write("-" * 52)
            cum = 0.0
            for k, cnt in counts.items():
                pct = 100.0 * cnt / total if total else 0.0
                cum += pct
                lbl = "." if pd.isna(k) else (str(int(k)) if isinstance(k, (int, float)) and float(int(k)) == k else str(k))
                ctx.console.write(f"{lbl:<16} | {int(cnt):>10,} {pct:>9.2f}% {cum:>9.2f}%")
            ctx.console.write("-" * 52)
            ctx.console.write(f"{'Total':<16} | {total:>10,} {100.0:>9.2f}%")
            ctx.state.r.scalars["N"] = float(total)
            ctx.state.r.scalars["r"] = float(len(counts))
            return

        v1, v2 = vlist[0], vlist[1]
        valid = mask.copy()
        if not show_missing:
            valid &= (~ds.get_missing_mask(v1)) & (~ds.get_missing_mask(v2))
        ct = pd.crosstab(ds.df.loc[valid, v1], ds.df.loc[valid, v2], margins=True, margins_name="Total")
        ctx.console.write(ct.to_string())
        n_total = int(ct.loc["Total", "Total"]) if "Total" in ct.index else 0
        ctx.state.r.scalars["N"] = float(n_total)
        ctx.state.r.scalars["r"] = float(ct.shape[0] - 1)
        ctx.state.r.scalars["c"] = float(ct.shape[1] - 1)
        if cmd.options.get("chi2") and ct.shape[0] > 2 and ct.shape[1] > 2:
            obs_mat = ct.iloc[:-1, :-1].to_numpy(dtype=np.float64)
            chi2_val, p_val, dof, _ = stats.chi2_contingency(obs_mat, correction=False)
            ctx.console.write(f"\n          Pearson chi2({dof}) = {chi2_val:8.4f}   Pr = {p_val:5.3f}")
            ctx.state.r.scalars["chi2"] = float(chi2_val)
            ctx.state.r.scalars["p"] = float(p_val)


class CorrelateCmd(BaseCommand):
    spec = CommandSpec("correlate", 3, "Pairwise / listwise correlation or covariance matrix")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        ds = ctx.dataset
        vlist = expand_varlist(cmd.main_args, ds) if cmd.main_args.strip() else [
            c for c in ds.columns if not ds.is_string(c)
        ]
        mask = compute_sample_mask(cmd, ds, ctx.state)
        sub = ds.df.loc[mask, vlist].dropna()
        n = len(sub)
        if n < 2:
            raise StataError("no observations", rc=2000)
        mat = sub.cov().to_numpy() if cmd.options.get("covariance") else sub.corr().to_numpy()
        ctx.console.write(f"(obs={n})")
        ctx.console.write("")
        df_mat = pd.DataFrame(mat, index=vlist, columns=vlist)
        ctx.console.write(df_mat.round(4).to_string())
        ctx.state.r.clear()
        ctx.state.r.scalars["N"] = float(n)
        ctx.state.r.matrices["C"] = mat


class TtestCmd(BaseCommand):
    spec = CommandSpec("ttest", 5, "One-sample, two-sample, or paired Student's t-test")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        ds = ctx.dataset
        mask = compute_sample_mask(cmd, ds, ctx.state)
        by_opt = cmd.options.get("by")
        ctx.state.r.clear()

        if isinstance(by_opt, str):
            var = expand_varlist(cmd.main_args, ds)[0]
            gvar = expand_varlist([by_opt], ds)[0]
            valid = mask & (~ds.get_missing_mask(var)) & (~ds.get_missing_mask(gvar))
            groups = ds.df.loc[valid, gvar].unique()
            if len(groups) != 2:
                raise StataError(f"by({gvar}) must define exactly 2 groups", rc=420)
            groups = sorted(groups)
            x1 = ds.df.loc[valid & (ds.df[gvar] == groups[0]), var].to_numpy(dtype=np.float64)
            x2 = ds.df.loc[valid & (ds.df[gvar] == groups[1]), var].to_numpy(dtype=np.float64)
            equal_var = not bool(cmd.options.get("unequal"))
            res = stats.ttest_ind(x1, x2, equal_var=equal_var)
            df_t = float(res.df) if hasattr(res, "df") else float(len(x1) + len(x2) - 2)
            p_two = float(res.pvalue)
            t_val = float(res.statistic)
            ctx.console.write(f"Two-sample t test with {'equal' if equal_var else 'unequal'} variances")
            ctx.console.write(f"  Group {groups[0]}: N={len(x1)}, mean={np.mean(x1):.6g}, sd={np.std(x1, ddof=1):.6g}")
            ctx.console.write(f"  Group {groups[1]}: N={len(x2)}, mean={np.mean(x2):.6g}, sd={np.std(x2, ddof=1):.6g}")
            ctx.console.write(f"  diff = mean({groups[0]}) - mean({groups[1]}) = {np.mean(x1) - np.mean(x2):.6g}")
            ctx.console.write(f"  t = {t_val:.4f}   df = {df_t:.2f}   Pr(|T| > |t|) = {p_two:.4f}")
            ctx.state.r.scalars.update(
                {
                    "N_1": float(len(x1)),
                    "N_2": float(len(x2)),
                    "mu_1": float(np.mean(x1)),
                    "mu_2": float(np.mean(x2)),
                    "sd_1": float(np.std(x1, ddof=1)),
                    "sd_2": float(np.std(x2, ddof=1)),
                    "t": t_val,
                    "df_t": df_t,
                    "p": p_two,
                }
            )
            return

        if "==" in cmd.main_args:
            left, right = cmd.main_args.split("==", 1)
            v1 = expand_varlist([left.strip()], ds)[0]
            rhs = right.strip()
            if rhs in ds.df.columns:
                v2 = expand_varlist([rhs], ds)[0]
                valid = mask & (~ds.get_missing_mask(v1)) & (~ds.get_missing_mask(v2))
                x1 = ds.df.loc[valid, v1].to_numpy(dtype=np.float64)
                x2 = ds.df.loc[valid, v2].to_numpy(dtype=np.float64)
                res = stats.ttest_rel(x1, x2)
                df_t = float(len(x1) - 1)
                t_val, p_two = float(res.statistic), float(res.pvalue)
                ctx.console.write(f"Paired t test: {v1} vs {v2} (N={len(x1)})")
                ctx.console.write(f"  t = {t_val:.4f}   df = {df_t:.0f}   Pr(|T| > |t|) = {p_two:.4f}")
                ctx.state.r.scalars.update({"N_1": float(len(x1)), "t": t_val, "df_t": df_t, "p": p_two})
            else:
                mu = float(eval_expression(rhs, ds, ctx.state, force_scalar=True).to_scalar())
                valid = mask & (~ds.get_missing_mask(v1))
                x1 = ds.df.loc[valid, v1].to_numpy(dtype=np.float64)
                res = stats.ttest_1samp(x1, popmean=mu)
                df_t = float(len(x1) - 1)
                t_val, p_two = float(res.statistic), float(res.pvalue)
                ctx.console.write(f"One-sample t test: {v1} == {mu:g} (N={len(x1)}, mean={np.mean(x1):.6g})")
                ctx.console.write(f"  t = {t_val:.4f}   df = {df_t:.0f}   Pr(|T| > |t|) = {p_two:.4f}")
                ctx.state.r.scalars.update(
                    {"N_1": float(len(x1)), "mu_1": float(np.mean(x1)), "t": t_val, "df_t": df_t, "p": p_two}
                )
            return

        raise StataError("syntax: ttest var == # | var1 == var2 | var, by(groupvar)", rc=198)
