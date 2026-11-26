"""areg (absorbed fixed effects), ivregress 2sls with weak-IV robust inference,
and control-function linear/probit estimators."""

from __future__ import annotations

import numpy as np
from scipy import stats

from openstata.core.errors import StataError
from openstata.commands.model_cmds import _expand_factor_term, _parse_vce, _print_coef_table
from openstata.lang.executor import compute_sample_mask
from openstata.lang.parser import ParsedCommand, split_top_level_tokens, strip_quotes
from openstata.plugins.base import BaseCommand, CommandSpec, ExecutionContext


def _absorb_groups(term: str, df, valid: np.ndarray) -> tuple[np.ndarray, int]:
    """Group codes and number of absorbed parameters for one absorb() term.

    Accepted: plain variable, i.var, or i.a#i.b (interaction of categoricals).
    A continuous variable absorbed in Stata gets one FE per distinct value,
    which is what grouping by the raw variable does.
    """
    if "#" in term:
        key = None
        for part in term.split("#"):
            vv = part[2:] if part.lower().startswith(("i.", "c.")) else part
            if vv not in df.columns:
                raise StataError(f"variable {vv} not found", rc=111)
            codes = df.loc[valid, vv].astype(str).to_numpy()
            key = codes if key is None else np.char.add(np.char.add(key, "|"), codes)
        uniq, inverse = np.unique(key, return_inverse=True)
        return inverse.astype(np.float64), max(len(uniq) - 1, 0)

    varname = term[2:] if term.lower().startswith(("i.", "c.")) else term
    if varname not in df.columns:
        raise StataError(f"variable {varname} not found", rc=111)
    codes = df.loc[valid, varname].astype(str).to_numpy()
    uniq, inverse = np.unique(codes, return_inverse=True)
    return inverse.astype(np.float64), max(len(uniq) - 1, 0)


def _within_demean(y: np.ndarray, X: np.ndarray, dim_list: list[tuple[np.ndarray, int]]) -> tuple[np.ndarray, np.ndarray, int]:
    """Iterative demeaning of y and X over each absorbed dimension."""
    yd, Xd = y.copy(), X.copy()
    for _ in range(200):
        max_change = 0.0
        for codes, _params in dim_list:
            uniq = np.unique(codes)
            # Group means via searchsorted-based accumulation
            codes_int = codes.astype(np.int64)
            means_y = np.zeros(len(uniq))
            means_X = np.zeros((len(uniq), Xd.shape[1]))
            counts = np.bincount(codes_int, minlength=len(uniq)).astype(float)
            np.add.at(means_y, codes_int, yd)
            np.add.at(means_X, codes_int, Xd)
            means_y /= counts
            means_X /= counts[:, None]
            yd -= means_y[codes_int]
            Xd -= means_X[codes_int]
            max_change = max(
                max_change,
                float(np.abs(means_y * counts).max(initial=0.0)),
                float(np.abs(means_X * counts[:, None]).max(initial=0.0)),
            )
        if max_change < 1e-10:
            break
    df_a = sum(max(params, 0) for _c, params in dim_list)
    return yd, Xd, df_a


def _onehot(codes: np.ndarray) -> np.ndarray:
    uniq = np.unique(codes)
    idx = np.searchsorted(uniq, codes)
    out = np.zeros((len(codes), len(uniq)))
    out[np.arange(len(codes)), idx] = 1.0
    return out


def _sandwich_vce(X: np.ndarray, resid: np.ndarray, kind: str, cl_codes: np.ndarray | None, df_r: float):
    n, k = X.shape
    xtx_pinv = np.linalg.pinv(X.T @ X)
    if kind == "robust":
        meat = X.T @ ((resid**2)[:, None] * X)
        return (n / df_r) * (xtx_pinv @ meat @ xtx_pinv)
    if kind == "cluster":
        uniq_c, inv_idx = np.unique(cl_codes, return_inverse=True)
        G = len(uniq_c)
        sums = np.zeros((G, k))
        np.add.at(sums, inv_idx, X * resid[:, None])
        V = (G / (G - 1.0)) * ((n - 1.0) / df_r) * (xtx_pinv @ (sums.T @ sums) @ xtx_pinv)
        return V
    return (float(resid @ resid) / df_r) * xtx_pinv


def _store_e(ctx: ExecutionContext, cmd_name: str, depvar: str, names: list[str],
             beta: np.ndarray, V: np.ndarray, valid: np.ndarray,
             scalars: dict[str, float]) -> None:
    ctx.state.e.clear()
    ctx.state.e.scalars.update(scalars)
    ctx.state.e.macros.update({"cmd": cmd_name, "depvar": depvar, "indepvars": " ".join(names)})
    ctx.state.e.matrices["b"] = beta.reshape(1, -1)
    ctx.state.e.matrices["V"] = V
    ctx.state.e.matrices["_sample"] = valid
    ctx.state.e.matrix_colnames["b"] = names
    ctx.state.e.matrix_colnames["V"] = names


class AregCmd(BaseCommand):
    """`areg` — linear regression with absorb()ed fixed effects."""

    spec = CommandSpec("areg", 4, "Linear regression with absorb()ed fixed effects")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        ds = ctx.dataset
        toks = split_top_level_tokens(cmd.main_args)
        if not toks:
            raise StataError("depvar required", rc=100)
        depvar = toks[0]
        ds.require_var(depvar)
        rhs_terms = toks[1:]
        absorb_raw = cmd.options.get("absorb")
        if not isinstance(absorb_raw, str) or not absorb_raw.strip():
            raise StataError("option absorb() required for areg", rc=198)
        absorb_terms = [t.strip() for t in strip_quotes(absorb_raw).split() if t.strip()]
        dfabsorb = bool(cmd.options.get("dfabsorb"))
        vce_kind, cl_var = _parse_vce(cmd)
        level = float(cmd.options.get("level", 95.0)) if isinstance(cmd.options.get("level"), str) else 95.0

        mask = compute_sample_mask(cmd, ds, ctx.state)
        valid = mask & ~ds.get_missing_mask(depvar)

        # Expand RHS terms, filtering observations with missing raw variables
        def expand_terms(terms):
            cols, names, rvars = [], [], []
            for t in terms:
                c, nm, rv = _expand_factor_term(t, ds.df, valid)
                cols.extend(c)
                names.extend(nm)
                rvars.extend(rv)
            return cols, names, rvars

        _, _, rhs_rvars = expand_terms(rhs_terms)
        for rv in dict.fromkeys(rhs_rvars):
            ds.require_var(rv)
            valid &= ~ds.get_missing_mask(rv)
        rhs_cols, rhs_names, _ = expand_terms(rhs_terms)

        dim_list = [_absorb_groups(t, ds.df, valid) for t in absorb_terms]
        y = ds.df.loc[valid, depvar].to_numpy(dtype=np.float64)
        X = np.column_stack([c[valid] for c in rhs_cols]) if rhs_cols else np.empty((int(valid.sum()), 0))
        n, k = X.shape

        yd, Xd, df_a = _within_demean(y, X, dim_list)
        if dfabsorb:
            # Replace #levels-1 per term with the rank of the absorbed indicators
            A = np.hstack([_onehot(codes) for codes, _ in dim_list])
            df_a = max(int(np.linalg.matrix_rank(A)) - 1, 0)

        if n <= k + df_a + 1:
            raise StataError("insufficient observations after absorbing", rc=2001)
        beta_t = np.linalg.lstsq(Xd, yd, rcond=None)[0]
        resid = yd - Xd @ beta_t
        # + 1: the constant absorbs one more degree of freedom (grand mean)
        df_r = n - k - df_a - 1
        b_cons = float(y.mean() - X.mean(axis=0) @ beta_t)
        names = rhs_names + ["_cons"]
        beta = np.concatenate([beta_t, [b_cons]])
        V_t = _sandwich_vce(
            Xd, resid, vce_kind,
            ds.df.loc[valid, cl_var].to_numpy() if cl_var else None,
            df_r,
        )
        se_rhs = np.sqrt(np.maximum(0.0, np.diag(V_t)))
        V_full = np.zeros((k + 1, k + 1))
        V_full[:k, :k] = V_t
        se = np.concatenate([se_rhs, [np.nan]])

        vcetype = {"robust": "Robust", "cluster": "Robust"}.get(vce_kind, "OLS")
        ctx.console.write(
            f"Linear regression absorbing fixed effects ({vcetype})   Number of obs = {n:>8}"
        )
        ctx.console.write(f"                                        Absorbed df    = {df_a:>8}")
        if vce_kind == "cluster":
            G = len(np.unique(ds.df.loc[valid, cl_var]))
            ctx.console.write(
                f"                                (Std. err. adjusted for {G} clusters in {cl_var})"
            )
        _print_coef_table(ctx, depvar, names, beta, se, df_t=df_r, level=level)
        _store_e(ctx, "areg", depvar, names, beta, V_full, valid,
                 {"N": float(n), "df_a": float(df_a), "df_r": float(df_r),
                  "N_absorbed": float(sum(len(np.unique(c)) for c, _ in dim_list))})
        ctx.state.e.macros["_rhs_terms"] = " ".join(rhs_terms)


def _parse_iv_design(src: str) -> tuple[str, list[str], list[str], list[str]]:
    """Parse 'depvar exogvars (endogvars = instruments) ...'."""
    depth = 0
    split_idx = None
    for pos, ch in enumerate(src):
        if ch == "(":
            if depth == 0 and split_idx is None:
                split_idx = pos
            depth += 1
        elif ch == ")":
            depth -= 1
    if split_idx is None:
        head = src.split()
        if not head:
            raise StataError("depvar required", rc=100)
        return head[0], head[1:], [], []
    head = src[:split_idx].split()
    if not head:
        raise StataError("depvar required", rc=100)
    exog = head[1:]
    blocks: list[str] = []
    depth = 0
    cur = ""
    tail = ""  # variables outside parens after the first '(' are exogenous too
    for ch in src[split_idx:]:
        if ch == "(":
            depth += 1
            cur = ""
        elif ch == ")":
            depth -= 1
            if depth == 0:
                blocks.append(cur)
                cur = ""
        elif depth > 0:
            cur += ch
        else:
            tail += ch
    if not blocks:
        raise StataError("endogenous variable block must be (endogvars = instruments)", rc=198)
    endog: list[str] = []
    inst: list[str] = []
    for block in blocks:
        if "=" not in block:
            raise StataError("endogenous variable block must be (endogvars = instruments)", rc=198)
        left, right = block.split("=", 1)
        endog.extend(left.split())
        inst.extend(right.split())
    exog.extend(tail.split())
    return head[0], exog, endog, inst


class IvregressCmd(BaseCommand):
    """`ivregress 2sls` with first-stage F and Anderson-Rubin weak-IV test."""

    spec = CommandSpec("ivregress", 5, "Instrumental-variables 2SLS with weak-IV diagnostics")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        from openstata.commands.model_cmds import build_design_matrix

        ds = ctx.dataset
        raw = cmd.main_args
        toks = split_top_level_tokens(raw)
        if not toks or toks[0].lower() != "2sls":
            raise StataError("only 'ivregress 2sls' is currently supported", rc=198)
        depvar, exog, endog, inst = _parse_iv_design(" ".join(toks[1:]))
        if not endog or not inst:
            raise StataError("both endogenous variables and instruments are required", rc=198)

        robust_flag = bool(cmd.options.get("robust") or cmd.options.get("r"))
        cluster = cmd.options.get("cluster")
        cluster = strip_quotes(str(cluster)) if isinstance(cluster, str) and cluster else None
        level = float(cmd.options.get("level", 95.0)) if isinstance(cmd.options.get("level"), str) else 95.0

        mask = compute_sample_mask(cmd, ds, ctx.state)
        extra = [cluster] if cluster else []
        # X: [exog..., endog..., _cons] (build_design_matrix appends _cons last)
        y, X, names, valid = build_design_matrix(depvar, exog + endog, ds, mask, extra_vars=extra)
        # Excluded instruments matrix (no constant)
        _yph, Z_exc, inst_names, _ = build_design_matrix(inst[0], inst, ds, mask, noconstant=True, extra_vars=extra)
        n, k = X.shape
        endog_idx = [names.index(v) for v in endog]
        exog_idx = [i for i in range(k) if i not in endog_idx]

        # Z = [exog part of X inclusive of _cons] + excluded instruments
        Z = np.hstack([X[:, exog_idx], Z_exc])
        ZtZ_pinv = np.linalg.pinv(Z.T @ Z)
        PzX = Z @ (ZtZ_pinv @ (Z.T @ X))
        XtPzX = X.T @ PzX
        XtPzX_pinv = np.linalg.pinv(XtPzX)
        b = XtPzX_pinv @ (PzX.T @ y)
        resid = y - X @ b
        df_r = n - k

        if robust_flag or cluster:
            meat = PzX.T @ ((resid**2)[:, None] * PzX)
            if cluster:
                codes = ds.df.loc[valid, cluster].to_numpy()
                uniq_c, inv_idx = np.unique(codes, return_inverse=True)
                G = len(uniq_c)
                sums = np.zeros((G, k))
                np.add.at(sums, inv_idx, PzX * resid[:, None])
                meat = sums.T @ sums
            scale = 1.0
            if cluster:
                G = len(np.unique(ds.df.loc[valid, cluster].to_numpy()))
                scale = (G / (G - 1.0)) * ((n - 1.0) / df_r)
            elif robust_flag:
                scale = n / df_r
            V = scale * (XtPzX_pinv @ meat @ XtPzX_pinv)
            n_clust = len(np.unique(ds.df.loc[valid, cluster].to_numpy())) if cluster else None
        else:
            s2 = float(resid @ resid) / df_r
            # X'Pz Pz X = (PzX)'(PzX)
            V = s2 * (XtPzX_pinv @ (PzX.T @ PzX) @ XtPzX_pinv)
            n_clust = None
        se = np.sqrt(np.maximum(0.0, np.diag(V)))
        vcetype = "Robust" if (robust_flag or cluster) else "OLS"

        ctx.console.write(f"Instrumental variables 2SLS ({vcetype})" f"   Number of obs = {n:>10}")
        if n_clust is not None:
            ctx.console.write(f"                                (Std. err. adjusted for {n_clust} clusters in {cluster})")

        # First-stage diagnostics: F for joint significance of excluded instruments
        k_excl = Z_exc.shape[1]
        R = np.zeros((k_excl, Z.shape[1]))
        for j in range(k_excl):
            R[j, Z.shape[1] - k_excl + j] = 1.0
        for v in endog:
            Y_end = ds.df.loc[valid, v].to_numpy(float)
            b_fs = np.linalg.lstsq(Z, Y_end, rcond=None)[0]
            Rb = R @ b_fs
            resid_fs = Y_end - Z @ b_fs
            s2_fs = float(resid_fs @ resid_fs) / (n - Z.shape[1])
            var_fs = s2_fs * ZtZ_pinv
            f_fs = float(Rb @ np.linalg.pinv(R @ var_fs @ R.T) @ Rb) / k_excl
            p_fs = float(stats.f.sf(f_fs, k_excl, n - Z.shape[1]))
            note = "   (weak: F < 10)" if f_fs < 10 else ""
            ctx.console.write(f"First-stage F ({v}):  {f_fs:>8.2f}   p = {p_fs:.4f}{note}")

        # Anderson-Rubin test on the reduced form: weak-IV robust for
        # H0: coefficient(s) of the endogenous variable(s) are zero
        b_rf = np.linalg.lstsq(Z, y, rcond=None)[0]
        resid_rf = y - Z @ b_rf
        s2_rf = float(resid_rf @ resid_rf) / (n - Z.shape[1])
        var_rf = s2_rf * ZtZ_pinv
        f_ar = float((R @ b_rf) @ np.linalg.pinv(R @ var_rf @ R.T) @ (R @ b_rf)) / k_excl
        p_ar = float(stats.f.sf(f_ar, k_excl, n - Z.shape[1]))
        ctx.console.write(f"Anderson-Rubin weak-IV test (endog=0):  F = {f_ar:>8.2f}   p = {p_ar:.4f}")

        _print_coef_table(ctx, depvar, names, b, se, df_t=df_r, level=level)
        _store_e(ctx, "ivregress", depvar, names, b, V, valid,
                 {"N": float(n), "df_r": float(df_r), "df_m": float(k - 1)})
        ctx.state.e.macros["_rhs_terms"] = " ".join(exog + endog)


class CfCmd(BaseCommand):
    """Control-function estimators: `cfregress` (linear) and `cfprobit`."""

    def __init__(self, family: str) -> None:
        self.family = family
        self.spec = CommandSpec(
            f"cf{family}",
            2,
            f"Control-function {'linear regression' if family == 'regress' else 'probit'} estimator",
        )

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        from openstata.commands.model_cmds import build_design_matrix

        ds = ctx.dataset
        toks = split_top_level_tokens(cmd.main_args)
        if not toks:
            raise StataError("depvar required", rc=100)
        depvar, exog, endog, inst = _parse_iv_design(cmd.main_args)
        if not endog or not inst:
            raise StataError(f"syntax: cf{self.family} depvar exogvars (endog = instruments)", rc=198)
        level = float(cmd.options.get("level", 95.0)) if isinstance(cmd.options.get("level"), str) else 95.0

        mask = compute_sample_mask(cmd, ds, ctx.state)
        Y, X, names, valid = build_design_matrix(depvar, exog + endog, ds, mask)
        _yph, Z_exc, _inst_names, _ = build_design_matrix(inst[0], inst, ds, mask, noconstant=True)
        n = X.shape[0]

        # First stage: endog on exogenous controls + instruments
        vhat = np.zeros((n, len(endog)))
        for j, v in enumerate(endog):
            endog_y = ds.df.loc[valid, v].to_numpy(float)
            X_exog = np.delete(X, names.index(v), axis=1)
            Z1 = np.hstack([X_exog, Z_exc])
            b1 = np.linalg.lstsq(Z1, endog_y, rcond=None)[0]
            vhat[:, j] = endog_y - Z1 @ b1

        X_aug = np.hstack([X, vhat])
        aug_names = names + [f"{v}_hat" for v in endog]

        if self.family == "regress":
            b = np.linalg.lstsq(X_aug, Y, rcond=None)[0]
            resid = Y - X_aug @ b
            k_aug = X_aug.shape[1]
            XtXi_pinv = np.linalg.pinv(X_aug.T @ X_aug)
            V = (n / (n - k_aug)) * (XtXi_pinv @ (X_aug.T @ ((resid**2)[:, None] * X_aug)) @ XtXi_pinv)
            se = np.sqrt(np.maximum(0.0, np.diag(V)))
            df_r = n - k_aug
            ctx.console.write(f"Control-function linear regression      Number of obs = {n:>8}")
            ctx.console.write("NOTE: SEs do not account for first-stage estimation "
                              "(point estimates match 2SLS)")
            _print_coef_table(ctx, depvar, aug_names, b, se, df_t=df_r, level=level)
            _store_e(ctx, "cfregress", depvar, aug_names, b, V, valid,
                     {"N": float(n), "df_r": float(df_r)})
        else:
            from scipy.special import expit

            y01 = (Y > 0).astype(float)
            beta = np.linalg.lstsq(X_aug, y01, rcond=None)[0]
            for _ in range(100):
                eta = np.clip(X_aug @ beta, -30, 30)
                p = np.clip(expit(eta), 1e-10, 1 - 1e-10)
                grad = X_aug.T @ (y01 - p)
                W = p * (1 - p)
                H = X_aug.T @ (W[:, None] * X_aug)
                step = np.linalg.solve(H + 1e-10 * np.eye(X_aug.shape[1]), grad)
                beta += step
                if np.max(np.abs(step)) < 1e-10:
                    break
            eta = np.clip(X_aug @ beta, -30, 30)
            p = np.clip(expit(eta), 1e-10, 1 - 1e-10)
            W = p * (1 - p)
            V = np.linalg.pinv(X_aug.T @ (W[:, None] * X_aug))
            se = np.sqrt(np.maximum(0.0, np.diag(V)))
            df_r = n - X_aug.shape[1]
            ctx.console.write(f"Control-function probit regression      Number of obs = {n:>8}")
            ctx.console.write("NOTE: SEs are naive; they do not account for first-stage estimation.")
            _print_coef_table(ctx, depvar, aug_names, beta, se, df_t=df_r, level=level)
            _store_e(ctx, "cfprobit", depvar, aug_names, beta, V, valid,
                     {"N": float(n), "df_r": float(df_r)})
