"""Data manipulation commands: generate, replace, egen, drop, keep, sort, merge, collapse, reshape."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from openstata.core.dataset import Dataset, ValueLabelTable
from openstata.core.errors import StataError
from openstata.core.missing import MISSING_DOT, MISSING_NONE, missing_codes_from_array
from openstata.io import read_dta
from openstata.lang.executor import compute_sample_mask
from openstata.lang.expr import eval_expression
from openstata.lang.parser import (
    ParsedCommand,
    expand_varlist,
    split_top_level_tokens,
    strip_quotes,
)
from openstata.plugins.base import BaseCommand, CommandSpec, ExecutionContext

_NUM_TYPES = {"byte", "int", "long", "float", "double"}


def _parse_assign(body: str) -> tuple[str | None, str, str]:
    if "=" not in body:
        raise StataError("= exp required", rc=198)
    lhs, rhs = body.split("=", 1)
    toks = lhs.strip().split()
    if len(toks) == 2:
        return toks[0].lower(), toks[1], rhs.strip()
    if len(toks) == 1:
        return None, toks[0], rhs.strip()
    raise StataError(f"invalid variable specification '{lhs.strip()}'", rc=198)


class GenerateCmd(BaseCommand):
    spec = CommandSpec("generate", 3, "Create a new variable")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        ds = ctx.dataset
        vtype, name, expr = _parse_assign(cmd.main_args)
        if name in ds.df.columns:
            raise StataError(f"variable {name} already defined", rc=110)

        mask = compute_sample_mask(cmd, ds, ctx.state)
        res = eval_expression(expr, ds, ctx.state)
        n = ds.nobs

        if res.is_string or (vtype and vtype.startswith("str")):
            if res.is_scalar:
                vals = [str(res.values[0]) if mask[i] else "" for i in range(n)]
            else:
                vals = [str(res.values[i]) if mask[i] else "" for i in range(n)]
            ds.add_string_var(name, vals, overwrite=False)
        else:
            if res.is_scalar:
                arr = np.full(n, float(res.values[0]), dtype=np.float64)
                mc = np.full(n, int(res.missing_codes[0]) if res.missing_codes is not None else 0, dtype=np.uint8)
            else:
                arr = np.asarray(res.values, dtype=np.float64).copy()
                mc = res.missing_codes.copy() if res.missing_codes is not None else missing_codes_from_array(arr)
            arr = np.where(mask, arr, np.nan)
            mc = np.where(mask, mc, MISSING_DOT).astype(np.uint8)
            stype = vtype if vtype in _NUM_TYPES else "float"
            ds.add_numeric_var(name, arr, mc, stata_type=stype, overwrite=False)
        n_miss = int(np.sum(ds.get_missing_mask(name)))
        if n_miss > 0:
            ctx.console.write(f"({n_miss} missing values generated)")


class ReplaceCmd(BaseCommand):
    spec = CommandSpec("replace", 3, "Replace values of an existing variable")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        ds = ctx.dataset
        _, name, expr = _parse_assign(cmd.main_args)
        ds.require_var(name)
        mask = compute_sample_mask(cmd, ds, ctx.state)
        res = eval_expression(expr, ds, ctx.state)
        n = ds.nobs

        if ds.is_string(name):
            if not res.is_string:
                raise StataError("type mismatch", rc=109)
            cur = ds.df[name].astype(str).tolist()
            new_vals = [str(res.values[0])] * n if res.is_scalar else [str(x) for x in res.values]
            changes = 0
            for i in range(n):
                if mask[i] and cur[i] != new_vals[i]:
                    cur[i] = new_vals[i]
                    changes += 1
            ds.df[name] = cur
            ds.dirty = True
            ctx.console.write(f"({changes} real changes made)")
        else:
            if res.is_string:
                raise StataError("type mismatch", rc=109)
            old_vals, old_mc = ds.get_numeric(name)
            if res.is_scalar:
                nv = np.full(n, float(res.values[0]), dtype=np.float64)
                nmc = np.full(n, int(res.missing_codes[0]) if res.missing_codes is not None else 0, dtype=np.uint8)
            else:
                nv = np.asarray(res.values, dtype=np.float64)
                nmc = res.missing_codes if res.missing_codes is not None else missing_codes_from_array(nv)
            changed_mask = mask & (
                (old_mc != nmc)
                | ((old_mc == 0) & (nmc == 0) & (old_vals != nv))
            )
            changes = int(np.sum(changed_mask))
            out_v = np.where(mask, nv, old_vals)
            out_mc = np.where(mask, nmc, old_mc).astype(np.uint8)
            out_v[out_mc > 0] = np.nan
            ds.df[name] = out_v
            ds.missing_codes[name] = out_mc
            ds.dirty = True
            ctx.console.write(f"({changes} real changes made)")


class EgenCmd(BaseCommand):
    spec = CommandSpec("egen", 4, "Extensions to generate (group & row aggregations)")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        ds = ctx.dataset
        _, newvar, rhs = _parse_assign(cmd.main_args)
        if newvar in ds.df.columns:
            raise StataError(f"variable {newvar} already defined", rc=110)
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$", rhs)
        if not m:
            raise StataError("syntax: egen newvar = fcn(arguments)", rc=198)
        fcn = m.group(1).lower()
        arg_str = m.group(2).strip()
        mask = compute_sample_mask(cmd, ds, ctx.state)
        by_opt = cmd.options.get("by")
        by_vars = expand_varlist(str(by_opt), ds) if isinstance(by_opt, str) else []
        n = ds.nobs

        # Row-wise functions
        if fcn in ("rowtotal", "rowmean", "rowmin", "rowmax", "rowmiss", "rownonmiss"):
            vlist = expand_varlist(arg_str, ds)
            mat = np.column_stack([ds.get_numeric(v)[0] for v in vlist])
            with np.errstate(all="ignore"):
                if fcn == "rowtotal":
                    vals = np.nansum(mat, axis=1)
                elif fcn == "rowmean":
                    vals = np.nanmean(mat, axis=1)
                elif fcn == "rowmin":
                    vals = np.nanmin(mat, axis=1)
                elif fcn == "rowmax":
                    vals = np.nanmax(mat, axis=1)
                elif fcn == "rowmiss":
                    vals = np.sum(np.isnan(mat), axis=1).astype(np.float64)
                else:
                    vals = np.sum(~np.isnan(mat), axis=1).astype(np.float64)
            vals = np.where(mask, vals, np.nan)
            ds.add_numeric_var(newvar, vals)
            return

        if fcn == "group":
            vlist = expand_varlist(arg_str, ds)
            sub = ds.df.loc[mask, vlist]
            codes, _ = pd.factorize(pd.MultiIndex.from_frame(sub) if len(vlist) > 1 else sub[vlist[0]], sort=True)
            out = np.full(n, np.nan)
            out_vals = np.where(codes >= 0, codes + 1.0, np.nan)
            out[mask] = out_vals
            ds.add_numeric_var(newvar, out, stata_type="long")
            return

        if fcn == "tag":
            vlist = expand_varlist(arg_str, ds)
            out = np.zeros(n, dtype=np.float64)
            sub = ds.df.loc[mask, vlist]
            first_flags = ~sub.duplicated(keep="first")
            out[mask] = first_flags.to_numpy(dtype=np.float64)
            ds.add_numeric_var(newvar, out, stata_type="byte")
            return

        # Group / whole-sample aggregation functions
        res = eval_expression(arg_str, ds, ctx.state)
        if res.is_string:
            raise StataError("type mismatch in egen", rc=109)
        src = np.where(mask, np.asarray(res.values, dtype=np.float64), np.nan)
        s = pd.Series(src, index=ds.df.index)
        agg_map = {
            "sum": "sum",
            "total": "sum",
            "mean": "mean",
            "sd": "std",
            "min": "min",
            "max": "max",
            "median": "median",
            "count": "count",
        }
        if fcn in agg_map:
            if by_vars:
                out_s = s.groupby([ds.df[c] for c in by_vars], dropna=False).transform(agg_map[fcn])
            else:
                val = getattr(s, agg_map[fcn])()
                out_s = pd.Series(np.full(n, float(val)), index=ds.df.index)
            arr = np.where(mask, out_s.to_numpy(dtype=np.float64), np.nan)
            ds.add_numeric_var(newvar, arr)
            return

        if fcn == "rank":
            if by_vars:
                out_s = s.groupby([ds.df[c] for c in by_vars], dropna=False).rank(method="average")
            else:
                out_s = s.rank(method="average")
            arr = np.where(mask, out_s.to_numpy(dtype=np.float64), np.nan)
            ds.add_numeric_var(newvar, arr)
            return

        raise StataError(f"unknown egen function {fcn}()", rc=133)


class DropCmd(BaseCommand):
    spec = CommandSpec("drop", 4, "Drop variables or observations")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        ds = ctx.dataset
        if cmd.if_expr is not None or cmd.in_raw is not None:
            if cmd.main_args.strip():
                raise StataError("cannot combine varlist with if/in in drop", rc=198)
            mask = compute_sample_mask(cmd, ds, ctx.state)
            n_drop = int(np.sum(mask))
            ds.keep_rows(~mask)
            ctx.console.write(f"({n_drop} observations deleted)")
        else:
            vars_drop = expand_varlist(cmd.main_args, ds)
            ds.drop_vars(vars_drop)


class KeepCmd(BaseCommand):
    spec = CommandSpec("keep", 4, "Keep variables or observations")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        ds = ctx.dataset
        if cmd.if_expr is not None or cmd.in_raw is not None:
            if cmd.main_args.strip():
                raise StataError("cannot combine varlist with if/in in keep", rc=198)
            mask = compute_sample_mask(cmd, ds, ctx.state)
            n_drop = int(np.sum(~mask))
            ds.keep_rows(mask)
            ctx.console.write(f"({n_drop} observations deleted)")
        else:
            vars_keep = expand_varlist(cmd.main_args, ds)
            ds.keep_vars(vars_keep)


class RenameCmd(BaseCommand):
    spec = CommandSpec("rename", 3, "Rename a variable")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        toks = split_top_level_tokens(cmd.main_args)
        if len(toks) != 2:
            raise StataError("syntax: rename oldname newname", rc=198)
        old = expand_varlist([toks[0]], ctx.dataset)[0]
        ctx.dataset.rename_var(old, toks[1])


class SortCmd(BaseCommand):
    spec = CommandSpec("sort", 2, "Sort observations ascending")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        vlist = expand_varlist(cmd.main_args, ctx.dataset)
        ctx.dataset.sort_rows(vlist, ascending=[True] * len(vlist))


class GsortCmd(BaseCommand):
    spec = CommandSpec("gsort", 5, "Sort observations ascending (+) or descending (-)")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        toks = split_top_level_tokens(cmd.main_args)
        if not toks:
            raise StataError("varlist required", rc=100)
        by_vars: list[str] = []
        asc: list[bool] = []
        for t in toks:
            if t.startswith("-"):
                by_vars.extend(expand_varlist([t[1:]], ctx.dataset))
                asc.append(False)
            elif t.startswith("+"):
                by_vars.extend(expand_varlist([t[1:]], ctx.dataset))
                asc.append(True)
            else:
                by_vars.extend(expand_varlist([t], ctx.dataset))
                asc.append(True)
        ctx.dataset.sort_rows(by_vars, ascending=asc)


class LabelCmd(BaseCommand):
    spec = CommandSpec("label", 2, "Manage dataset, variable, and value labels")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        ds = ctx.dataset
        toks = split_top_level_tokens(cmd.main_args)
        if not toks:
            raise StataError("label subcommand required", rc=198)
        sub = toks[0].lower()
        if sub == "data":
            ds.data_label = strip_quotes(toks[1]) if len(toks) > 1 else ""
            ds.dirty = True
            return
        if sub in ("variable", "var"):
            if len(toks) < 2:
                raise StataError("varname required", rc=198)
            v = expand_varlist([toks[1]], ds)[0]
            ds.meta[v].label = strip_quotes(toks[2]) if len(toks) > 2 else ""
            ds.dirty = True
            return
        if sub == "define":
            if len(toks) < 2:
                raise StataError("lblname required", rc=198)
            lblname = toks[1]
            pairs = toks[2:]
            if len(pairs) % 2 != 0:
                raise StataError("value-label pairs must alternate integer and text", rc=198)
            if lblname in ds.value_labels and not (cmd.options.get("add") or cmd.options.get("modify") or cmd.options.get("replace")):
                raise StataError(f"label {lblname} already defined", rc=110)
            tbl = (
                ValueLabelTable(lblname, {})
                if cmd.options.get("replace") or lblname not in ds.value_labels
                else ds.value_labels[lblname]
            )
            for i in range(0, len(pairs), 2):
                tbl.mapping[int(pairs[i])] = strip_quotes(pairs[i + 1])
            ds.value_labels[lblname] = tbl
            ds.dirty = True
            return
        if sub == "values":
            if len(toks) < 2:
                raise StataError("syntax: label values varlist [lblname]", rc=198)
            if len(toks) == 2:
                for v in expand_varlist([toks[1]], ds):
                    ds.meta[v].value_label = ""
            else:
                lblname = toks[-1]
                for v in expand_varlist(toks[1:-1], ds):
                    ds.meta[v].value_label = lblname
            ds.dirty = True
            return
        if sub == "list":
            names = toks[1:] if len(toks) > 1 else list(ds.value_labels.keys())
            for nm in names:
                tbl = ds.value_labels.get(nm)
                if tbl:
                    ctx.console.write(f"{nm}:")
                    for k, v in sorted(tbl.mapping.items()):
                        ctx.console.write(f"  {k:>6} {v}")
            return
        if sub == "drop":
            for nm in toks[1:]:
                ds.value_labels.pop(nm, None)
            ds.dirty = True
            return
        raise StataError(f"unknown label subcommand {sub}", rc=198)


class EncodeCmd(BaseCommand):
    spec = CommandSpec("encode", 3, "Encode string variable to labeled numeric variable")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        ds = ctx.dataset
        v = expand_varlist(cmd.main_args, ds)[0]
        if not ds.is_string(v):
            raise StataError("encode requires a string variable", rc=109)
        gen = cmd.options.get("generate") or cmd.options.get("gen")
        if not isinstance(gen, str) or not gen:
            raise StataError("option generate(newvar) required", rc=198)
        s = ds.df[v].astype(str)
        uniq = sorted({x for x in s if x != ""})
        mapping = {i + 1: val for i, val in enumerate(uniq)}
        inv = {val: float(i + 1) for i, val in enumerate(uniq)}
        arr = np.array([inv.get(x, np.nan) for x in s], dtype=np.float64)
        lblname = str(cmd.options.get("label") or gen)
        ds.value_labels[lblname] = ValueLabelTable(lblname, mapping)
        ds.add_numeric_var(gen, arr, stata_type="long", label=ds.meta[v].label)
        ds.meta[gen].value_label = lblname


class DecodeCmd(BaseCommand):
    spec = CommandSpec("decode", 3, "Decode labeled numeric variable to string variable")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        ds = ctx.dataset
        v = expand_varlist(cmd.main_args, ds)[0]
        gen = cmd.options.get("generate") or cmd.options.get("gen")
        if not isinstance(gen, str) or not gen:
            raise StataError("option generate(newvar) required", rc=198)
        tbl = ds.value_labels.get(ds.meta[v].value_label)
        if tbl is None:
            raise StataError(f"{v} has no value label", rc=182)
        vals, mc = ds.get_numeric(v)
        out = [
            tbl.mapping.get(int(val), "") if (m == 0 and float(int(val)) == val) else ""
            for val, m in zip(vals, mc)
        ]
        ds.add_string_var(gen, out, label=ds.meta[v].label)


class DestringCmd(BaseCommand):
    spec = CommandSpec("destring", 8, "Convert string variables to numeric")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        ds = ctx.dataset
        vlist = expand_varlist(cmd.main_args, ds) if cmd.main_args.strip() else ds.columns
        replace = bool(cmd.options.get("replace"))
        force = bool(cmd.options.get("force"))
        gen_opt = cmd.options.get("generate") or cmd.options.get("gen")
        gen_names = str(gen_opt).split() if isinstance(gen_opt, str) else []
        if not replace and len(gen_names) != len(vlist):
            raise StataError("must specify replace or generate(newvarlist)", rc=198)
        for idx, v in enumerate(vlist):
            if not ds.is_string(v):
                continue
            s = ds.df[v].astype(str).str.strip()
            num = pd.to_numeric(s.replace("", np.nan), errors="coerce")
            bad = (s != "") & num.isna()
            if bad.any() and not force:
                raise StataError(f"{v} contains nonnumeric characters; no generate", rc=109)
            target = v if replace else gen_names[idx]
            ds.add_numeric_var(target, num.to_numpy(dtype=np.float64), overwrite=replace)


class TostringCmd(BaseCommand):
    spec = CommandSpec("tostring", 8, "Convert numeric variables to string")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        ds = ctx.dataset
        vlist = expand_varlist(cmd.main_args, ds) if cmd.main_args.strip() else ds.columns
        replace = bool(cmd.options.get("replace"))
        gen_opt = cmd.options.get("generate") or cmd.options.get("gen")
        gen_names = str(gen_opt).split() if isinstance(gen_opt, str) else []
        if not replace and len(gen_names) != len(vlist):
            raise StataError("must specify replace or generate(newvarlist)", rc=198)
        for idx, v in enumerate(vlist):
            if ds.is_string(v):
                continue
            vals, mc = ds.get_numeric(v)
            out = [
                "" if m > 0 else (str(int(x)) if float(int(x)) == x else f"{x:g}")
                for x, m in zip(vals, mc)
            ]
            target = v if replace else gen_names[idx]
            ds.add_string_var(target, out, overwrite=replace)


class DuplicatesCmd(BaseCommand):
    spec = CommandSpec("duplicates", 10, "Report, tag, or drop duplicate observations")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        ds = ctx.dataset
        toks = split_top_level_tokens(cmd.main_args)
        if not toks:
            raise StataError("subcommand required (report, list, tag, drop)", rc=198)
        sub = toks[0].lower()
        vlist = expand_varlist(toks[1:], ds) if len(toks) > 1 else ds.columns
        if sub == "report":
            dups = ds.df.duplicated(subset=vlist, keep=False)
            surplus = ds.df.duplicated(subset=vlist, keep="first")
            ctx.console.write(f"Total observations: {ds.nobs}")
            ctx.console.write(f"Surplus (excess) copies: {int(surplus.sum())}")
            ctx.state.r.scalars["N"] = float(ds.nobs)
            ctx.state.r.scalars["unique_value"] = float(ds.nobs - int(surplus.sum()))
            return
        if sub == "drop":
            if not cmd.options.get("force") and vlist == ds.columns:
                pass
            keep_mask = ~ds.df.duplicated(subset=vlist, keep="first").to_numpy()
            n_del = int(np.sum(~keep_mask))
            ds.keep_rows(keep_mask)
            ctx.console.write(f"({n_del} observations deleted)")
            return
        if sub == "tag":
            gen = cmd.options.get("generate") or cmd.options.get("gen")
            if not isinstance(gen, str):
                raise StataError("generate(newvar) required for duplicates tag", rc=198)
            counts = ds.df.groupby(vlist, dropna=False)[vlist[0]].transform("count") - 1
            ds.add_numeric_var(gen, counts.to_numpy(dtype=np.float64), stata_type="long")
            return
        raise StataError(f"unknown duplicates subcommand {sub}", rc=198)


class MergeCmd(BaseCommand):
    spec = CommandSpec("merge", 3, "Merge dataset in memory with another .dta dataset")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        toks = split_top_level_tokens(cmd.main_args)
        if len(toks) < 2:
            raise StataError("syntax: merge 1:1|m:1|1:m|m:m keyvars using filename", rc=198)
        mtype = toks[0].lower()
        if mtype not in ("1:1", "m:1", "1:m", "m:m"):
            raise StataError(f"invalid merge type {mtype}", rc=198)
        if not cmd.using_path:
            raise StataError("using filename required", rc=198)
        p = Path(cmd.using_path)
        if not p.is_absolute():
            p = ctx.state.cwd / p
        if p.suffix == "":
            p = p.with_suffix(".dta")

        using_ds = read_dta(p)
        key_vars = expand_varlist(toks[1:], ctx.dataset)
        for kv in key_vars:
            using_ds.require_var(kv)

        validate_map = {"1:1": "one_to_one", "m:1": "many_to_one", "1:m": "one_to_many", "m:m": "many_to_many"}
        try:
            merged = pd.merge(
                ctx.dataset.df,
                using_ds.df,
                on=key_vars,
                how="outer",
                indicator="_merge_ind",
                suffixes=("", "_using_dup"),
                validate=validate_map[mtype],
            )
        except pd.errors.MergeError as exc:
            raise StataError(f"merge key not unique: {exc}", rc=459)

        # Resolve overlapping non-key variables: master takes precedence by default
        for col in list(merged.columns):
            if col.endswith("_using_dup"):
                base = col[: -len("_using_dup")]
                if cmd.options.get("replace") or cmd.options.get("update"):
                    mask_upd = merged[base].isna() if not cmd.options.get("replace") else merged[col].notna()
                    merged[base] = np.where(mask_upd, merged[col], merged[base])
                merged = merged.drop(columns=[col])

        ind_map = {"left_only": 1.0, "right_only": 2.0, "both": 3.0}
        merge_codes = merged["_merge_ind"].astype(str).map(ind_map).to_numpy(dtype=np.float64)
        merged = merged.drop(columns=["_merge_ind"])

        # Build merged Dataset preserving metadata
        new_meta = dict(using_ds.meta)
        new_meta.update(ctx.dataset.meta)
        new_vlabels = dict(using_ds.value_labels)
        new_vlabels.update(ctx.dataset.value_labels)

        ctx.dataset.df = merged
        ctx.dataset.meta = new_meta
        ctx.dataset.value_labels = new_vlabels
        ctx.dataset.missing_codes = {}
        ctx.dataset._sync_metadata()

        mvar = str(cmd.options.get("gen") or cmd.options.get("generate") or "_merge")
        if not cmd.options.get("nogenerate") and not cmd.options.get("nogen"):
            ctx.dataset.add_numeric_var(mvar, merge_codes, stata_type="byte", overwrite=True)

        n1 = int(np.sum(merge_codes == 1))
        n2 = int(np.sum(merge_codes == 2))
        n3 = int(np.sum(merge_codes == 3))
        ctx.console.write(f"  Result                           Number of obs")
        ctx.console.write(f"  ----------------------------------------------")
        ctx.console.write(f"  Not matched                       {n1 + n2:>12}")
        ctx.console.write(f"      from master                   {n1:>12}  (_merge==1)")
        ctx.console.write(f"      from using                    {n2:>12}  (_merge==2)")
        ctx.console.write(f"  Matched                           {n3:>12}  (_merge==3)")
        ctx.console.write(f"  ----------------------------------------------")

        keep_opt = cmd.options.get("keep")
        if isinstance(keep_opt, str):
            allowed = set()
            for tok in keep_opt.lower().split():
                if tok in ("1", "master"):
                    allowed.add(1.0)
                elif tok in ("2", "using"):
                    allowed.add(2.0)
                elif tok in ("3", "match", "matched"):
                    allowed.add(3.0)
            ctx.dataset.keep_rows(np.isin(merge_codes, list(allowed)))
        ctx.dataset.dirty = True


class AppendCmd(BaseCommand):
    spec = CommandSpec("append", 2, "Append observations from another .dta dataset")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        fpath = cmd.using_path or cmd.main_args.strip()
        if not fpath:
            raise StataError("using filename required", rc=198)
        p = Path(strip_quotes(fpath))
        if not p.is_absolute():
            p = ctx.state.cwd / p
        if p.suffix == "":
            p = p.with_suffix(".dta")
        other = read_dta(p)
        ctx.dataset.df = pd.concat([ctx.dataset.df, other.df], ignore_index=True)
        for k, v in other.meta.items():
            ctx.dataset.meta.setdefault(k, v)
        for k, v in other.value_labels.items():
            ctx.dataset.value_labels.setdefault(k, v)
        ctx.dataset.missing_codes = {}
        ctx.dataset._sync_metadata()
        ctx.dataset.dirty = True


class CollapseCmd(BaseCommand):
    spec = CommandSpec("collapse", 4, "Collapse dataset into summary statistics by group")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        ds = ctx.dataset
        mask = compute_sample_mask(cmd, ds, ctx.state)
        by_opt = cmd.options.get("by")
        by_vars = expand_varlist(str(by_opt), ds) if isinstance(by_opt, str) else []

        # Parse clist like `(mean) x y (sum) s_z=z`
        toks = split_top_level_tokens(cmd.main_args)
        stat = "mean"
        specs: list[tuple[str, str, str]] = []  # (stat, target_name, src_var)
        for t in toks:
            if t.startswith("(") and t.endswith(")"):
                stat = t[1:-1].strip().lower()
                continue
            if "=" in t:
                tgt, src = t.split("=", 1)
                specs.append((stat, tgt.strip(), expand_varlist([src.strip()], ds)[0]))
            else:
                for v in expand_varlist([t], ds):
                    specs.append((stat, v, v))

        if not specs:
            raise StataError("clist required for collapse", rc=198)

        sub = ds.df.loc[mask].copy()
        stat_map = {
            "mean": "mean",
            "sum": "sum",
            "sd": "std",
            "min": "min",
            "max": "max",
            "median": "median",
            "count": "count",
            "first": "first",
            "last": "last",
        }

        if by_vars:
            grouped = sub.groupby(by_vars, as_index=False, dropna=False)
            out_df = grouped[by_vars].first()
            for st, tgt, src in specs:
                if st not in stat_map:
                    raise StataError(f"unsupported collapse stat ({st})", rc=198)
                agg_s = grouped[src].agg(stat_map[st])
                out_df[tgt] = agg_s[src].to_numpy()
        else:
            row: dict[str, list] = {}
            for st, tgt, src in specs:
                if st not in stat_map:
                    raise StataError(f"unsupported collapse stat ({st})", rc=198)
                val = getattr(sub[src], stat_map[st])()
                row[tgt] = [val]
            out_df = pd.DataFrame(row)

        ctx.dataset.df = out_df
        ctx.dataset.meta = {}
        ctx.dataset.missing_codes = {}
        ctx.dataset.sort_by = list(by_vars)
        ctx.dataset._sync_metadata()
        ctx.dataset.dirty = True


class ReshapeCmd(BaseCommand):
    spec = CommandSpec("reshape", 7, "Reshape dataset between wide and long layouts")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        ds = ctx.dataset
        toks = split_top_level_tokens(cmd.main_args)
        if len(toks) < 2:
            raise StataError("syntax: reshape wide|long stublist, i(varlist) j(var [string])", rc=198)
        direction = toks[0].lower()
        stubs = toks[1:]
        i_opt = cmd.options.get("i")
        j_opt = cmd.options.get("j")
        if not isinstance(i_opt, str) or not isinstance(j_opt, str):
            raise StataError("options i() and j() are required", rc=198)
        i_vars = expand_varlist(i_opt, ds)
        j_parts = j_opt.split()
        j_var = j_parts[0]
        j_is_string = len(j_parts) > 1 and j_parts[1].lower() == "string"

        if direction == "long":
            # Identify columns matching `<stub><suffix>`
            records = []
            for idx, stub in enumerate(stubs):
                matched = [c for c in ds.columns if c.startswith(stub) and c != stub and c not in i_vars]
                if not matched:
                    raise StataError(f"no variables match stub {stub}", rc=111)
                melted = ds.df[i_vars + matched].melt(
                    id_vars=i_vars,
                    value_vars=matched,
                    var_name=j_var,
                    value_name=stub,
                )
                melted[j_var] = melted[j_var].str[len(stub) :]
                if not j_is_string:
                    melted[j_var] = pd.to_numeric(melted[j_var], errors="coerce")
                records.append(melted.set_index(i_vars + [j_var]))
            out_df = pd.concat(records, axis=1).reset_index()
        elif direction == "wide":
            ds.require_var(j_var)
            piv = ds.df.pivot(index=i_vars, columns=j_var, values=stubs)
            def _fmt_j(val):
                if isinstance(val, (int, float, np.floating, np.integer)) and float(int(val)) == float(val):
                    return str(int(val))
                return str(val)
            piv.columns = [f"{col[0]}{_fmt_j(col[1])}" for col in piv.columns]
            out_df = piv.reset_index()
        else:
            raise StataError("reshape direction must be 'wide' or 'long'", rc=198)

        ctx.dataset.df = out_df
        ctx.dataset.meta = {}
        ctx.dataset.missing_codes = {}
        ctx.dataset._sync_metadata()
        ctx.dataset.dirty = True
