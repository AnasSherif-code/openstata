"""Data I/O and inspection commands: use, save, import/export delimited, input, describe, list, count."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from openstata.core.dataset import Dataset
from openstata.core.errors import StataError
from openstata.io import read_delimited, read_dta, read_excel, read_jasp, write_delimited, write_dta
from openstata.lang.executor import compute_sample_mask
from openstata.lang.parser import (
    ParsedCommand,
    expand_varlist,
    split_top_level_tokens,
    strip_quotes,
)
from openstata.plugins.base import BaseCommand, CommandSpec, ConsoleWriter, ExecutionContext


def _resolve_path(raw: str, ctx: ExecutionContext, default_ext: str = "") -> Path:
    p = Path(strip_quotes(raw))
    if not p.is_absolute():
        p = ctx.state.cwd / p
    if default_ext and p.suffix == "":
        p = p.with_suffix(default_ext)
    return p


class UseCmd(BaseCommand):
    spec = CommandSpec("use", 1, "Load a Stata .dta dataset into memory")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        if ctx.dataset.dirty and ctx.dataset.nvars > 0 and not cmd.options.get("clear"):
            raise StataError("no; dataset in memory has changed since last saved", rc=4)
        fpath = cmd.using_path or (split_top_level_tokens(cmd.main_args)[0] if cmd.main_args.strip() else "")
        if not fpath:
            raise StataError("filename required", rc=198)
        p = _resolve_path(fpath, ctx, ".dta")
        loaded = read_dta(p)
        ctx.dataset.df = loaded.df
        ctx.dataset.meta = loaded.meta
        ctx.dataset.missing_codes = loaded.missing_codes
        ctx.dataset.value_labels = loaded.value_labels
        ctx.dataset.data_label = loaded.data_label
        ctx.dataset.characteristics = loaded.characteristics
        ctx.dataset.sort_by = loaded.sort_by
        ctx.dataset.source_path = loaded.source_path
        ctx.dataset.dirty = False
        if loaded.data_label:
            ctx.console.write(f"({loaded.data_label})")


class SaveCmd(BaseCommand):
    spec = CommandSpec("save", 2, "Save current dataset to a Stata .dta file")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        raw = cmd.main_args.strip()
        if not raw:
            if not ctx.dataset.source_path:
                raise StataError("filename required", rc=198)
            p = Path(ctx.dataset.source_path)
        else:
            p = _resolve_path(split_top_level_tokens(raw)[0], ctx, ".dta")
        write_dta(ctx.dataset, p, replace=bool(cmd.options.get("replace")))
        ctx.console.write(f"file {p} saved")


class ImportCmd(BaseCommand):
    spec = CommandSpec("import", 3, "Import external files (delimited, excel, jasp)")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        toks = split_top_level_tokens(cmd.main_args)
        if not toks:
            raise StataError("import delimited/excel/jasp filename required", rc=198)
        kind = toks[0].lower()
        if kind in ("xlsx", "xls"):
            kind = "excel"
        if kind not in ("delimited", "excel", "jasp"):
            raise StataError(
                "only 'import delimited', 'import excel', and 'import jasp' are currently supported",
                rc=198,
            )
        if ctx.dataset.dirty and ctx.dataset.nvars > 0 and not cmd.options.get("clear"):
            raise StataError("no; dataset in memory has changed since last saved", rc=4)
        fpath = cmd.using_path or (toks[1] if len(toks) >= 2 else "")
        if not fpath:
            raise StataError("filename required", rc=198)
        p = _resolve_path(fpath, ctx)
        if kind == "delimited":
            delim = cmd.options.get("delimiters") or cmd.options.get("delimiter")
            delim_str = strip_quotes(str(delim)) if isinstance(delim, str) else None
            loaded = read_delimited(p, delimiter=delim_str)
        elif kind == "excel":
            sheet = cmd.options.get("sheet")
            sheet_name = strip_quotes(str(sheet)) if isinstance(sheet, str) and sheet else None
            loaded = read_excel(p, sheet=sheet_name)
        else:
            loaded = read_jasp(p)
        ctx.dataset.df = loaded.df
        ctx.dataset.meta = loaded.meta
        ctx.dataset.missing_codes = loaded.missing_codes
        ctx.dataset.value_labels = loaded.value_labels
        ctx.dataset.data_label = getattr(loaded, "data_label", "")
        ctx.dataset.source_path = str(p)
        ctx.dataset.dirty = False
        ctx.console.write(f"({ctx.dataset.nvars} vars, {ctx.dataset.nobs} obs)")


class ExportCmd(BaseCommand):
    spec = CommandSpec("export", 3, "Export dataset (e.g. export delimited)")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        toks = split_top_level_tokens(cmd.main_args)
        if not toks or toks[0].lower() != "delimited":
            raise StataError("only 'export delimited' is currently supported", rc=198)
        fpath = cmd.using_path or (toks[1] if len(toks) >= 2 else "")
        if not fpath:
            raise StataError("filename required", rc=198)
        p = _resolve_path(fpath, ctx)
        delim = cmd.options.get("delimiter")
        delim_str = strip_quotes(str(delim)) if isinstance(delim, str) else ","
        write_delimited(
            ctx.dataset,
            p,
            delimiter=delim_str,
            replace=bool(cmd.options.get("replace")),
            novarnames=bool(cmd.options.get("novarnames")),
        )
        ctx.console.write(f"file {p} saved")


def execute_input_block(header: str, rows: list[str], dataset: Dataset, console: ConsoleWriter) -> None:
    """Execute `input [types] varlist ... end` inline data entry."""
    toks = split_top_level_tokens(header)
    spec_toks = toks[1:]
    col_specs: list[tuple[str, str]] = []  # (name, stype)
    i = 0
    while i < len(spec_toks):
        t = spec_toks[i]
        tl = t.lower()
        if tl in ("byte", "int", "long", "float", "double") or (tl.startswith("str") and (tl == "strl" or tl[3:].isdigit())):
            if i + 1 >= len(spec_toks):
                raise StataError("variable name missing after type in input", rc=198)
            col_specs.append((spec_toks[i + 1], tl if tl != "strl" else "strL"))
            i += 2
        else:
            col_specs.append((t, "double"))
            i += 1

    if not col_specs:
        if dataset.nvars == 0:
            raise StataError("varlist required for input", rc=198)
        col_specs = [(c, dataset.meta[c].stata_type) for c in dataset.columns]

    data_cols: dict[str, list] = {name: [] for name, _ in col_specs}
    mc_cols: dict[str, list[int]] = {name: [] for name, _ in col_specs}

    from openstata.core.missing import MISSING_DOT, MISSING_NONE, parse_missing_literal

    for row in rows:
        rtoks = split_top_level_tokens(row)
        if len(rtoks) != len(col_specs):
            raise StataError(f"input row has {len(rtoks)} values, expected {len(col_specs)}", rc=198)
        for (name, stype), tok in zip(col_specs, rtoks):
            if stype.startswith("str"):
                data_cols[name].append(strip_quotes(tok))
            else:
                mcode = parse_missing_literal(tok)
                if mcode is not None:
                    data_cols[name].append(np.nan)
                    mc_cols[name].append(mcode)
                else:
                    data_cols[name].append(float(tok))
                    mc_cols[name].append(MISSING_NONE)

    n_new = len(rows)
    if dataset.nvars == 0:
        for name, stype in col_specs:
            if stype.startswith("str"):
                dataset.add_string_var(name, data_cols[name], overwrite=True)
            else:
                dataset.add_numeric_var(
                    name,
                    np.array(data_cols[name], dtype=np.float64),
                    np.array(mc_cols[name], dtype=np.uint8),
                    stata_type=stype,
                    overwrite=True,
                )
    else:
        for name, stype in col_specs:
            if name not in dataset.df.columns:
                raise StataError(f"variable {name} not found", rc=111)
        new_df = pd.DataFrame({n: data_cols[n] for n, _ in col_specs})
        dataset.df = pd.concat([dataset.df, new_df], ignore_index=True)
        for name, stype in col_specs:
            if not stype.startswith("str"):
                dataset.missing_codes[name] = np.concatenate(
                    [dataset.missing_codes[name], np.array(mc_cols[name], dtype=np.uint8)]
                )
        dataset.dirty = True
    console.write(f"({n_new} observations read)")


class DescribeCmd(BaseCommand):
    spec = CommandSpec("describe", 1, "Describe dataset structure and variables")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        ds = ctx.dataset
        vars_to_show = expand_varlist(cmd.main_args, ds) if cmd.main_args.strip() else ds.columns
        ctx.console.write(f"Contains data" + (f" from {ds.source_path}" if ds.source_path else ""))
        ctx.console.write(f"  Observations: {ds.nobs:>12,}")
        ctx.console.write(f"     Variables: {ds.nvars:>12,}")
        if ds.data_label:
            ctx.console.write(f"    Data label: {ds.data_label}")
        if not cmd.options.get("short"):
            ctx.console.write("-" * 76)
            ctx.console.write(f"{'Variable':<16}{'Storage':<10}{'Display':<12}{'Value':<14}{'Variable label'}")
            ctx.console.write(f"{'name':<16}{'type':<10}{'format':<12}{'label':<14}")
            ctx.console.write("-" * 76)
            for v in vars_to_show:
                m = ds.meta[v]
                ctx.console.write(f"{m.name:<16}{m.stata_type:<10}{m.fmt:<12}{m.value_label:<14}{m.label}")
            ctx.console.write("-" * 76)
        if ds.sort_by:
            ctx.console.write(f"Sorted by: {' '.join(ds.sort_by)}")
        ctx.state.r.clear()
        ctx.state.r.scalars["N"] = float(ds.nobs)
        ctx.state.r.scalars["k"] = float(ds.nvars)


class ListCmd(BaseCommand):
    spec = CommandSpec("list", 1, "List observations")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        ds = ctx.dataset
        vars_show = expand_varlist(cmd.main_args, ds) if cmd.main_args.strip() else ds.columns
        if not vars_show:
            return
        mask = compute_sample_mask(cmd, ds, ctx.state)
        indices = np.flatnonzero(mask)
        nolab = bool(cmd.options.get("nolabel"))
        widths = [max(len(v), 8) for v in vars_show]
        for j, v in enumerate(vars_show):
            for r in indices[:200]:
                widths[j] = max(widths[j], len(ds.format_value(v, int(r), nolabel=nolab)))
        header = "  " + "  ".join(f"{v:>{w}}" for v, w in zip(vars_show, widths))
        ctx.console.write(header)
        ctx.console.write("  " + "-" * (len(header) - 2))
        for r in indices:
            row_cells = [f"{ds.format_value(v, int(r), nolabel=nolab):>{w}}" for v, w in zip(vars_show, widths)]
            ctx.console.write(f"{int(r) + 1:>4}. " + "  ".join(row_cells))


class CountCmd(BaseCommand):
    spec = CommandSpec("count", 3, "Count observations satisfying condition")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        mask = compute_sample_mask(cmd, ctx.dataset, ctx.state)
        cnt = int(np.sum(mask))
        ctx.console.write(f"  {cnt}")
        ctx.state.r.clear()
        ctx.state.r.scalars["N"] = float(cnt)


class CodebookCmd(BaseCommand):
    spec = CommandSpec("codebook", 8, "Inspect variable distributions and missing counts")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        ds = ctx.dataset
        vars_show = expand_varlist(cmd.main_args, ds) if cmd.main_args.strip() else ds.columns
        for v in vars_show:
            m = ds.meta[v]
            miss = int(np.sum(ds.get_missing_mask(v)))
            ctx.console.write("-" * 72)
            ctx.console.write(f"{v}   ({m.label})" if m.label else v)
            ctx.console.write(f"  type: {m.stata_type}    missing: {miss}/{ds.nobs}")
            if not ds.is_string(v):
                vals, mc = ds.get_numeric(v)
                valid = vals[mc == 0]
                if len(valid):
                    uniq = len(np.unique(valid))
                    ctx.console.write(f"  unique values: {uniq}   range: [{np.min(valid):g}, {np.max(valid):g}]   mean: {np.mean(valid):g}")
            else:
                s = ds.df[v].astype(str)
                uniq = s[s != ""].nunique()
                ctx.console.write(f"  unique non-empty values: {uniq}")
