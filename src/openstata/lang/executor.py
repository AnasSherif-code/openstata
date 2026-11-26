"""OpenStata script and interactive command execution runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np

from openstata.core.dataset import Dataset
from openstata.core.errors import BreakSignal, ContinueSignal, ExitSignal, StataError
from openstata.core.state import StateStore
from openstata.lang.expr import eval_expression
from openstata.lang.macros import expand_macros
from openstata.lang.parser import (
    ParsedCommand,
    parse_command_line,
    parse_in_range,
    split_top_level_tokens,
    strip_quotes,
)
from openstata.lang.preprocessor import (
    BlockStmt,
    CommandStmt,
    InputStmt,
    ProgramDefStmt,
    Stmt,
    parse_script,
)
from openstata.plugins.base import ConsoleWriter, DefaultConsole, ExecutionContext
from openstata.plugins.registry import CommandRegistry


@dataclass
class AdoProgram:
    """In-memory representation of a `program define` block."""

    name: str
    body: list[Stmt]


class ScriptRunner:
    """Executes Stata AST statements, .do files, .ado programs, and REPL lines."""

    def __init__(
        self,
        dataset: Dataset | None = None,
        state: StateStore | None = None,
        registry: CommandRegistry | None = None,
        console: ConsoleWriter | None = None,
    ) -> None:
        self.dataset = dataset if dataset is not None else Dataset()
        self.state = state if state is not None else StateStore()
        self.registry = registry if registry is not None else CommandRegistry()
        self.registry.load_entry_points()
        self.console = console if console is not None else DefaultConsole(self.state)

    def _inline_eval(self, expr: str, state: StateStore, dataset: Dataset | None) -> str:
        res = eval_expression(expr, dataset, state, force_scalar=True)
        v = res.to_scalar()
        if isinstance(v, str):
            return v
        if np.isnan(v):
            return "."
        return str(int(v)) if float(int(v)) == v and abs(v) < 1e12 else f"{v:g}"

    def expand(self, text: str) -> str:
        return expand_macros(text, self.state, self.dataset, eval_fn=self._inline_eval)

    def run_source(self, source: str) -> None:
        stmts = parse_script(source)
        self.run_stmts(stmts)

    def run_file(self, path: str | Path, args: list[str] | None = None) -> None:
        p = Path(path)
        if not p.is_absolute():
            p = self.state.cwd / p
        if not p.exists():
            raise StataError(f"file {p} not found", rc=601)
        source = p.read_text(encoding="utf-8")
        self.state.push_local_scope()
        try:
            if args:
                for i, a in enumerate(args, start=1):
                    self.state.locals[str(i)] = a
                self.state.locals["0"] = " ".join(args)
            self.run_source(source)
        except ExitSignal:
            return
        finally:
            self.state.pop_local_scope(self.dataset)

    def run_stmts(self, stmts: list[Stmt], silent: bool = False) -> None:
        for stmt in stmts:
            if isinstance(stmt, CommandStmt):
                self._exec_command_stmt(stmt, silent=silent)
            elif isinstance(stmt, BlockStmt):
                self._exec_block_stmt(stmt, silent=silent)
            elif isinstance(stmt, ProgramDefStmt):
                self.state.programs[stmt.name] = AdoProgram(name=stmt.name, body=stmt.body)
            elif isinstance(stmt, InputStmt):
                self._exec_input_stmt(stmt)

    def _exec_input_stmt(self, stmt: InputStmt) -> None:
        from openstata.commands.io_cmds import execute_input_block

        expanded_header = self.expand(stmt.header)
        expanded_rows = [self.expand(r) for r in stmt.rows]
        execute_input_block(expanded_header, expanded_rows, self.dataset, self.console)

    def _exec_block_stmt(self, stmt: BlockStmt, silent: bool = False) -> None:
        header = self.expand(stmt.header).strip()
        low = header.lower()

        # Prefix blocks: quietly { ... }, capture { ... }, noisily { ... }
        if low in ("", "{") or low in ("qui", "quietly"):
            self.run_stmts(stmt.body, silent=True if low.startswith("qui") else silent)
            return
        if low in ("noi", "noisily"):
            self.run_stmts(stmt.body, silent=False)
            return
        if low in ("cap", "capture"):
            try:
                self.run_stmts(stmt.body, silent=True)
                self.state.r.scalars["_rc"] = 0.0
            except StataError as err:
                self.state.r.scalars["_rc"] = float(err.rc)
            return

        # `if cond { ... } else { ... }`
        if low.startswith("if "):
            cond_expr = header[3:].strip()
            res = eval_expression(cond_expr, self.dataset, self.state, force_scalar=True)
            truth = bool(res.as_truth_mask(1)[0])
            if truth:
                self.run_stmts(stmt.body, silent=silent)
            elif stmt.else_body is not None:
                self.run_stmts(stmt.else_body, silent=silent)
            return

        # `while cond { ... }`
        if low.startswith("while "):
            cond_expr = header[6:].strip()
            while True:
                res = eval_expression(cond_expr, self.dataset, self.state, force_scalar=True)
                if not bool(res.as_truth_mask(1)[0]):
                    break
                try:
                    self.run_stmts(stmt.body, silent=silent)
                except ContinueSignal:
                    continue
                except BreakSignal:
                    break
            return

        # `forvalues lcl = range { ... }`
        m_fval = re.match(r"^(?:forvalues|forval|forv)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$", header, flags=re.IGNORECASE)
        if m_fval:
            lname = m_fval.group(1)
            rspec = m_fval.group(2).strip()
            for val in _expand_numlist(rspec):
                self.state.locals[lname] = str(int(val)) if float(int(val)) == val else str(val)
                try:
                    self.run_stmts(stmt.body, silent=silent)
                except ContinueSignal:
                    continue
                except BreakSignal:
                    break
            return

        # `foreach lcl in|of ... { ... }`
        m_fe = re.match(r"^(?:foreach|foreac)\s+([A-Za-z_][A-Za-z0-9_]*)\s+(in|of)\s+(.+)$", header, flags=re.IGNORECASE)
        if m_fe:
            lname = m_fe.group(1)
            kind = m_fe.group(2).lower()
            spec = m_fe.group(3).strip()
            items = _expand_foreach_items(kind, spec, self.dataset, self.state)
            for item in items:
                self.state.locals[lname] = item
                try:
                    self.run_stmts(stmt.body, silent=silent)
                except ContinueSignal:
                    continue
                except BreakSignal:
                    break
            return

        raise StataError(f"unrecognized block header: {header}", rc=198)

    def _exec_command_stmt(self, stmt: CommandStmt, silent: bool = False) -> None:
        raw = stmt.text.strip()
        if not raw:
            return

        # Avoid premature macro expansion inside `local` / `global` macro name?
        # In Stata, macros on the line ARE expanded before execution.
        expanded = self.expand(raw)
        if not expanded.strip():
            return

        # Handle single-line `if cond cmd` (rare; usually block, but supported)
        parsed = parse_command_line(expanded)
        if not parsed.command:
            return

        cmd_low = parsed.command.lower()
        if cmd_low == "continue":
            if parsed.options.get("break"):
                raise BreakSignal()
            raise ContinueSignal()

        effective_silent = (silent or parsed.quietly) and not parsed.noisily
        ctx = ExecutionContext(
            dataset=self.dataset,
            state=self.state,
            console=DefaultConsole(self.state, silent=effective_silent),
            runner=self,
        )

        try:
            self._dispatch(parsed, ctx)
            self.state.r.scalars["_rc"] = 0.0
        except StataError as err:
            if parsed.capture:
                self.state.r.scalars["_rc"] = float(err.rc)
                return
            raise

    def _dispatch(self, parsed: ParsedCommand, ctx: ExecutionContext) -> None:
        cmd_impl = self.registry.resolve(parsed.command)
        if cmd_impl is not None:
            if parsed.by_vars:
                self._exec_with_by(parsed, cmd_impl, ctx)
            else:
                cmd_impl.run(parsed, ctx)
            return

        # Fallback: look for program in memory or .ado on adopath
        prog = self._resolve_program(parsed.command)
        if prog is not None:
            self._run_program(prog, parsed, ctx)
            return

        raise StataError(f"command {parsed.command} is unrecognized", rc=199)

    def _exec_with_by(self, parsed: ParsedCommand, cmd_impl: Any, ctx: ExecutionContext) -> None:
        ds = ctx.dataset
        if parsed.by_sort:
            ds.sort_rows(parsed.by_vars + parsed.by_sort_extra)
        elif ds.sort_by[: len(parsed.by_vars)] != parsed.by_vars:
            raise StataError("not sorted", rc=5)

        if ds.nobs == 0:
            return

        # Partition contiguous groups
        group_cols = parsed.by_vars
        df_keys = ds.df[group_cols]
        # Build group boundaries
        boundaries = [0]
        for i in range(1, ds.nobs):
            if any(df_keys.iat[i, j] != df_keys.iat[i - 1, j] for j in range(len(group_cols))):
                boundaries.append(i)
        boundaries.append(ds.nobs)

        # Save original dataset, execute per slice, or pass slice
        orig_ds = ds
        updated_parts = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            hdr = ", ".join(f"{col} = {orig_ds.format_value(col, start)}" for col in group_cols)
            ctx.console.write(f"\n-> {hdr}")
            sub_df = orig_ds.df.iloc[start:end].reset_index(drop=True)
            sub_mc = {k: v[start:end].copy() for k, v in orig_ds.missing_codes.items()}
            sub_ds = Dataset(
                df=sub_df,
                meta={k: v for k, v in orig_ds.meta.items()},
                missing_codes=sub_mc,
                value_labels=orig_ds.value_labels,
                data_label=orig_ds.data_label,
                sort_by=list(orig_ds.sort_by),
            )
            sub_ctx = ExecutionContext(dataset=sub_ds, state=ctx.state, console=ctx.console, runner=self)
            sub_parsed = ParsedCommand(**vars(parsed))
            sub_parsed.by_vars = []
            cmd_impl.run(sub_parsed, sub_ctx)
            updated_parts.append(sub_ctx.dataset)

        # Stitch modified columns back if row counts unchanged or changed (e.g. drop/keep)
        if all(p.nobs == (b2 - b1) for p, b1, b2 in zip(updated_parts, boundaries[:-1], boundaries[1:])) and all(
            p.columns == updated_parts[0].columns for p in updated_parts
        ):
            import pandas as pd

            orig_ds.df = pd.concat([p.df for p in updated_parts], ignore_index=True)
            for col in orig_ds.df.columns:
                if col in updated_parts[0].meta:
                    orig_ds.meta[col] = updated_parts[0].meta[col]
                if not orig_ds.meta[col].is_string():
                    orig_ds.missing_codes[col] = np.concatenate([p.missing_codes[col] for p in updated_parts])

    def _resolve_program(self, name: str) -> AdoProgram | None:
        if name in self.state.programs:
            return self.state.programs[name]
        for base in self.state.adopath:
            base_path = base if base.is_absolute() else self.state.cwd / base
            candidate = base_path / f"{name}.ado"
            if candidate.exists():
                src = candidate.read_text(encoding="utf-8")
                stmts = parse_script(src)
                for s in stmts:
                    if isinstance(s, ProgramDefStmt):
                        self.state.programs[s.name] = AdoProgram(name=s.name, body=s.body)
                if name in self.state.programs:
                    return self.state.programs[name]
        return None

    def _run_program(self, prog: AdoProgram, parsed: ParsedCommand, ctx: ExecutionContext) -> None:
        self.state.push_local_scope()
        try:
            raw_arg = parsed.body.strip()
            if parsed.options_str:
                raw_arg = f"{raw_arg}, {parsed.options_str}" if raw_arg else f", {parsed.options_str}"
            self.state.locals["0"] = raw_arg
            toks = split_top_level_tokens(parsed.body.strip())
            for idx, tok in enumerate(toks, start=1):
                self.state.locals[str(idx)] = strip_quotes(tok)
            self.run_stmts(prog.body, silent=parsed.quietly and not parsed.noisily)
        except ExitSignal:
            return
        finally:
            self.state.pop_local_scope(self.dataset)


def compute_sample_mask(parsed: ParsedCommand, dataset: Dataset, state: StateStore) -> np.ndarray:
    """Compute boolean mask combining `if` expression and `in` range."""
    n = dataset.nobs
    mask = np.ones(n, dtype=bool)
    if parsed.in_raw is not None:
        start, end = parse_in_range(parsed.in_raw, n)
        in_mask = np.zeros(n, dtype=bool)
        in_mask[start - 1 : end] = True
        mask &= in_mask
    if parsed.if_expr is not None:
        res = eval_expression(parsed.if_expr, dataset, state, force_scalar=False)
        mask &= res.as_truth_mask(n)
    return mask


def _expand_numlist(spec: str) -> list[float]:
    """Expand `1/5`, `0(2)10`, `10(-1)1`, `1 2 to 5`."""
    s = spec.strip()
    m = re.match(r"^([+-]?\d+(?:\.\d+)?)\s*/\s*([+-]?\d+(?:\.\d+)?)$", s)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        step = 1.0 if b >= a else -1.0
        vals = []
        cur = a
        while (step > 0 and cur <= b + 1e-12) or (step < 0 and cur >= b - 1e-12):
            vals.append(round(cur, 10))
            cur += step
        return vals
    m2 = re.match(
        r"^([+-]?\d+(?:\.\d+)?)\s*\(\s*([+-]?\d+(?:\.\d+)?)\s*\)\s*([+-]?\d+(?:\.\d+)?)$",
        s,
    )
    if m2:
        a, step, b = float(m2.group(1)), float(m2.group(2)), float(m2.group(3))
        if step == 0:
            raise StataError("step cannot be 0 in numlist", rc=198)
        vals = []
        cur = a
        while (step > 0 and cur <= b + 1e-12) or (step < 0 and cur >= b - 1e-12):
            vals.append(round(cur, 10))
            cur += step
        return vals
    # Space-separated numbers
    return [float(tok) for tok in s.split()]


def _expand_foreach_items(kind: str, spec: str, dataset: Dataset, state: StateStore) -> list[str]:
    if kind == "in":
        return [strip_quotes(t) for t in split_top_level_tokens(spec)]
    # `of <listtype> <list>`
    parts = spec.split(None, 1)
    ltype = parts[0].lower()
    rest = parts[1].strip() if len(parts) > 1 else ""
    if ltype in ("var", "vars", "varlist"):
        from openstata.lang.parser import expand_varlist

        return expand_varlist(rest, dataset)
    if ltype in ("loc", "local"):
        return [strip_quotes(t) for t in split_top_level_tokens(state.locals.get(rest, ""))]
    if ltype in ("glo", "global"):
        return [strip_quotes(t) for t in split_top_level_tokens(state.globals.get(rest, ""))]
    if ltype in ("num", "numlist"):
        return [str(int(v)) if float(int(v)) == v else str(v) for v in _expand_numlist(rest)]
    if ltype in ("new", "newlist"):
        return split_top_level_tokens(rest)
    raise StataError(f"unsupported foreach list type '{ltype}'", rc=198)
