"""System, scripting, macro, ado-programming, and package commands."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys

import numpy as np

from openstata.core.errors import ExitSignal, StataError
from openstata.lang.expr import eval_expression
from openstata.lang.parser import (
    ParsedCommand,
    expand_varlist,
    parse_command_line,
    split_top_level_tokens,
    strip_quotes,
)
from openstata.plugins.base import BaseCommand, CommandSpec, ExecutionContext
from openstata.plugins.registry import install_package


class ClearCmd(BaseCommand):
    spec = CommandSpec("clear", 5, "Clear dataset or results from memory")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        arg = cmd.main_args.strip().lower()
        if arg in ("", "data"):
            ctx.dataset.clear()
        elif arg == "all":
            ctx.dataset.clear()
            ctx.state.scalars.clear()
            ctx.state.matrices.clear()
            ctx.state.r.clear()
            ctx.state.e.clear()
            ctx.state.programs.clear()
        elif arg == "results":
            ctx.state.r.clear()
            ctx.state.e.clear()
        elif arg == "programs":
            ctx.state.programs.clear()
        else:
            raise StataError(f"clear {arg} not supported", rc=198)


class VersionCmd(BaseCommand):
    spec = CommandSpec("version", 4, "Set/check version compatibility")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        return


class SetCmd(BaseCommand):
    spec = CommandSpec("set", 2, "Set system parameters or observation count")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        toks = split_top_level_tokens(cmd.main_args)
        if len(toks) < 2:
            raise StataError("syntax: set <option> <value>", rc=198)
        key, val = toks[0].lower(), toks[1]
        if key == "obs":
            target = int(val)
            if target < ctx.dataset.nobs:
                raise StataError("cannot reduce obs with set obs; use drop/keep", rc=198)
            add_n = target - ctx.dataset.nobs
            if add_n > 0:
                if ctx.dataset.nvars == 0:
                    import pandas as pd

                    ctx.dataset.df = pd.DataFrame(index=range(target))
                else:
                    import pandas as pd

                    extra = {c: ([""] * add_n if ctx.dataset.is_string(c) else [np.nan] * add_n) for c in ctx.dataset.columns}
                    ctx.dataset.df = pd.concat([ctx.dataset.df, pd.DataFrame(extra)], ignore_index=True)
                    for c in ctx.dataset.columns:
                        if not ctx.dataset.is_string(c):
                            ctx.dataset.missing_codes[c] = np.concatenate(
                                [ctx.dataset.missing_codes[c], np.ones(add_n, dtype=np.uint8)]
                            )
                ctx.dataset.dirty = True
            return
        if key == "seed":
            seed = int(val)
            ctx.state.settings["seed"] = seed
            np.random.seed(seed)
            return
        if key in ("linesize", "maxiter"):
            ctx.state.settings[key] = int(val)
            return
        ctx.state.settings[key] = val.lower()


class CdCmd(BaseCommand):
    spec = CommandSpec("cd", 2, "Change working directory")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        target = strip_quotes(cmd.main_args.strip()) if cmd.main_args.strip() else str(Path.home())
        p = Path(target)
        if not p.is_absolute():
            p = ctx.state.cwd / p
        p = p.resolve()
        if not p.is_dir():
            raise StataError(f"directory {p} not found", rc=601)
        ctx.state.cwd = p
        ctx.console.write(str(p))


class PwdCmd(BaseCommand):
    spec = CommandSpec("pwd", 3, "Display current working directory")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        ctx.console.write(str(ctx.state.cwd))


class DisplayCmd(BaseCommand):
    spec = CommandSpec("display", 2, "Display strings and scalar expressions")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        toks = split_top_level_tokens(cmd.body)
        parts: list[str] = []
        i = 0
        while i < len(toks):
            t = toks[i]
            tl = t.lower()
            if tl == "as" and i + 1 < len(toks):
                i += 2
                continue
            if tl in ("_newline", "_n"):
                parts.append("\n")
                i += 1
                continue
            if t.startswith('"') or t.startswith('`"'):
                parts.append(strip_quotes(t))
                i += 1
                continue
            res = eval_expression(t, ctx.dataset, ctx.state, force_scalar=True)
            v = res.to_scalar()
            if isinstance(v, str):
                parts.append(v)
            elif np.isnan(v):
                parts.append(".")
            else:
                parts.append(str(int(v)) if float(int(v)) == v and abs(v) < 1e12 else f"{v:g}")
            i += 1
        ctx.console.write("".join(parts))


class LocalCmd(BaseCommand):
    spec = CommandSpec("local", 3, "Define or modify a local macro")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        body = cmd.body.strip()
        if not body:
            for k, v in ctx.state.locals.items():
                ctx.console.write(f"{k}: {v}")
            return
        m = re.match(r"^([A-Za-z_0-9]+)\s*=\s*(.+)$", body)
        if m:
            name, expr = m.group(1), m.group(2).strip()
            res = eval_expression(expr, ctx.dataset, ctx.state, force_scalar=True)
            v = res.to_scalar()
            ctx.state.locals[name] = "." if (isinstance(v, float) and np.isnan(v)) else (str(int(v)) if isinstance(v, float) and float(int(v)) == v else str(v))
            return
        m_colon = re.match(r"^([A-Za-z_0-9]+)\s*:\s*(.+)$", body)
        if m_colon:
            name, ext = m_colon.group(1), m_colon.group(2).strip()
            if ext.lower().startswith("word count "):
                s = strip_quotes(ext[11:].strip())
                ctx.state.locals[name] = str(len(s.split()))
                return
            if ext.lower().startswith("list "):
                ctx.state.locals[name] = ext[5:].strip()
                return
            ctx.state.locals[name] = strip_quotes(ext)
            return
        parts = body.split(None, 1)
        name = parts[0]
        val = strip_quotes(parts[1].strip()) if len(parts) > 1 else ""
        ctx.state.locals[name] = val


class GlobalCmd(BaseCommand):
    spec = CommandSpec("global", 2, "Define or modify a global macro")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        body = cmd.body.strip()
        if not body:
            for k, v in ctx.state.globals.items():
                ctx.console.write(f"{k}: {v}")
            return
        m = re.match(r"^([A-Za-z_0-9]+)\s*=\s*(.+)$", body)
        if m:
            name, expr = m.group(1), m.group(2).strip()
            res = eval_expression(expr, ctx.dataset, ctx.state, force_scalar=True)
            v = res.to_scalar()
            ctx.state.globals[name] = "." if (isinstance(v, float) and np.isnan(v)) else str(v)
            return
        parts = body.split(None, 1)
        name = parts[0]
        val = strip_quotes(parts[1].strip()) if len(parts) > 1 else ""
        ctx.state.globals[name] = val


class ScalarCmd(BaseCommand):
    spec = CommandSpec("scalar", 3, "Define or list scalars")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        body = cmd.body.strip()
        if body.lower() in ("", "list", "dir"):
            for k, v in ctx.state.scalars.items():
                ctx.console.write(f"  {k} = {v:g}")
            return
        if body.lower().startswith("drop "):
            for n in body[5:].split():
                ctx.state.scalars.pop(n, None)
            return
        m = re.match(r"^(?:define\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$", body, flags=re.IGNORECASE)
        if not m:
            raise StataError("syntax: scalar name = exp", rc=198)
        res = eval_expression(m.group(2), ctx.dataset, ctx.state, force_scalar=True)
        v = res.to_scalar()
        if isinstance(v, str):
            raise StataError("type mismatch in scalar", rc=109)
        ctx.state.scalars[m.group(1)] = float(v)


class TempvarCmd(BaseCommand):
    spec = CommandSpec("tempvar", 7, "Allocate temporary variable names")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        for name in cmd.main_args.split():
            ctx.state.locals[name] = ctx.state.new_tempvar()


class TempnameCmd(BaseCommand):
    spec = CommandSpec("tempname", 8, "Allocate temporary scalar/matrix names")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        for name in cmd.main_args.split():
            ctx.state.locals[name] = ctx.state.new_tempname()


class TempfileCmd(BaseCommand):
    spec = CommandSpec("tempfile", 8, "Allocate temporary file paths")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        for name in cmd.main_args.split():
            ctx.state.locals[name] = ctx.state.new_tempfile()


class ReturnCmd(BaseCommand):
    spec = CommandSpec("return", 3, "Store or list r() return values")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        _handle_store_cmd(cmd.body, ctx.state.r, "r", ctx)


class EreturnCmd(BaseCommand):
    spec = CommandSpec("ereturn", 4, "Store or list e() estimation values")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        _handle_store_cmd(cmd.body, ctx.state.e, "e", ctx)


def _handle_store_cmd(body: str, store, prefix: str, ctx: ExecutionContext) -> None:
    toks = split_top_level_tokens(body.strip())
    if not toks or toks[0].lower() == "list":
        if store.scalars:
            ctx.console.write("scalars:")
            for k, v in store.scalars.items():
                ctx.console.write(f"  {prefix}({k}) = {v:g}")
        if store.macros:
            ctx.console.write("macros:")
            for k, v in store.macros.items():
                ctx.console.write(f"  {prefix}({k}) : \"{v}\"")
        if store.matrices:
            ctx.console.write("matrices:")
            for k, m in store.matrices.items():
                ctx.console.write(f"  {prefix}({k}) : {m.shape[0]} x {m.shape[1]}")
        return
    sub = toks[0].lower()
    if sub == "clear":
        store.clear()
        return
    if sub in ("scalar", "sca") and "=" in body:
        rest = body.split(None, 1)[1]
        k, expr = rest.split("=", 1)
        res = eval_expression(expr, ctx.dataset, ctx.state, force_scalar=True)
        store.scalars[k.strip()] = float(res.to_scalar())
        return
    if sub in ("local", "loc") and len(toks) >= 2:
        k = toks[1]
        val = body.split(k, 1)[1].strip()
        if val.startswith("="):
            res = eval_expression(val[1:], ctx.dataset, ctx.state, force_scalar=True)
            val = str(res.to_scalar())
        else:
            val = strip_quotes(val)
        store.macros[k] = val
        return
    if sub in ("matrix", "mat") and "=" in body:
        rest = body.split(None, 1)[1]
        k, expr = rest.split("=", 1)
        m = ctx.state.matrices.get(expr.strip())
        if m is None:
            raise StataError(f"matrix {expr.strip()} not found", rc=111)
        store.matrices[k.strip()] = m.copy()
        return
    raise StataError(f"unsupported return subcommand: {body}", rc=198)


class ArgsCmd(BaseCommand):
    spec = CommandSpec("args", 4, "Bind positional arguments `1', `2', ... to local names")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        for idx, name in enumerate(cmd.main_args.split(), start=1):
            ctx.state.locals[name] = ctx.state.locals.get(str(idx), "")


class SyntaxCmd(BaseCommand):
    spec = CommandSpec("syntax", 3, "Parse calling arguments inside a Stata program")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        raw_call = ctx.state.locals.get("0", "")
        parsed_call = parse_command_line(f"__prog {raw_call}")
        spec_str = f"{cmd.body}, {cmd.options_str}" if cmd.options_str else cmd.body.strip()

        # Split spec into main part and optional `[ , options ]`
        main_spec = spec_str
        opt_spec = ""
        m_opt = re.search(r"\[\s*,(.*)\]\s*$", spec_str)
        if m_opt:
            opt_spec = m_opt.group(1).strip()
            main_spec = spec_str[: m_opt.start()].strip()
        elif "," in spec_str:
            main_spec, opt_spec = spec_str.split(",", 1)

        if "varlist" in main_spec.lower():
            if parsed_call.main_args.strip():
                vars_expanded = expand_varlist(parsed_call.main_args, ctx.dataset)
                ctx.state.locals["varlist"] = " ".join(vars_expanded)
            elif "[varlist" in main_spec.lower() or "[anything" in main_spec.lower():
                ctx.state.locals["varlist"] = " ".join(ctx.dataset.columns)
            else:
                raise StataError("varlist required", rc=100)
        elif "anything" in main_spec.lower():
            ctx.state.locals["anything"] = parsed_call.main_args.strip()
        elif "varname" in main_spec.lower():
            vars_expanded = expand_varlist(parsed_call.main_args, ctx.dataset)
            if len(vars_expanded) != 1:
                raise StataError("single varname required", rc=103)
            ctx.state.locals["varlist"] = vars_expanded[0]

        ctx.state.locals["if"] = f"if {parsed_call.if_expr}" if parsed_call.if_expr else ""
        ctx.state.locals["in"] = f"in {parsed_call.in_raw}" if parsed_call.in_raw else ""
        ctx.state.locals["using"] = f'using "{parsed_call.using_path}"' if parsed_call.using_path else ""

        # Parse declared options, e.g. `Level(real 95) Robust Title(string)`
        opt_spec_flat = opt_spec.replace("[", " ").replace("]", " ")
        for tok in split_top_level_tokens(opt_spec_flat):
            tok_clean = tok.strip()
            if not tok_clean or tok_clean == "*":
                continue
            lp = tok_clean.find("(")
            if lp != -1 and tok_clean.endswith(")"):
                opt_raw = tok_clean[:lp]
                opt_key = opt_raw.lower()
                min_len = sum(1 for c in opt_raw if c.isupper()) or len(opt_key)
                inner = tok_clean[lp + 1 : -1].split()
                default_val = inner[1] if len(inner) >= 2 else ""
                matched_val = default_val
                for k_in, v_in in parsed_call.options.items():
                    if opt_key.startswith(k_in) and len(k_in) >= min_len:
                        matched_val = v_in
                        break
                ctx.state.locals[opt_key] = "" if matched_val is True or matched_val is False else str(matched_val)
            else:
                opt_key = tok_clean.lower()
                min_len = sum(1 for c in tok_clean if c.isupper()) or len(opt_key)
                matched = any(opt_key.startswith(k_in) and len(k_in) >= min_len for k_in in parsed_call.options)
                ctx.state.locals[opt_key] = opt_key if matched else ""


class MarksampleCmd(BaseCommand):
    spec = CommandSpec("marksample", 6, "Create tempvar touse marking valid observations")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        touse = ctx.state.new_tempvar()
        ctx.state.locals["touse"] = touse
        n = ctx.dataset.nobs
        mask = np.ones(n, dtype=bool)
        if_clause = ctx.state.locals.get("if", "").strip()
        in_clause = ctx.state.locals.get("in", "").strip()
        if if_clause.lower().startswith("if "):
            res = eval_expression(if_clause[3:], ctx.dataset, ctx.state)
            mask &= res.as_truth_mask(n)
        if in_clause.lower().startswith("in "):
            from openstata.lang.parser import parse_in_range

            s, e = parse_in_range(in_clause[3:].strip(), n)
            in_m = np.zeros(n, dtype=bool)
            in_m[s - 1 : e] = True
            mask &= in_m
        if "novarlist" not in cmd.options and ctx.state.locals.get("varlist"):
            for v in ctx.state.locals["varlist"].split():
                if v in ctx.dataset.df.columns:
                    mask &= ~ctx.dataset.get_missing_mask(v)
        ctx.dataset.add_numeric_var(touse, mask.astype(np.float64), stata_type="byte", overwrite=True)


class DoCmd(BaseCommand):
    spec = CommandSpec("do", 2, "Execute a .do script file")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        toks = split_top_level_tokens(cmd.body)
        if not toks:
            raise StataError("filename required", rc=198)
        fpath = strip_quotes(toks[0])
        args = [strip_quotes(t) for t in toks[1:]]
        ctx.runner.run_file(fpath, args=args)


class RunCmd(BaseCommand):
    spec = CommandSpec("run", 3, "Execute a .do script file quietly")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        toks = split_top_level_tokens(cmd.body)
        if not toks:
            raise StataError("filename required", rc=198)
        fpath = strip_quotes(toks[0])
        args = [strip_quotes(t) for t in toks[1:]]
        orig_console = ctx.runner.console
        from openstata.plugins.base import DefaultConsole

        ctx.runner.console = DefaultConsole(ctx.state, silent=True)
        try:
            ctx.runner.run_file(fpath, args=args)
        finally:
            ctx.runner.console = orig_console


class LogCmd(BaseCommand):
    spec = CommandSpec("log", 3, "Open or close a session log file")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        toks = split_top_level_tokens(cmd.main_args)
        if toks and toks[0].lower() == "close":
            if ctx.state.log_file is not None:
                ctx.state.log_file.close()
                ctx.state.log_file = None
                ctx.console.write(f"log closed: {ctx.state.log_path}")
                ctx.state.log_path = None
            return
        if cmd.using_path:
            p = Path(cmd.using_path)
            if not p.is_absolute():
                p = ctx.state.cwd / p
            if p.exists() and not cmd.options.get("replace") and not cmd.options.get("append"):
                raise StataError(f"file {p} already exists", rc=602)
            p.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if cmd.options.get("append") else "w"
            ctx.state.log_file = p.open(mode, encoding="utf-8")
            ctx.state.log_path = p
            ctx.console.write(f"log opened: {p}")
            return
        raise StataError("syntax: log using <file> [, replace append] | log close", rc=198)


class ExitCmd(BaseCommand):
    spec = CommandSpec("exit", 2, "Exit current program/script")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        raise ExitSignal(0)


class AssertCmd(BaseCommand):
    spec = CommandSpec("assert", 2, "Verify boolean expression holds")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        from openstata.lang.executor import compute_sample_mask

        mask = compute_sample_mask(cmd, ctx.dataset, ctx.state)
        res = eval_expression(cmd.main_args, ctx.dataset, ctx.state)
        truth = res.as_truth_mask(ctx.dataset.nobs)
        if not np.all(truth[mask]):
            n_bad = int(np.sum(~truth[mask]))
            raise StataError(f"assertion is false ({n_bad} contradictions)", rc=9)


class WhichCmd(BaseCommand):
    spec = CommandSpec("which", 5, "Show whether a command is built-in, plugin, or .ado")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        name = cmd.main_args.strip()
        if not name:
            raise StataError("command name required", rc=198)
        c = ctx.runner.registry.resolve(name)
        if c is not None:
            ctx.console.write(f"{c.spec.name}: {c.spec.source} (package: {c.spec.package_name})")
            return
        for base in ctx.state.adopath:
            p = (base if base.is_absolute() else ctx.state.cwd / base) / f"{name}.ado"
            if p.exists():
                ctx.console.write(str(p.resolve()))
                return
        raise StataError(f"command {name} not found as built-in, plugin, or ado-file", rc=111)


class HelpCmd(BaseCommand):
    spec = CommandSpec("help", 1, "List available commands")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        target = cmd.main_args.strip().lower()
        specs = ctx.runner.registry.list_commands()
        if target:
            c = ctx.runner.registry.resolve(target)
            if c is None:
                raise StataError(f"help topic '{target}' not found", rc=111)
            ctx.console.write(f"{c.spec.name} [{c.spec.source}] - {c.spec.summary}")
            return
        ctx.console.write("Available commands:")
        for sp in specs:
            ctx.console.write(f"  {sp.name:<18} [{sp.source}] {sp.summary}")


class AdopathCmd(BaseCommand):
    spec = CommandSpec("adopath", 7, "Display or modify the .ado search path")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        toks = split_top_level_tokens(cmd.main_args)
        if not toks:
            for idx, p in enumerate(ctx.state.adopath, start=1):
                ctx.console.write(f"  [{idx}] {p}")
            return
        op = toks[0]
        if op in ("+", "++") and len(toks) >= 2:
            p = Path(strip_quotes(toks[1]))
            if op == "++":
                ctx.state.adopath.insert(0, p)
            else:
                ctx.state.adopath.append(p)
            return
        if op == "-" and len(toks) >= 2:
            target = Path(strip_quotes(toks[1]))
            ctx.state.adopath = [p for p in ctx.state.adopath if p != target]
            return
        raise StataError("syntax: adopath [+ | ++ | - <dir>]", rc=198)


class PkgCmd(BaseCommand):
    spec = CommandSpec("pkg", 3, "Manage third-party Python and .ado extension packages")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        toks = split_top_level_tokens(cmd.main_args)
        if not toks or toks[0].lower() == "list":
            ctx.console.write("Installed plugin commands:")
            plugins = [s for s in ctx.runner.registry.list_commands() if s.source == "plugin"]
            if not plugins:
                ctx.console.write("  (none)")
            for s in plugins:
                ctx.console.write(f"  {s.name:<18} (package: {s.package_name}) {s.summary}")
            plus_dir = ctx.state.adopath[-1] if ctx.state.adopath else Path.home() / ".openstata" / "ado" / "plus"
            ados = sorted(p.name for p in plus_dir.glob("*.ado")) if plus_dir.exists() else []
            ctx.console.write(f"Installed .ado files in {plus_dir}: {', '.join(ados) if ados else '(none)'}")
            return
        sub = toks[0].lower()
        if sub == "install" and len(toks) >= 2:
            target = strip_quotes(toks[1])
            plus_dir = ctx.state.adopath[-1] if ctx.state.adopath else Path.home() / ".openstata" / "ado" / "plus"
            msg = install_package(target, plus_dir)
            ctx.runner.registry.load_entry_points(force=True)
            ctx.console.write(msg)
            return
        if sub == "remove" and len(toks) >= 2:
            target = strip_quotes(toks[1])
            plus_dir = ctx.state.adopath[-1] if ctx.state.adopath else Path.home() / ".openstata" / "ado" / "plus"
            ado_cand = plus_dir / (target if target.endswith(".ado") else f"{target}.ado")
            if ado_cand.exists():
                ado_cand.unlink()
                ctx.console.write(f"Removed {ado_cand.name}")
                return
            proc = subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", target],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                raise StataError(f"failed to remove package {target}: {proc.stderr}", rc=601)
            ctx.console.write(f"Uninstalled Python package {target}")
            return
        raise StataError("syntax: pkg list | pkg install <path_or_pkg> | pkg remove <name>", rc=198)
