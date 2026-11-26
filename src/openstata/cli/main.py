"""Interactive REPL and batch CLI entrypoint for OpenStata."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from openstata import __version__
from openstata.commands import register_builtin_commands
from openstata.core.errors import ExitSignal, StataError
from openstata.lang.executor import ScriptRunner
from openstata.plugins.registry import CommandRegistry
from .repl import _needs_more_lines, run_repl


def create_runner() -> ScriptRunner:
    """Create a fully wired ScriptRunner with built-in and plugin commands."""
    registry = CommandRegistry()
    register_builtin_commands(registry)
    return ScriptRunner(registry=registry)


def _needs_more_lines(buffer: list[str]) -> bool:
    """Detect whether REPL input is inside an open brace, input, or program block."""
    if not buffer:
        return False
    text = "\n".join(buffer)
    if buffer[-1].rstrip().endswith("///"):
        return True
    opens = text.count("{")
    closes = text.count("}")
    if opens > closes:
        return True
    first_low = buffer[0].strip().lower()
    if first_low.startswith("program ") or first_low == "input" or first_low.startswith("input "):
        return not any(ln.strip().lower() == "end" for ln in buffer[1:])
    return False


def run_repl(runner: ScriptRunner) -> int:
    print(f"OpenStata {__version__} (clean-room Python statistical engine)")
    print("License: PolyForm-Noncommercial-1.0.0 (study & research use only)")
    print("Type 'help' for commands, 'exit' or Ctrl-D to quit.\n")

    buf: list[str] = []
    while True:
        prompt = "> " if buf else ". "
        try:
            line = input(prompt)
        except EOFError:
            print("")
            return 0
        except KeyboardInterrupt:
            print("\n--Break--")
            buf.clear()
            continue

        if not buf and line.strip().lower() in ("quit", "q"):
            return 0

        buf.append(line)
        if _needs_more_lines(buf):
            continue

        src = "\n".join(buf)
        buf.clear()
        try:
            runner.run_source(src)
        except ExitSignal as sig:
            return sig.rc
        except StataError as err:
            print(str(err), file=sys.stderr)
        except Exception as exc:
            print(f"internal error: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="openstata", description="OpenStata statistical engine")
    parser.add_argument("target", nargs="?", help="Path to a .do script file, or 'do'")
    parser.add_argument("args", nargs="*", help="Arguments passed to the .do script")
    parser.add_argument("-b", "--batch", action="store_true", help="Run in batch mode")
    parser.add_argument("-e", "--execute", help="Execute inline Stata commands and exit")
    parser.add_argument("--version", action="version", version=f"openstata {__version__}")

    ns = parser.parse_args(argv)
    runner = create_runner()

    if ns.execute is not None:
        try:
            runner.run_source(ns.execute)
            return 0
        except ExitSignal as sig:
            return sig.rc
        except StataError as err:
            print(str(err), file=sys.stderr)
            return err.rc

    if ns.target is not None:
        script = ns.target
        script_args = list(ns.args)
        if script.lower() == "do" and script_args:
            script = script_args.pop(0)
        try:
            runner.run_file(Path(script), args=script_args)
            return 0
        except ExitSignal as sig:
            return sig.rc
        except StataError as err:
            print(str(err), file=sys.stderr)
            return err.rc

    return run_repl(runner)


if __name__ == "__main__":
    raise SystemExit(main())
