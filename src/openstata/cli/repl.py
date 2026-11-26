"""REPL implementation for OpenStata.
Provides _needs_more_lines detection and run_repl interactive loop.
"""

from __future__ import annotations

import sys
from openstata import __version__
from openstata.lang.executor import ScriptRunner
from openstata.core.errors import ExitSignal, StataError


def _needs_more_lines(buffer: list[str]) -> bool:
    """Detect whether REPL input is inside an open brace, input, or program block.
    """
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
