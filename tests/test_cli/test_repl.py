"""REPL unit test for OpenStata.
Ensures that the interactive loop processes a simple command and exits cleanly.
"""

import builtins
import unittest
import builtins
from openstata.cli.repl import run_repl
from openstata.plugins.registry import CommandRegistry
from openstata.commands import register_builtin_commands
from openstata.lang.executor import ScriptRunner


def _make_runner() -> ScriptRunner:
    registry = CommandRegistry()
    register_builtin_commands(registry)
    return ScriptRunner(registry=registry)

class TestREPL(unittest.TestCase):
    def test_basic(self):
        inputs = iter(["display 1+1", "exit"])
        def fake_input(prompt: str = ""):
            try:
                return next(inputs)
            except StopIteration:
                raise EOFError
        # Monkey patch builtins.input
        original_input = builtins.input
        builtins.input = fake_input
        try:
            rc = run_repl(_make_runner())
            self.assertEqual(rc, 0)
        finally:
            builtins.input = original_input
from openstata.cli.repl import run_repl
from openstata.plugins.registry import CommandRegistry
from openstata.commands import register_builtin_commands
from openstata.lang.executor import ScriptRunner


def _make_runner() -> ScriptRunner:
    """Create a ScriptRunner with built‑in commands registered."""
    registry = CommandRegistry()
    register_builtin_commands(registry)
    return ScriptRunner(registry=registry)


def test_repl_basic(monkeypatch):
    """Feed a simple command to the REPL and verify it exits with code 0.

    The input sequence provides a Stata command ``display 1+1`` followed by ``exit``.
    The REPL should execute the command (printing ``2``) and then return.
    """
    inputs = iter(["display 1+1", "exit"])
    # Monkey‑patch builtins.input to return the next line, then raise EOFError.
    def fake_input(prompt: str = ""):
        try:
            return next(inputs)
        except StopIteration:
            raise EOFError
    monkeypatch.setattr(builtins, "input", fake_input)

    rc = run_repl(_make_runner())
    assert rc == 0
