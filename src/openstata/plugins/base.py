"""Public plugin interface for OpenStata commands (built-in and third-party)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from openstata.core.dataset import Dataset
    from openstata.core.state import StateStore
    from openstata.lang.parser import ParsedCommand


@dataclass
class CommandSpec:
    """Declarative metadata for a Stata-style command."""

    name: str
    min_abbrev: int | None = None
    summary: str = ""
    source: str = "builtin"  # builtin | plugin | ado
    package_name: str = "openstata"


class ConsoleWriter(Protocol):
    def write(self, text: str = "") -> None: ...
    def write_error(self, text: str) -> None: ...


class DefaultConsole:
    """Simple stdout/log-aware console writer."""

    def __init__(self, state: "StateStore | None" = None, silent: bool = False) -> None:
        self.state = state
        self.silent = silent

    def write(self, text: str = "") -> None:
        if self.silent:
            return
        print(text)
        if self.state is not None and self.state.log_file is not None:
            self.state.log_file.write(text + "\n")
            self.state.log_file.flush()

    def write_error(self, text: str) -> None:
        print(text)
        if self.state is not None and self.state.log_file is not None:
            self.state.log_file.write(text + "\n")
            self.state.log_file.flush()


@dataclass
class ExecutionContext:
    """Context passed to every command implementation."""

    dataset: "Dataset"
    state: "StateStore"
    console: ConsoleWriter
    runner: Any  # ScriptRunner instance for nested do/run/ado calls


class BaseCommand:
    """Base class for built-in and third-party Python OpenStata commands."""

    spec: CommandSpec = CommandSpec(name="")

    def run(self, cmd: "ParsedCommand", ctx: ExecutionContext) -> None:
        raise NotImplementedError
