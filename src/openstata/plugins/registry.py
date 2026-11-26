"""Command registry, abbreviation resolver, entry-point loader, and package manager."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
import subprocess
import sys
from typing import Type

from openstata.core.errors import StataError
from openstata.plugins.base import BaseCommand, CommandSpec

ENTRY_POINT_GROUP = "openstata.commands"


class CommandRegistry:
    """Resolves built-in, Python entry-point, and .ado commands with Stata abbreviations."""

    def __init__(self) -> None:
        self._commands: dict[str, BaseCommand] = {}
        self._loaded_entry_points = False

    def register(self, command: BaseCommand | Type[BaseCommand]) -> None:
        inst = command() if isinstance(command, type) else command
        name = inst.spec.name.lower()
        if not name:
            raise ValueError("CommandSpec.name must not be empty")
        self._commands[name] = inst

    def load_entry_points(self, force: bool = False) -> list[str]:
        """Discover third-party Python plugins registered under `openstata.commands`."""
        if self._loaded_entry_points and not force:
            return []
        loaded: list[str] = []
        eps = metadata.entry_points()
        group_eps = eps.select(group=ENTRY_POINT_GROUP) if hasattr(eps, "select") else eps.get(ENTRY_POINT_GROUP, [])
        for ep in group_eps:
            obj = ep.load()
            inst = obj() if isinstance(obj, type) else obj
            if not getattr(inst.spec, "name", ""):
                inst.spec = CommandSpec(
                    name=ep.name.lower(),
                    source="plugin",
                    package_name=getattr(ep.dist, "name", ep.name) if hasattr(ep, "dist") else ep.name,
                )
            else:
                inst.spec.source = "plugin"
                if hasattr(ep, "dist") and ep.dist is not None:
                    inst.spec.package_name = ep.dist.name
            self.register(inst)
            loaded.append(inst.spec.name)
        self._loaded_entry_points = True
        return loaded

    def resolve(self, raw_name: str) -> BaseCommand | None:
        name = raw_name.lower()
        if name in self._commands:
            return self._commands[name]

        # Match minimum unambiguous abbreviation
        matches: list[BaseCommand] = []
        for cmd_name, inst in self._commands.items():
            min_len = inst.spec.min_abbrev if inst.spec.min_abbrev is not None else len(cmd_name)
            if len(name) >= min_len and cmd_name.startswith(name):
                matches.append(inst)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise StataError(f"ambiguous command abbreviation '{raw_name}'", rc=199)
        return None

    def list_commands(self) -> list[CommandSpec]:
        return sorted((c.spec for c in self._commands.values()), key=lambda s: s.name)


def install_package(spec: str, ado_plus_dir: Path) -> str:
    """Install either a local/archive .ado package or a Python plugin package."""
    p = Path(spec)
    # Local directory containing .ado files -> copy to PLUS directory
    if p.exists() and p.is_dir() and any(p.rglob("*.ado")):
        ado_plus_dir.mkdir(parents=True, exist_ok=True)
        copied: list[str] = []
        for ado_file in p.rglob("*.ado"):
            target = ado_plus_dir / ado_file.name
            target.write_text(ado_file.read_text(encoding="utf-8"), encoding="utf-8")
            copied.append(ado_file.name)
        return f"Installed .ado files into {ado_plus_dir}: {', '.join(copied)}"

    # Python package (local directory with pyproject.toml, wheel, git+https://, or PyPI name)
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", spec],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise StataError(f"package install failed:\n{proc.stderr or proc.stdout}", rc=601)
    return proc.stdout.strip() or f"Installed Python package '{spec}'."
