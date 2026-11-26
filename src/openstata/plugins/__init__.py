"""Plugin architecture for OpenStata."""

from openstata.plugins.base import BaseCommand, CommandSpec, ConsoleWriter, DefaultConsole, ExecutionContext
from openstata.plugins.registry import CommandRegistry, ENTRY_POINT_GROUP, install_package

__all__ = [
    "BaseCommand",
    "CommandRegistry",
    "CommandSpec",
    "ConsoleWriter",
    "DefaultConsole",
    "ENTRY_POINT_GROUP",
    "ExecutionContext",
    "install_package",
]
