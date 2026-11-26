"""Core data structures and execution state for OpenStata."""

from openstata.core.dataset import Dataset, ValueLabelTable, VariableMeta
from openstata.core.errors import BreakSignal, ContinueSignal, ExitSignal, StataError
from openstata.core.missing import (
    MISSING_DOT,
    MISSING_NONE,
    format_missing_code,
    missing_codes_from_array,
    parse_missing_literal,
    stata_compare,
)
from openstata.core.state import ResultsStore, StateStore

__all__ = [
    "BreakSignal",
    "ContinueSignal",
    "Dataset",
    "ExitSignal",
    "MISSING_DOT",
    "MISSING_NONE",
    "ResultsStore",
    "StateStore",
    "StataError",
    "ValueLabelTable",
    "VariableMeta",
    "format_missing_code",
    "missing_codes_from_array",
    "parse_missing_literal",
    "stata_compare",
]
