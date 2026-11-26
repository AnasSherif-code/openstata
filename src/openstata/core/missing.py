"""Extended missing-value utilities matching Stata semantics (., .a - .z).

In Stata, missing values satisfy:
    all finite numbers < . < .a < .b < ... < .z
Missing codes are stored alongside numeric columns as uint8 arrays:
    0 -> non-missing
    1 -> '.'
    2..27 -> '.a'..'.z'
"""

from __future__ import annotations

import numpy as np

MISSING_NONE = 0
MISSING_DOT = 1
CODE_TO_LABEL: dict[int, str] = {1: "."}
LABEL_TO_CODE: dict[str, int] = {".": 1}

for _idx, _ch in enumerate("abcdefghijklmnopqrstuvwxyz", start=2):
    _lbl = f".{_ch}"
    CODE_TO_LABEL[_idx] = _lbl
    LABEL_TO_CODE[_lbl] = _idx


def parse_missing_literal(token: str) -> int | None:
    """Return missing code (1..27) if token is '.' or '.a'..'.z', else None."""
    return LABEL_TO_CODE.get(token.strip())


def format_missing_code(code: int) -> str:
    """Return '.' or '.a'..'.z' for missing code 1..27."""
    return CODE_TO_LABEL.get(int(code), ".")


def missing_codes_from_array(values: np.ndarray) -> np.ndarray:
    """Build default uint8 missing-code array (1 for NaN, 0 otherwise)."""
    arr = np.asarray(values)
    out = np.zeros(len(arr), dtype=np.uint8)
    if arr.dtype.kind in ("f", "c"):
        out[np.isnan(arr)] = MISSING_DOT
    return out


def stata_compare(
    left_vals: np.ndarray,
    left_mc: np.ndarray,
    right_vals: np.ndarray,
    right_mc: np.ndarray,
    op: str,
) -> np.ndarray:
    """Elementwise comparison honoring Stata ordering (finite < . < .a < ... < .z).

    Supported ops: '==', '!=', '<', '<=', '>', '>='.
    """
    lm = left_mc > 0
    rm = right_mc > 0
    both_valid = (~lm) & (~rm)
    both_missing = lm & rm

    result = np.zeros(np.broadcast_shapes(left_vals.shape, right_vals.shape), dtype=bool)

    if op == "==":
        result[both_valid] = left_vals[both_valid] == right_vals[both_valid]
        result[both_missing] = left_mc[both_missing] == right_mc[both_missing]
    elif op == "!=":
        result[both_valid] = left_vals[both_valid] != right_vals[both_valid]
        result[both_missing] = left_mc[both_missing] != right_mc[both_missing]
        result[lm ^ rm] = True
    elif op == "<":
        result[both_valid] = left_vals[both_valid] < right_vals[both_valid]
        result[(~lm) & rm] = True
        result[both_missing] = left_mc[both_missing] < right_mc[both_missing]
    elif op == "<=":
        result[both_valid] = left_vals[both_valid] <= right_vals[both_valid]
        result[(~lm) & rm] = True
        result[both_missing] = left_mc[both_missing] <= right_mc[both_missing]
    elif op == ">":
        result[both_valid] = left_vals[both_valid] > right_vals[both_valid]
        result[lm & (~rm)] = True
        result[both_missing] = left_mc[both_missing] > right_mc[both_missing]
    elif op == ">=":
        result[both_valid] = left_vals[both_valid] >= right_vals[both_valid]
        result[lm & (~rm)] = True
        result[both_missing] = left_mc[both_missing] >= right_mc[both_missing]
    else:
        raise ValueError(f"Unsupported comparison operator: {op}")
    return result
