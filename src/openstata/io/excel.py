"""Excel (.xlsx/.xls) import.

Reads a worksheet into a Dataset. Requires the optional ``openpyxl``
dependency. By default the first row of the sheet is used as variable names;
``firstrow=False``-style loading (generated names) is available via
``varnames=0``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from openstata.core.dataset import Dataset
from openstata.core.errors import StataError

EXCEL_SUFFIXES = {".xlsx", ".xlsm", ".xlsb", ".xls", ".ods"}


def read_excel(
    path: str | Path,
    sheet: str | int | None = None,
    varnames: int | str = 1,
) -> Dataset:
    p = Path(path)
    if not p.exists():
        raise StataError(f"file {p} not found", rc=601)
    if p.suffix.lower() not in EXCEL_SUFFIXES:
        raise StataError(f"file {p} is not a recognized Excel file", rc=610)
    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:
        raise StataError(
            "import excel requires the openpyxl package; install it with:\n"
            "    pip install openpyxl",
            rc=610,
        ) from exc

    header = 0 if varnames == 1 else None
    try:
        # sheet_name=None makes pandas return a dict of all sheets; the
        # default must be the first sheet
        df = pd.read_excel(p, sheet_name=sheet if sheet is not None else 0, header=header)
    except ValueError as exc:
        # Unknown sheet name / index out of range
        raise StataError(f"failed to read Excel sheet: {exc}", rc=610) from exc
    if header is None:
        df.columns = [f"v{i + 1}" for i in range(len(df.columns))]
    else:
        df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    ds = Dataset(df=df, source_path=str(p))
    ds.dirty = False
    return ds
