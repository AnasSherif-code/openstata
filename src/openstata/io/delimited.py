"""Delimited text (CSV/TSV) import and export."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from openstata.core.dataset import Dataset
from openstata.core.errors import StataError


def read_delimited(
    path: str | Path,
    delimiter: str | None = None,
    varnames: int | str = 1,
    encoding: str = "utf-8",
) -> Dataset:
    p = Path(path)
    if not p.exists():
        raise StataError(f"file {p} not found", rc=601)
    sep = delimiter if delimiter is not None else ","
    header = 0 if varnames == 1 else None
    df = pd.read_csv(p, sep=sep, header=header, encoding=encoding)
    if header is None:
        df.columns = [f"v{i + 1}" for i in range(len(df.columns))]
    else:
        df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    ds = Dataset(df=df, source_path=str(p))
    ds.dirty = False
    return ds


def write_delimited(
    dataset: Dataset,
    path: str | Path,
    delimiter: str = ",",
    replace: bool = False,
    novarnames: bool = False,
) -> None:
    p = Path(path)
    if p.exists() and not replace:
        raise StataError(f"file {p} already exists", rc=602)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Render extended missing codes as empty or Stata-style strings if needed
    out_df = dataset.df.copy()
    out_df.to_csv(p, sep=delimiter, index=False, header=not novarnames)
