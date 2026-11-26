"""JASP (.jasp) file import.

A .jasp file is a ZIP archive. Its data table is stored as a CSV inside the
archive (typically ``data/database.csv`` in current JASP versions), with a
``meta.json`` describing the analysis. We locate the main data CSV, read it
with pandas, and drop the row-index column JASP writes alongside the data.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from openstata.core.dataset import Dataset
from openstata.core.errors import StataError

# Names JASP uses for its row-index column ("Unnamed: N" is what pandas
# calls an empty CSV header cell). Real data columns like `id` are never
# dropped, even when they happen to contain a 1..n counter.
_INDEX_NAMES = {"", "index", "..."}


def _find_data_csv(zf: zipfile.ZipFile) -> str:
    """Return the archive member holding the main data table."""
    names = zf.namelist()
    csvs = [n for n in names if n.lower().endswith(".csv")]
    if not csvs:
        raise StataError("no data table found in .jasp archive", rc=610)
    # Prefer JASP's canonical location, then the largest CSV by size
    def rank(name: str) -> tuple[int, int]:
        lower = name.lower()
        canonical = 0 if lower in ("data/database.csv", "database.csv") else 1
        size = zf.getinfo(name).file_size
        return (canonical, -size)

    return sorted(csvs, key=rank)[0]


def _drop_index_column(df: pd.DataFrame) -> pd.DataFrame:
    """Remove JASP's leading row-index column if present."""
    first = df.columns[0]
    first_str = str(first).strip()
    if first_str in _INDEX_NAMES or first_str.startswith("Unnamed"):
        values = df.iloc[:, 0].to_numpy()
        # Only drop a genuine 0/1-based row counter, never real data
        n = len(df)
        is_counter = (
            np_less_or_equal_range(values, 0) or np_less_or_equal_range(values, 1)
        )
        if is_counter:
            df = df.iloc[:, 1:]
    return df


def np_less_or_equal_range(values, start: int) -> bool:
    """True if values == start..start+n-1."""
    n = len(values)
    if n == 0:
        return False
    try:
        nums = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    except (TypeError, ValueError):
        return False
    if np.isnan(nums).any():
        return False
    return bool(np.array_equal(nums, np.arange(start, start + n)))


def read_jasp(path: str | Path) -> Dataset:
    p = Path(path)
    if not p.exists():
        raise StataError(f"file {p} not found", rc=601)
    if not zipfile.is_zipfile(p):
        raise StataError(f"file {p} is not a valid .jasp archive", rc=610)
    try:
        with zipfile.ZipFile(p) as zf:
            member = _find_data_csv(zf)
            raw = zf.read(member)
            df = pd.read_csv(
                pd.io.common.BytesIO(raw),
                sep=",",
                encoding="utf-8",
            )
            # Read metadata for the dataset label if present
            label = ""
            meta_name = next(
                (n for n in zf.namelist() if n.lower().endswith("meta.json")), None
            )
            if meta_name:
                try:
                    meta = json.loads(zf.read(meta_name).decode("utf-8", errors="replace"))
                    label = str(meta.get("name", "") or "")
                except (json.JSONDecodeError, UnicodeDecodeError):
                    label = ""
    except (zipfile.BadZipFile, pd.errors.ParserError) as exc:
        raise StataError(f"failed to read .jasp archive: {exc}", rc=610) from exc

    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    df = _drop_index_column(df)
    ds = Dataset(df=df, source_path=str(p), data_label=label)
    ds.dirty = False
    return ds
