"""In-memory Stata-compatible Dataset representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from openstata.core.errors import StataError
from openstata.core.missing import (
    MISSING_DOT,
    MISSING_NONE,
    format_missing_code,
    missing_codes_from_array,
)


STATA_NUMERIC_TYPES = {"byte", "int", "long", "float", "double"}


@dataclass
class VariableMeta:
    """Metadata for a single variable in a Dataset."""

    name: str
    stata_type: str = "double"  # byte, int, long, float, double, str#, strL
    fmt: str = "%9.0g"
    label: str = ""
    value_label: str = ""

    def is_string(self) -> bool:
        return self.stata_type.startswith("str")


@dataclass
class ValueLabelTable:
    """Mapping from integer values to string labels."""

    name: str
    mapping: dict[int, str] = field(default_factory=dict)

    def copy(self) -> "ValueLabelTable":
        return ValueLabelTable(name=self.name, mapping=dict(self.mapping))


class Dataset:
    """Columnar dataset with Stata metadata, value labels, and missing codes."""

    def __init__(
        self,
        df: pd.DataFrame | None = None,
        meta: dict[str, VariableMeta] | None = None,
        missing_codes: dict[str, np.ndarray] | None = None,
        value_labels: dict[str, ValueLabelTable] | None = None,
        data_label: str = "",
        characteristics: dict[str, dict[str, str]] | None = None,
        sort_by: list[str] | None = None,
        source_path: str | None = None,
    ) -> None:
        self.df: pd.DataFrame = df.reset_index(drop=True) if df is not None else pd.DataFrame()
        self.meta: dict[str, VariableMeta] = meta if meta is not None else {}
        self.missing_codes: dict[str, np.ndarray] = missing_codes if missing_codes is not None else {}
        self.value_labels: dict[str, ValueLabelTable] = value_labels if value_labels is not None else {}
        self.data_label: str = data_label
        self.characteristics: dict[str, dict[str, str]] = (
            characteristics if characteristics is not None else {}
        )
        self.sort_by: list[str] = list(sort_by) if sort_by else []
        self.source_path: str | None = source_path
        self.dirty: bool = False
        self.panel_var: str | None = None
        self.time_var: str | None = None
        self._sync_metadata()

    @property
    def nobs(self) -> int:
        return len(self.df)

    @property
    def nvars(self) -> int:
        return len(self.df.columns)

    @property
    def columns(self) -> list[str]:
        return list(self.df.columns)

    def _sync_metadata(self) -> None:
        """Ensure every DataFrame column has matching VariableMeta and missing codes."""
        n = len(self.df)
        for col in self.df.columns:
            series = self.df[col]
            if col not in self.meta:
                if pd.api.types.is_string_dtype(series) or series.dtype == object:
                    max_len = max(
                        (len(str(v).encode("utf-8")) for v in series.dropna()),
                        default=1,
                    )
                    stype = "strL" if max_len > 2045 else f"str{max(1, max_len)}"
                    fmt = f"%{max(1, min(max_len, 244))}s"
                    self.meta[col] = VariableMeta(name=col, stata_type=stype, fmt=fmt)
                elif pd.api.types.is_integer_dtype(series):
                    self.df[col] = series.astype(np.float64)
                    self.meta[col] = VariableMeta(name=col, stata_type="long", fmt="%12.0g")
                else:
                    self.df[col] = pd.to_numeric(series, errors="coerce").astype(np.float64)
                    self.meta[col] = VariableMeta(name=col, stata_type="double", fmt="%9.0g")

            if self.meta[col].is_string():
                self.df[col] = self.df[col].fillna("").astype(str)
                self.missing_codes.pop(col, None)
            else:
                if not pd.api.types.is_float_dtype(self.df[col]):
                    self.df[col] = pd.to_numeric(self.df[col], errors="coerce").astype(np.float64)
                mc = self.missing_codes.get(col)
                if mc is None or len(mc) != n:
                    self.missing_codes[col] = missing_codes_from_array(self.df[col].to_numpy())

        stale = [k for k in self.meta if k not in self.df.columns]
        for k in stale:
            self.meta.pop(k, None)
            self.missing_codes.pop(k, None)

    def clear(self) -> None:
        self.df = pd.DataFrame()
        self.meta.clear()
        self.missing_codes.clear()
        self.value_labels.clear()
        self.data_label = ""
        self.characteristics.clear()
        self.sort_by = []
        self.source_path = None
        self.panel_var = None
        self.time_var = None
        self.dirty = False

    def copy(self) -> "Dataset":
        ds = Dataset(
            df=self.df.copy(deep=True),
            meta={k: VariableMeta(**vars(v)) for k, v in self.meta.items()},
            missing_codes={k: v.copy() for k, v in self.missing_codes.items()},
            value_labels={k: v.copy() for k, v in self.value_labels.items()},
            data_label=self.data_label,
            characteristics={k: dict(v) for k, v in self.characteristics.items()},
            sort_by=list(self.sort_by),
            source_path=self.source_path,
        )
        ds.dirty = self.dirty
        ds.panel_var = self.panel_var
        ds.time_var = self.time_var
        return ds

    def require_var(self, name: str) -> str:
        if name not in self.df.columns:
            raise StataError(f"variable {name} not found", rc=111)
        return name

    def is_string(self, name: str) -> bool:
        self.require_var(name)
        return self.meta[name].is_string()

    def get_numeric(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        self.require_var(name)
        if self.is_string(name):
            raise StataError(f"type mismatch: {name} is string", rc=109)
        vals = self.df[name].to_numpy(dtype=np.float64, copy=False)
        mc = self.missing_codes[name]
        return vals, mc

    def get_missing_mask(self, name: str) -> np.ndarray:
        self.require_var(name)
        if self.is_string(name):
            return (self.df[name].astype(str) == "").to_numpy()
        return self.missing_codes[name] > 0

    def add_numeric_var(
        self,
        name: str,
        values: np.ndarray,
        missing_codes: np.ndarray | None = None,
        stata_type: str = "double",
        fmt: str = "%9.0g",
        label: str = "",
        overwrite: bool = False,
    ) -> None:
        if name in self.df.columns and not overwrite:
            raise StataError(f"variable {name} already defined", rc=110)
        arr = np.asarray(values, dtype=np.float64)
        mc = (
            np.asarray(missing_codes, dtype=np.uint8)
            if missing_codes is not None
            else missing_codes_from_array(arr)
        )
        arr = arr.copy()
        arr[mc > 0] = np.nan
        self.df[name] = arr
        self.missing_codes[name] = mc.copy()
        self.meta[name] = VariableMeta(name=name, stata_type=stata_type, fmt=fmt, label=label)
        self.dirty = True

    def add_string_var(
        self,
        name: str,
        values: list[str] | pd.Series,
        label: str = "",
        overwrite: bool = False,
    ) -> None:
        if name in self.df.columns and not overwrite:
            raise StataError(f"variable {name} already defined", rc=110)
        s = pd.Series(values, index=self.df.index if len(self.df) else None).fillna("").astype(str)
        max_len = max((len(v.encode("utf-8")) for v in s), default=1)
        stype = "strL" if max_len > 2045 else f"str{max(1, max_len)}"
        fmt = f"%{max(1, min(max_len, 244))}s"
        self.df[name] = s
        self.missing_codes.pop(name, None)
        self.meta[name] = VariableMeta(name=name, stata_type=stype, fmt=fmt, label=label)
        self.dirty = True

    def drop_vars(self, names: list[str]) -> None:
        for n in names:
            self.require_var(n)
        self.df = self.df.drop(columns=names)
        for n in names:
            self.meta.pop(n, None)
            self.missing_codes.pop(n, None)
        self.sort_by = [s for s in self.sort_by if s not in names]
        self.dirty = True

    def keep_vars(self, names: list[str]) -> None:
        keep_set = set(names)
        to_drop = [c for c in self.df.columns if c not in keep_set]
        if to_drop:
            self.drop_vars(to_drop)
        # Preserve requested order if needed
        ordered = [n for n in names if n in self.df.columns]
        self.df = self.df[ordered]

    def keep_rows(self, mask: np.ndarray) -> None:
        mask = np.asarray(mask, dtype=bool)
        self.df = self.df.loc[mask].reset_index(drop=True)
        for col, mc in list(self.missing_codes.items()):
            self.missing_codes[col] = mc[mask]
        self.dirty = True

    def rename_var(self, old: str, new: str) -> None:
        self.require_var(old)
        if new in self.df.columns and new != old:
            raise StataError(f"variable {new} already defined", rc=110)
        self.df = self.df.rename(columns={old: new})
        meta = self.meta.pop(old)
        meta.name = new
        self.meta[new] = meta
        if old in self.missing_codes:
            self.missing_codes[new] = self.missing_codes.pop(old)
        self.sort_by = [new if s == old else s for s in self.sort_by]
        if old in self.characteristics:
            self.characteristics[new] = self.characteristics.pop(old)
        self.dirty = True

    def sort_rows(self, by_vars: list[str], ascending: list[bool] | None = None, stable: bool = True) -> None:
        if ascending is None:
            ascending = [True] * len(by_vars)
        if len(by_vars) != len(ascending):
            raise StataError("sort variable / direction length mismatch", rc=198)
        for v in by_vars:
            self.require_var(v)
        if self.nobs == 0:
            self.sort_by = list(by_vars)
            return

        # Build sort keys honoring Stata extended missing ordering (finite < . < .a < ... < .z).
        idx = np.arange(self.nobs)
        # Sort from least significant key to most significant using stable mergesort
        for var, asc in reversed(list(zip(by_vars, ascending))):
            if self.is_string(var):
                col = self.df[var].to_numpy()[idx]
                order = np.argsort(col, kind="mergesort")
                if not asc:
                    order = order[::-1]
                idx = idx[order]
            else:
                vals = self.df[var].to_numpy(dtype=np.float64)[idx]
                mc = self.missing_codes[var][idx]
                is_miss = mc > 0
                # Map missing values after all finite numbers via tuple/lexsort:
                # primary key: is_miss (0 or 1), secondary: mc if miss else vals
                primary = is_miss.astype(np.int8)
                secondary = np.where(is_miss, mc.astype(np.float64), vals)
                if not asc:
                    # Descending: reverse non-missing; missing values still sort appropriately
                    # In gsort -var, missing values come FIRST in Stata!
                    primary = -primary
                    secondary = -secondary
                order = np.lexsort((secondary, primary))
                idx = idx[order]

        self.df = self.df.iloc[idx].reset_index(drop=True)
        for col in list(self.missing_codes.keys()):
            self.missing_codes[col] = self.missing_codes[col][idx]
        if all(ascending):
            self.sort_by = list(by_vars)
        else:
            self.sort_by = []
        self.dirty = True

    def format_value(self, var: str, row_idx: int, nolabel: bool = False) -> str:
        """Render a cell value as Stata displays it."""
        if self.is_string(var):
            return str(self.df[var].iat[row_idx])
        mc = int(self.missing_codes[var][row_idx])
        if mc > MISSING_NONE:
            return format_missing_code(mc)
        val = float(self.df[var].iat[row_idx])
        meta = self.meta[var]
        if not nolabel and meta.value_label and meta.value_label in self.value_labels:
            if float(int(val)) == val:
                lbl = self.value_labels[meta.value_label].mapping.get(int(val))
                if lbl is not None:
                    return lbl
        if float(int(val)) == val and abs(val) < 1e12:
            return str(int(val))
        return f"{val:.6g}"

    def cast_numeric_type(self, name: str, target_type: str) -> None:
        self.require_var(name)
        if target_type not in STATA_NUMERIC_TYPES:
            raise StataError(f"invalid numeric type {target_type}", rc=198)
        self.meta[name].stata_type = target_type
        self.dirty = True
