"""Execution state: macro scopes, scalars, matrices, r()/e()/c() stores, and adopath."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ResultsStore:
    """Stores r(), e(), or s() return values."""

    scalars: dict[str, float] = field(default_factory=dict)
    macros: dict[str, str] = field(default_factory=dict)
    matrices: dict[str, np.ndarray] = field(default_factory=dict)
    matrix_colnames: dict[str, list[str]] = field(default_factory=dict)
    matrix_rownames: dict[str, list[str]] = field(default_factory=dict)

    def clear(self) -> None:
        self.scalars.clear()
        self.macros.clear()
        self.matrices.clear()
        self.matrix_colnames.clear()
        self.matrix_rownames.clear()

    def get_value(self, key: str) -> Any:
        if key in self.scalars:
            return self.scalars[key]
        if key in self.macros:
            return self.macros[key]
        if key in self.matrices:
            return self.matrices[key]
        return None


class StateStore:
    """Global & scoped execution state for an OpenStata session."""

    def __init__(self) -> None:
        self.globals: dict[str, str] = {}
        self.local_stack: list[dict[str, str]] = [{}]
        self.scalars: dict[str, float] = {}
        self.matrices: dict[str, np.ndarray] = {}
        self.r = ResultsStore()
        self.e = ResultsStore()
        self.s = ResultsStore()
        self.settings: dict[str, Any] = {
            "more": "off",
            "linesize": 80,
            "seed": 12345,
            "varabbrev": "on",
            "maxiter": 300,
        }
        self._temp_counter = 0
        self._temp_files_stack: list[list[Path]] = [[]]
        self._temp_vars_stack: list[list[str]] = [[]]
        self.programs: dict[str, Any] = {}
        self.cwd: Path = Path.cwd()
        self.log_file: Any = None
        self.log_path: Path | None = None

        home_ado = Path.home() / ".openstata" / "ado"
        self.adopath: list[Path] = [
            Path("."),
            home_ado / "personal",
            home_ado / "plus",
        ]

    @property
    def locals(self) -> dict[str, str]:
        return self.local_stack[-1]

    def push_local_scope(self) -> None:
        self.local_stack.append({})
        self._temp_files_stack.append([])
        self._temp_vars_stack.append([])

    def pop_local_scope(self, dataset: Any = None) -> list[str]:
        self.local_stack.pop()
        for p in self._temp_files_stack.pop():
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass
        tvars = self._temp_vars_stack.pop()
        if dataset is not None:
            existing = [v for v in tvars if v in dataset.df.columns]
            if existing:
                dataset.drop_vars(existing)
        return tvars

    def new_tempvar(self) -> str:
        self._temp_counter += 1
        name = f"__000{self._temp_counter:04d}"
        self._temp_vars_stack[-1].append(name)
        return name

    def new_tempname(self) -> str:
        self._temp_counter += 1
        return f"__000{self._temp_counter:04d}"

    def new_tempfile(self) -> str:
        self._temp_counter += 1
        fd, path_str = tempfile.mkstemp(prefix="openstata_", suffix=".dta")
        os.close(fd)
        p = Path(path_str)
        if p.exists():
            p.unlink()
        self._temp_files_stack[-1].append(p)
        return str(p)

    def get_c_value(self, key: str, dataset: Any = None) -> Any:
        if key == "N":
            return dataset.nobs if dataset is not None else 0
        if key == "k":
            return dataset.nvars if dataset is not None else 0
        if key == "pwd":
            return str(self.cwd)
        if key == "os":
            return os.name
        if key == "pi":
            return float(np.pi)
        if key == "epsdouble":
            return float(np.finfo(np.float64).eps)
        if key == "version":
            return 17.0
        if key == "current_date":
            from datetime import datetime
            return datetime.now().strftime("%d %b %Y")
        if key == "current_time":
            from datetime import datetime
            return datetime.now().strftime("%H:%M:%S")
        if key == "adopath":
            return ";".join(str(p) for p in self.adopath)
        if key in self.settings:
            return self.settings[key]
        return None
