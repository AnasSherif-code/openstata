"""Tests for `import jasp` and `import excel`.

JASP (.jasp) files are ZIP archives whose main data table is a CSV (typically
`data/database.csv`) plus a `meta.json`; the archive fixture below mirrors the
real layout. Excel fixtures are written with openpyxl.
"""

import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd

from openstata.cli.main import create_runner
from openstata.core.dataset import Dataset
from openstata.core.errors import StataError


class RecordingConsole:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def write(self, text: str = "") -> None:
        self.lines.append(text)

    def write_error(self, text: str) -> None:
        self.lines.append(text)


def make_jasp(tmp: Path, rows_csv: str, *, member: str = "data/database.csv", with_meta: bool = True) -> Path:
    p = tmp / "sample.jasp"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr(member, rows_csv)
        if with_meta:
            zf.writestr("meta.json", '{"name": "Test JASP dataset"}')
        zf.writestr("results/some-analysis.json", "{}")
    return p


class TestImportJasp(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tmp_path = Path(self.tmp.name)
        self.dataset = Dataset()
        self.console = RecordingConsole()
        self.runner = create_runner()
        self.runner.dataset = self.dataset
        self.runner.console = self.console

    def test_import_jasp_standard_layout(self):
        p = make_jasp(
            self.tmp_path,
            '...,age,income\n1,34,55000\n2,41,60000\n',
        )
        self.runner.run_source(f'import jasp "{p}"')
        # OpenStata stores all numerics as float64 (Stata-style), so compare
        # values only
        pd.testing.assert_frame_equal(
            self.dataset.df[["age", "income"]].reset_index(drop=True),
            pd.DataFrame({"age": [34, 41], "income": [55000, 60000]}),
            check_dtype=False,
        )
        self.assertEqual(self.dataset.data_label, "Test JASP dataset")

    def test_import_jasp_database_at_root(self):
        p = make_jasp(
            self.tmp_path,
            "x,y\n1,2\n3,4\n",
            member="database.csv",
        )
        self.runner.run_source(f'import jasp "{p}"')
        self.assertEqual(list(self.dataset.df.columns), ["x", "y"])
        self.assertEqual(len(self.dataset.df), 2)

    def test_import_jasp_picks_largest_csv(self):
        p = self.tmp_path / "multi.jasp"
        with zipfile.ZipFile(p, "w") as zf:
            zf.writestr("data/small.csv", "a\n1\n")
            zf.writestr("data/database.csv", "a,b\n1,2\n3,4\n")
        self.runner.run_source(f'import jasp "{p}"')
        self.assertEqual(list(self.dataset.df.columns), ["a", "b"])

    def test_import_jasp_without_index_column(self):
        # Real JASP CSVs always carry the index; plain CSVs must load untouched
        p = make_jasp(self.tmp_path, "a,b\n5,6\n7,8\n")
        self.runner.run_source(f'import jasp "{p}"')
        self.assertEqual(list(self.dataset.df.columns), ["a", "b"])
        self.assertEqual(self.dataset.df["a"].tolist(), [5, 7])

    def test_import_jasp_invalid_file(self):
        p = self.tmp_path / "notajasp.txt"
        p.write_text("hello", encoding="utf-8")
        with self.assertRaises(StataError):
            self.runner.run_source(f'import jasp "{p}"')

    def test_import_jasp_zip_without_csv(self):
        p = self.tmp_path / "empty.jasp"
        with zipfile.ZipFile(p, "w") as zf:
            zf.writestr("meta.json", "{}")
        with self.assertRaises(StataError):
            self.runner.run_source(f'import jasp "{p}"')


class TestImportExcel(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tmp_path = Path(self.tmp.name)
        self.dataset = Dataset()
        self.console = RecordingConsole()
        self.runner = create_runner()
        self.runner.dataset = self.dataset
        self.runner.console = self.console

    def _make_xlsx(self, name: str = "book.xlsx") -> Path:
        p = self.tmp_path / name
        df = pd.DataFrame({"age": [34, 41], "income": [55000, 60000]})
        with pd.ExcelWriter(p, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="main", index=False)
            df.rename(columns={"age": "score"}).to_excel(writer, sheet_name="other", index=False)
        return p

    def test_import_excel_default_first_sheet(self):
        p = self._make_xlsx()
        self.runner.run_source(f'import excel "{p}"')
        pd.testing.assert_frame_equal(
            self.dataset.df[["age", "income"]].reset_index(drop=True),
            pd.DataFrame({"age": [34, 41], "income": [55000, 60000]}),
            check_dtype=False,
        )

    def test_import_excel_sheet_option(self):
        p = self._make_xlsx()
        self.runner.run_source(f'import excel "{p}", sheet(other)')
        self.assertEqual(list(self.dataset.df.columns), ["score", "income"])

    def test_import_excel_sheet_not_found(self):
        p = self._make_xlsx()
        with self.assertRaises(StataError):
            self.runner.run_source(f'import excel "{p}", sheet(nosuch)')
        # Dataset in memory must be untouched after a failed import
        self.assertEqual(self.dataset.nobs, 0)

    def test_import_excel_rejects_non_excel(self):
        p = self.tmp_path / "notes.txt"
        p.write_text("hi", encoding="utf-8")
        with self.assertRaises(StataError):
            self.runner.run_source(f'import excel "{p}"')

    def test_import_excel_alias_xlsx(self):
        p = self._make_xlsx("book2.xlsx")
        self.runner.run_source(f'import xlsx "{p}"')
        self.assertEqual(list(self.dataset.df.columns), ["age", "income"])


if __name__ == "__main__":
    unittest.main()
