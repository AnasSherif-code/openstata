"""Tests for clean-room .dta 118 reader/writer and pandas interoperability."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from openstata.core.dataset import Dataset, ValueLabelTable, VariableMeta
from openstata.core.missing import MISSING_DOT, LABEL_TO_CODE
from openstata.io import read_dta, write_dta, read_delimited, write_delimited


class TestDtaIO(unittest.TestCase):
    def test_dta_118_roundtrip_with_extended_missing_and_labels(self) -> None:
        df = pd.DataFrame(
            {
                "id": [1.0, 2.0, 3.0, 4.0],
                "score": [10.5, np.nan, np.nan, 42.25],
                "name": ["alpha", "beta", "gamma", "delta"],
                "notes": ["short", "x" * 2500, "", "end"],
            }
        )
        mc_score = np.array([0, MISSING_DOT, LABEL_TO_CODE[".a"], 0], dtype=np.uint8)
        meta = {
            "id": VariableMeta("id", "long", "%12.0g", "Identifier", "id_lbl"),
            "score": VariableMeta("score", "double", "%9.2f", "Test score", ""),
            "name": VariableMeta("name", "str10", "%10s", "Short label", ""),
            "notes": VariableMeta("notes", "strL", "%9s", "Long notes", ""),
        }
        vlabels = {"id_lbl": ValueLabelTable("id_lbl", {1: "First", 2: "Second"})}
        ds = Dataset(
            df=df,
            meta=meta,
            missing_codes={"id": np.zeros(4, dtype=np.uint8), "score": mc_score},
            value_labels=vlabels,
            data_label="OpenStata unit test dataset",
            characteristics={"_dta": {"note1": "clean-room test"}},
            sort_by=["id"],
        )

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.dta"
            write_dta(ds, p, replace=True)

            # Read back with OpenStata
            loaded = read_dta(p)
            self.assertEqual(loaded.nobs, 4)
            self.assertEqual(loaded.nvars, 4)
            self.assertEqual(loaded.data_label, "OpenStata unit test dataset")
            self.assertEqual(loaded.sort_by, ["id"])
            self.assertEqual(loaded.meta["score"].label, "Test score")
            self.assertEqual(loaded.value_labels["id_lbl"].mapping[1], "First")
            self.assertEqual(loaded.characteristics["_dta"]["note1"], "clean-room test")
            self.assertEqual(loaded.df["notes"].iloc[1], "x" * 2500)
            self.assertEqual(int(loaded.missing_codes["score"][1]), MISSING_DOT)
            self.assertEqual(int(loaded.missing_codes["score"][2]), LABEL_TO_CODE[".a"])
            self.assertAlmostEqual(float(loaded.df["score"].iloc[0]), 10.5)

            # Verify interoperability with pandas.read_stata
            pdf = pd.read_stata(p, convert_categoricals=False)
            self.assertEqual(list(pdf.columns), ["id", "score", "name", "notes"])
            self.assertEqual(pdf["notes"].iloc[1], "x" * 2500)
            self.assertTrue(pd.isna(pdf["score"].iloc[1]))
            self.assertTrue(pd.isna(pdf["score"].iloc[2]))

    def test_delimited_roundtrip(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["u", "v", "w"]})
        ds = Dataset(df=df)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "data.csv"
            write_delimited(ds, csv_path, replace=True)
            ds2 = read_delimited(csv_path)
            self.assertEqual(ds2.nobs, 3)
            self.assertEqual(ds2.columns, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
