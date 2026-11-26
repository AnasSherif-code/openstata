"""Tests for third-party .ado packages, Python plugin registration, and CLI batch runner."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from openstata.cli.main import create_runner, main
from openstata.core.dataset import Dataset


class TestPluginsAndCLI(unittest.TestCase):
    def test_ado_package_install_and_execution(self) -> None:
        runner = create_runner()
        ado_dir = Path(__file__).resolve().parents[1] / "examples" / "sample_pkg_ado"
        runner.run_source(f'adopath + "{ado_dir.as_posix()}"')

        vals = list(range(1, 21))
        runner.dataset = Dataset(pd.DataFrame({"x": vals}))
        runner.run_source(
            """
            winsorize_simple x, generate(xw) low(5) high(95)
            assert xw >= r(lower)
            assert xw <= r(upper)
            """
        )
        self.assertIn("xw", runner.dataset.columns)
        self.assertLess(float(runner.dataset.df["xw"].max()), 20.0)

    def test_python_plugin_command_api(self) -> None:
        # Import example plugin directly and register via CommandRegistry
        ext_src = Path(__file__).resolve().parents[1] / "examples" / "sample_plugin_py" / "src"
        sys.path.insert(0, str(ext_src))
        try:
            from sample_openstata_ext import ZscoreCmd

            runner = create_runner()
            runner.registry.register(ZscoreCmd())
            runner.dataset = Dataset(pd.DataFrame({"x": [10.0, 20.0, 30.0, 40.0]}))
            runner.run_source("zscore x, generate(zx)")
            zx = runner.dataset.df["zx"].to_numpy()
            self.assertAlmostEqual(float(np.mean(zx)), 0.0, places=10)
            self.assertAlmostEqual(float(np.std(zx, ddof=1)), 1.0, places=10)
        finally:
            sys.path.remove(str(ext_src))

    def test_cli_execute_and_do_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            do_path = Path(tmp) / "job.do"
            out_dta = (Path(tmp) / "out.dta").as_posix()
            do_path.write_text(
                f"""
                clear
                set obs 4
                gen a = _n * 2
                save "{out_dta}", replace
                """,
                encoding="utf-8",
            )
            rc = main([str(do_path)])
            self.assertEqual(rc, 0)
            self.assertTrue(Path(out_dta).exists())


if __name__ == "__main__":
    unittest.main()
