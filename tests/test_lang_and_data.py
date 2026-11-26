"""Tests for Stata language semantics, macros, control flow, and data management commands."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from openstata.cli.main import create_runner
from openstata.core.errors import StataError


class TestLangAndData(unittest.TestCase):
    def test_missing_value_ordering_and_comparisons(self) -> None:
        runner = create_runner()
        runner.run_source(
            """
            input x
            10
            100000
            .
            .a
            .b
            end
            gen gt_big = (x > 100)
            gen is_a = (x == .a)
            gen a_lt_b = (.a < .b)
            assert gt_big == 1 in 2/5
            assert gt_big == 0 in 1
            assert is_a == 1 in 4
            assert a_lt_b == 1
            """
        )
        self.assertEqual(runner.dataset.nobs, 5)

    def test_macros_loops_delimit_and_continuations(self) -> None:
        runner = create_runner()
        runner.run_source(
            """
            clear
            set obs 5
            gen x = _n
            local sum 0
            forvalues i = 1/5 {
                local sum = `sum' + `i'
            }
            assert `sum' == 15

            * Line continuation and inline expression macro
            gen y = x * 2 ///
                + 1
            di "result=`=2+3'"
            assert y == 11 in 5

            #delimit ;
            local total 0;
            foreach v of varlist x y {;
                qui summarize `v';
                local total = `total' + `r(sum)';
            };
            #delimit cr
            assert `total' == 50
            """
        )

    def test_bysort_egen_collapse_merge_reshape(self) -> None:
        runner = create_runner()
        with tempfile.TemporaryDirectory() as tmp:
            lookup_path = (Path(tmp) / "lookup.dta").as_posix()
            runner.run_source(
                f"""
                clear
                input id str4 grp val
                1 A 10
                1 A 20
                2 B 30
                end
                save "{lookup_path}", replace

                clear
                input id str4 grp
                1 A
                2 B
                3 C
                end
                merge 1:m id grp using "{lookup_path}"
                count if _merge == 3
                assert r(N) == 3
                count if _merge == 1
                assert r(N) == 1

                drop if missing(val)
                bysort grp: egen mval = mean(val)
                assert mval == 15 if grp == "A"

                collapse (mean) mean_val=val (sum) sum_val=val, by(grp)
                sort grp
                assert mean_val == 15 in 1
                assert sum_val == 30 in 1
                assert mean_val == 30 in 2
                """
            )

            # Reshape wide -> long -> wide
            runner.run_source(
                """
                clear
                input id inc2020 inc2021
                1 100 120
                2 200 250
                end
                reshape long inc, i(id) j(year)
                assert _N == 4
                reshape wide inc, i(id) j(year)
                assert _N == 2
                assert inc2021 == 250 if id == 2
                """
            )


if __name__ == "__main__":
    unittest.main()
