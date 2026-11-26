"""gencohort command – create cohort variable for panel data.

The command mirrors Stata's `gencohort` utility.  Syntax (simplified):
    gencohort, timevar(<time>) groupvar(<group>) [cohortvar(<newvar>)]
If ``cohortvar`` is omitted the variable ``cohort`` is created.
The command adds the new variable to the current dataset (in-place).
"""

from __future__ import annotations

from openstata.core.errors import StataError
from openstata.plugins.base import BaseCommand, CommandSpec, ExecutionContext
from openstata.lang.parser import ParsedCommand
from openstata.causal.utils import make_cohort


class GencohortCmd(BaseCommand):
    """Create a cohort identifier from a time and group variable.

    Options:
        timevar(name)   – name of the time variable (required)
        groupvar(name)  – name of the group identifier (required)
        cohortvar(name) – name of the output variable (default: "cohort")
    """

    # No abbreviation: "gen" belongs to "generate", so gencohort must be typed in full
    spec = CommandSpec("gencohort", len("gencohort"), "Generate cohort variable for panel data")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        ds = ctx.dataset
        opts = cmd.options
        # Required options
        timevar = opts.get("timevar")
        groupvar = opts.get("groupvar")
        if not timevar or not isinstance(timevar, str):
            raise StataError("option timevar() required", rc=198)
        if not groupvar or not isinstance(groupvar, str):
            raise StataError("option groupvar() required", rc=198)
        cohortvar = opts.get("cohortvar") or "cohort"
        if not isinstance(cohortvar, str):
            raise StataError("cohortvar() must be a string", rc=198)
        # Validate that the variables exist in the dataset
        ds.require_var(timevar)
        ds.require_var(groupvar)
        # Compute cohort series
        cohort_series = make_cohort(ds.df, timevar, groupvar, cohortvar=cohortvar)
        # Insert into dataset – replace if already present
        ds.df[cohortvar] = cohort_series
        # No output, just a confirmation message similar to Stata
        ctx.console.write(f"variable {cohortvar} generated")
