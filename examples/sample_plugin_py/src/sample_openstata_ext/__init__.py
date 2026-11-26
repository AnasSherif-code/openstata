"""Example third-party Python command plugin for OpenStata."""

import numpy as np

from openstata.core.errors import StataError
from openstata.lang.executor import compute_sample_mask
from openstata.lang.parser import ParsedCommand, expand_varlist
from openstata.plugins.base import BaseCommand, CommandSpec, ExecutionContext


class ZscoreCmd(BaseCommand):
    """Standardize a numeric variable: `zscore var, generate(newvar)`."""

    spec = CommandSpec(name="zscore", min_abbrev=4, summary="Create standardized z-score variable")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        ds = ctx.dataset
        vlist = expand_varlist(cmd.main_args, ds)
        if len(vlist) != 1:
            raise StataError("zscore requires exactly one variable", rc=103)
        var = vlist[0]
        gen = cmd.options.get("generate") or cmd.options.get("gen")
        newvar = str(gen) if isinstance(gen, str) and gen else f"z_{var}"

        mask = compute_sample_mask(cmd, ds, ctx.state)
        vals, mc = ds.get_numeric(var)
        valid = mask & (mc == 0)
        x = vals[valid]
        if len(x) < 2:
            raise StataError("insufficient observations", rc=2001)
        mu = float(np.mean(x))
        sd = float(np.std(x, ddof=1))
        out = np.where(valid, (vals - mu) / sd if sd > 0 else 0.0, np.nan)
        ds.add_numeric_var(newvar, out, stata_type="double")
        ctx.state.r.clear()
        ctx.state.r.scalars["mean"] = mu
        ctx.state.r.scalars["sd"] = sd
        ctx.console.write(f"Generated {newvar} (mean={mu:g}, sd={sd:g})")
