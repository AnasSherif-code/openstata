# Authoring Third-Party Packages & Plugins for OpenStata

OpenStata supports two extension mechanisms so researchers and contributors can add new statistical estimators, data tools, or domain workflows without modifying core code.

---

## Option 1: `.ado` User Packages (Stata Script Syntax)

Place `<command_name>.ado` files in a directory on `adopath` (e.g., `~/.openstata/ado/personal`, `~/.openstata/ado/plus`, or add a directory via `adopath + /path/to/pkg`).

Example (`examples/sample_pkg_ado/winsorize_simple.ado`):
```stata
program define winsorize_simple
    syntax varlist(max=1) [if] [in] [, Cuts(real 5) GENerate(string)]
    marksample touse
    quietly summarize `varlist' if `touse', detail
    ...
    return scalar p_low = `lo'
end
```

Install a local folder containing `.ado` files using:
```stata
pkg install "examples/sample_pkg_ado"
```

---

## Option 2: Python Extension Packages (Entry Points)

1. Subclass `openstata.plugins.BaseCommand`:

```python
from openstata.plugins import BaseCommand, CommandSpec, ExecutionContext
from openstata.lang import ParsedCommand, expand_varlist

class ZscoreCmd(BaseCommand):
    spec = CommandSpec(name="zscore", min_abbrev=4, summary="Standardize variables")

    def run(self, cmd: ParsedCommand, ctx: ExecutionContext) -> None:
        ...
```

2. Expose it in your package's `pyproject.toml`:

```toml
[project.entry-points."openstata.commands"]
zscore = "my_pkg:ZscoreCmd"
```

3. Install your package (`pip install -e ./my_pkg` or `pkg install ./my_pkg`). OpenStata automatically discovers it at startup and lists it in `help` and `pkg list`.
