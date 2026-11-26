# OpenStata

A modular, clean-room Python reimplementation of core statistical, data-management, and scripting features inspired by Stata syntax, built for **academic study, research, and collaborative extension**.

> **License:** PolyForm Noncommercial 1.0.0 (study, research, and non-commercial use only).  
> **Clean-room notice:** OpenStata is written from scratch in Python using publicly documented file-format and statistical specifications.

## Quickstart

```bash
pip install -e .
openstata                   # Start interactive REPL
openstata analysis.do       # Run a .do script
```

## Extending OpenStata

Contributors and third-party authors can add commands in two ways:
1. **`.ado` packages**: Place `<cmd>.ado` files in your `adopath` (`~/.openstata/ado/personal` or `~/.openstata/ado/plus`) using `program define` and `syntax`.
2. **Python extension packages**: Publish a standard Python package exposing commands via the `[project.entry-points."openstata.commands"]` entry-point group (see `docs/plugin_authoring.md` and `examples/`).
