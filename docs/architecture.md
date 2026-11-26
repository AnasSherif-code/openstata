# OpenStata Architecture

- `openstata.core`: `Dataset` (columnar `DataFrame` + `VariableMeta`, `ValueLabelTable`, uint8 extended missing-value codes `.`, `.a`–`.z`), `StateStore` (`local`/`global` macro scopes, `scalars`, `matrices`, `r()`/`e()`/`c()`, `tempvar`/`tempfile`), and `StataError`.
- `openstata.io`: Clean-room `.dta` 118/119 binary codec (`read_dta`, `write_dta`) and delimited text (`read_delimited`, `write_delimited`).
- `openstata.lang`: Script preprocessor (`*`, `//`, `/* */`, `///`, `#delimit ;`, blocks), iterative macro expansion, universal command parser (`[capture] [quietly] [by:] cmd ...`), and recursive-descent vectorized expression evaluator.
- `openstata.commands`: Built-in command implementations organized by domain (`system_cmds`, `io_cmds`, `data_cmds`, `stats_cmds`, `model_cmds`).
- `openstata.plugins`: `BaseCommand`, `CommandRegistry` (abbreviation matching + `openstata.commands` entry-point loader), and `pkg` installer.
- `openstata.cli`: Interactive REPL and batch `.do` executor.
