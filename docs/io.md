# Data import & export

OpenStata reads and writes several file formats. All importers load data into
the in-memory dataset and follow Stata's storage conventions (all numeric
variables are stored as doubles, `_sync_metadata` assigns variable metadata
automatically).

## Supported formats

| Command | Formats | Notes |
|---------|---------|-------|
| `use` / `save` | Stata `.dta` (118/119 + legacy) | Native format, keeps labels & missing codes |
| `import delimited` / `export delimited` | CSV, TSV, other delimited text | `delimiters()` option |
| `import excel` | `.xlsx`, `.xlsm`, `.xlsb`, `.xls`, `.ods` | Requires `openpyxl` (installed with the package) |
| `import jasp` | `.jasp` | JASP projects (ZIP archives containing a CSV data table) |
| `input` | inline data entry | Declared typed columns |

## `import excel`

```stata
import excel "data/survey.xlsx" [, sheet(sheetname)]
import xlsx "data/survey.xlsx"          % xlsx/xls are accepted aliases
```

- The **first row of the sheet is used as variable names** by default.
- `sheet(sheetname)` selects a worksheet by name or (numeric) position;
  the first sheet is used by default.
- Column names are sanitized to Stata-style names (spaces become `_`).
- A failed import (bad sheet, wrong format) leaves the dataset in memory
  untouched.

Example:

```stata
import excel "survey.xlsx", sheet(responses)
summarize
```

## `import jasp`

A `.jasp` file is a ZIP archive produced by the JASP statistics program. The
main data table lives inside the archive as a CSV (typically
`data/database.csv`), alongside a `meta.json`.

```stata
import jasp "analysis.jasp"
describe
```

Behavior:

- Data is read from the archive's data CSV. If several CSVs are present, the
  canonical `database.csv` (largest) is used.
- JASP's leading row-index column (`...` / empty header, a plain 1..n or
  0..n-1 counter) is dropped automatically. Genuine data columns are never
  dropped, even when they contain a 1..n sequence.
- The archive's `meta.json` `name` field, if present, becomes the dataset
  label (`describe` shows it).
- Variable names are sanitized like other text imports.

Both importers can be combined with `save` to convert a file to native
`.dta`:

```stata
import jasp "analysis.jasp"
save "analysis.dta", replace
```

## Errors

| Situation | Message |
|-----------|---------|
| Missing file | `file ... not found (r601)` |
| Not a ZIP / no data CSV in a `.jasp` | `is not a valid .jasp archive` / `no data table found` |
| Unknown Excel sheet | `failed to read Excel sheet: ...` |
| Unsaved changes in memory | `no; dataset in memory has changed since last saved (r4)` — pass `clear` to discard |
