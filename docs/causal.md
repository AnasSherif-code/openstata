# Causal inference commands

OpenStata provides a small set of causal-inference commands in the
`openstata.causal` package. They are registered as built-in commands and can be
used both interactively and in `.do` scripts.

## `didregress`

Basic two-way Difference-in-Differences estimator.

### Syntax

```stata
didregress depvar, treat(treatvar) time(timevar) [robust] [cluster(clustvar)]
```

- `depvar` — outcome variable (first and only positional argument).
- `treat(treatvar)` — binary treatment indicator (required). The option may be
  abbreviated `treat()` only; use the full option name.
- `time(timevar)` — binary post-treatment period indicator (required).
- `robust` / `cluster(clustvar)` — currently parsed but **not yet applied**;
  ordinary OLS standard errors are reported.

### Estimation

The command fits OLS of `depvar` on a constant, the treatment indicator, the
time indicator, and their interaction. The coefficient on the interaction term
(`treatvar#c.timevar`) is the DID estimate. Coefficients, standard errors, *t*
statistics, *p*-values, and 95% confidence intervals are printed in the same
table format as `regress`.

### Example

```stata
* Deterministic dataset with DID effect = 2
input time treated y
0 0 5
1 0 7
0 1 8
1 1 12
end

didregress y, treat(treated) time(time)
```

Expected output (abridged):

```
------------------------------------------------------------------------------
           y | Coefficient   Std. err.         t    P>|t|  [95% conf. interval]
------------------------------------------------------------------------------
       _cons |     5.00000  ...
     treated |     3.00000  ...
        time |     2.00000  ...
treated#c.time |   2.00000  ...
------------------------------------------------------------------------------
```

The interaction coefficient `2.0` recovers the true DID effect.

### Limitations

This is a minimal implementation. Not yet supported:

- robust (HC1) and cluster-robust standard errors (the options are accepted
  but silently ignored),
- weighting via `[weight=...]`,
- factor-variable expansions in `treat()`/`time()`,
- the `over()` suboption and event-study / heterogeneous-treatment designs.

See `tests/test_causal/test_didregress.py` for executable examples.

## `gencohort`

Create a cohort identifier from a time and a group variable, mirroring
Stata's `gencohort` utility.

### Syntax

```stata
gencohort, timevar(timevar) groupvar(groupvar) [cohortvar(newvar)]
```

- `timevar(name)` — time variable (required).
- `groupvar(name)` — group identifier (required).
- `cohortvar(name)` — name of the generated variable (default: `cohort`).

`gencohort` must be spelled out in full: `gen` resolves to `generate`.

### Behavior

For each group, the cohort value is the group's first (minimum) time period.
The variable is added to the dataset in memory, replacing any existing
variable of the same name.

### Example

```stata
input year id y
2000 1 10
2001 1 12
2002 1 14
2000 2 20
2001 2 22
2002 2 24
end

gencohort, timevar(year) groupvar(id)
* -> variable cohort generated; cohort = 2000 for all observations
```

See `tests/test_causal/test_gencohort.py` for executable examples.
