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
- `treat(treatvar)` — binary treatment indicator (required).
- `time(timevar)` — binary post-treatment period indicator (required).
- `robust` — HC1 (heteroskedasticity-robust) standard errors.
- `cluster(clustvar)` — cluster-robust standard errors by `clustvar`
  (requires at least 2 clusters).

### Estimation

The command fits OLS of `depvar` on a constant, the treatment indicator, the
time indicator, and their interaction. The coefficient on the interaction term
(`treatvar#c.timevar`) is the DID estimate. Coefficients, standard errors, *t*
statistics, *p*-values, and 95% confidence intervals are printed in the same
table format as `regress`.

Variance estimation mirrors `regress`:

- default: `V = s2 (X'X)^-1` with `s2 = RSS/(n-k)`; residual df = `n-k`.
- `robust`: HC1 sandwich, `V = (n/(n-k)) (X'X)^-1 X' diag(e^2) X (X'X)^-1`.
- `cluster(var)`: cluster-robust sandwich over the `G` clusters with the
  Stata small-sample correction `(G/(G-1)) ((n-1)/(n-k))`; residual df = `G-1`.

The fitted model is also stored in `e()` (`b`, `V`, and the `N`, `df_r`,
`df_m` scalars) for post-estimation use.

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
Difference-in-Differences regression (OLS)  Number of obs = 8
------------------------------------------------------------------------------
           y | Coefficient   Std. err.         t    P>|t|  [95% conf. interval]
------------------------------------------------------------------------------
       _cons |     5.50000  ...
     treated |     3.00000  ...
        time |     2.50000  ...
treated#c.time |     2.00000  ...
------------------------------------------------------------------------------
```

The interaction coefficient `2` recovers the true DID effect. With
`, robust` the standard errors become the HC1 sandwich values, and with
`, cluster(id)` the cluster-robust values with a note
`(Std. err. adjusted for N clusters in id)`.

### Limitations

This is a minimal implementation. Not yet supported:

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
