# Extended estimation commands

This document covers the commands added in the "extended estimation" wave:
effect sizes, linear combinations of coefficients, the Bayesian bootstrap,
the `mean`/`proportion`/`ratio`/`total` family with `over()` and
`dofsubpop`, absorbed fixed effects (`areg`), instrumental-variables 2SLS
with weak-instrument diagnostics, and control-function estimators.

All commands are also usable with `by:`-prefixes and support the standard
`if`/`in` qualifiers unless noted.

## esize twosample

Effect sizes for two independent samples.

**Syntax**

```
esize twosample varname [if] [in], by(groupvar) [effect(cohend) reverse level(#)]
esize twosample varname1 == varname2 [if] [in] [, effect(...) reverse level(#)]
```

**Options**

| Option | Description |
|--------|-------------|
| `by(groupvar)` | Compare two groups defined by `groupvar` (must yield exactly 2 groups). |
| `varname1 == varname2` | Compare two independent variables. |
| `effect()` | `cohend` (default), `hedgesg`, or `glassdelta`. |
| `reverse` | Reverse the order of the groups (flips the sign of the estimate). |
| `level(#)` | Confidence level, default 95. |

**Example**

```
sysuse-like data with y and g
esize twosample y, by(g) effect(cohend)
```

Output reports the effect-size estimate, its z statistic, standard error,
and confidence interval (normal reference).

## lincom

Point estimate, standard error, t statistic, p-value and confidence
interval for a linear combination of the last estimation command's
coefficients.

**Syntax**

```
lincom exp [ , level(#) coef ]
```

`exp` may use coefficient names (including `_cons`) with `+ - * /` and
parentheses, e.g. `lincom 2*x1 + 3*x2 - 1`.

**Options**

| Option | Description |
|--------|-------------|
| `coef` | After `logit`, report the linear combination itself instead of exponentiating to an odds ratio. |
| `level(#)` | Confidence level, default 95. |

**Example**

```
regress y x1 x2
lincom 2*x1 - x2 + _cons
```

The standard error is `sqrt(l V l')` with `V` the stored variance-covariance
matrix. After `logit`, results are displayed as odds ratios unless `coef`
is specified.

## bayesboot

Bayesian (Dirichlet-weight) bootstrap for linear regression.

**Syntax**

```
bayesboot depvar indepvars [if] [in] [, reps(#) level(#) seed(#)]
```

Each replication resamples the observations with Dirichlet(1) weights,
re-estimates OLS, and the final table reports the posterior mean, standard
deviation, the observed (point) estimate, and a percentile interval per
coefficient.

| Option | Default | Description |
|--------|---------|-------------|
| `reps()` | 400 | Number of bootstrap replications. |
| `level()` | 95 | Percentile-interval level. |
| `seed()` | none | Seed for reproducibility. |

**Example**

```
bayesboot y x1 x2, reps(1000) seed(123)
```

## mean, proportion, ratio, total

Estimation of means, proportions, ratios of means and totals, each with
optional subgroup statistics.

**Syntax**

```
mean varlist [if] [in] [, over(varname) dofsubpop level(#)]
proportion varname [if] [in] [, over(varname) dofsubpop level(#)]
ratio numvar/denvar [if] [in] [, over(varname) dofsubpop level(#)]
total varlist [if] [in] [, over(varname) dofsubpop level(#)]
```

**Behavior**

- `proportion` of a 0/1 variable reports a single proportion with the
  binomial standard error `sqrt(p(1-p)/n)`.
- `proportion` of a categorical variable reports one row per observed value
  (`cat @ 2`), each with the binomial standard error.
- `ratio` accepts both `ratio y/x` and `ratio y / x`; the standard error is
  the delta-method SE for a ratio of means.
- `over(varname)` produces one row per variable per subgroup.
- `dofsubpop` restricts the degrees-of-freedom calculation to observations
  within each `over()` subpopulation (default: whole sample minus one).

**Example**

```
mean x1 y, over(group)
proportion cat
ratio income/consumption, over(region) dofsubpop
```

## areg

Linear regression with absorbed (high-dimensional) fixed effects via
within-transformation, equivalent to adding dummy variables for the
absorbed groups.

**Syntax**

```
areg depvar indepvars [if] [in], absorb(groupvar) [dfabsorb robust cluster(varname) level(#)]
```

**Options**

| Option | Description |
|--------|-------------|
| `absorb(groupvar)` | Variable whose groups define fixed effects (required). |
| `dfabsorb` | Count degrees of freedom for the absorbed effects using the rank of the one-hot matrix (adjusts for pairwise collinearity among absorbed categories). |
| `robust`, `cluster()` | Heteroskedasticity- or cluster-robust standard errors. |
| `level(#)` | Confidence level, default 95. |

**Notes**

- The coefficient estimates match an explicit dummy-variable (LSDV)
  regression; the intercept is reported up to the absorbed-effects
  normalization and is not comparable to an LSDV `_cons`.
- `e(df_a)` holds the absorbed degrees of freedom, `e(df_r)` the residual
  degrees of freedom (`n - k - df_a` by default).

**Example**

```
areg wage educ, absorb(state_id) cluster(state_id)
```

## ivregress 2sls

Instrumental-variables two-stage least squares with first-stage
diagnostics.

**Syntax**

```
ivregress 2sls depvar [exogvars] (endogvars = instvars) [if] [in] [, robust cluster(varname) level(#)]
```

Both orderings are accepted: `ivregress 2sls y (x_endog = z) x1` and
`ivregress 2sls y x1 (x_endog = z)` are equivalent.

**Options**

| Option | Description |
|--------|-------------|
| `robust` | Heteroskedasticity-robust (GMM sandwich) standard errors. |
| `cluster(varname)` | Cluster-robust standard errors. |
| `level(#)` | Confidence level, default 95. |

**Weak-instrument diagnostics**

- **First-stage F**: for each endogenous variable, the F statistic for the
  joint significance of the excluded instruments. The note `(weak: F < 10)`
  is appended when the Staiger–Stock rule of thumb triggers.
- **Anderson–Rubin test**: reduced-form F test of the endogenous variable's
  effect being zero — robust to weak instruments and reported regardless of
  first-stage strength.

**Example**

```
ivregress 2sls y (x_endog = z_inst) x1, robust
```

## cfregress and cfprobit

Control-function estimators: a two-step procedure that augments the main
regression with first-stage residuals.

**Syntax**

```
cfregress depvar [exogvars] (endogvars = instvars) [if] [in] [, level(#)]
cfprobit  depvar [exogvars] (endogvars = instvars) [if] [in] [, level(#)]
```

**Notes and limitations**

- For the **linear** model the point estimates are algebraically identical
  to `ivregress 2sls` (verified in the test suite); standard errors do not
  account for first-stage estimation and a NOTE is printed.
- For **probit**, the two-step control-function approach is an
  approximation: unlike the linear case, the second-stage coefficients are
  not guaranteed to equal the structural probit parameters. The standard
  errors are naive. For fully consistent probit estimation with endogenous
  regressors, full-information maximum likelihood (Stata's `ivprobit`) is a
  future extension.

**Example**

```
cfregress y (x_endog = z_inst) x1
cfprobit d (x_endog = z_inst)
```
