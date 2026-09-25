# Validation against R and Stata

Checks our JIVE / UJIVE estimators against independent implementations on identical data.

| Step | Command (from the repo root) | Needs |
|---|---|---|
| 1. Make the shared data | `python validation/make_data.py` | numpy, pandas |
| 2. R results | `Rscript validation/run_r.R` | R, [`kylebutts/jive`](https://github.com/kylebutts/jive) (setup notes at the top of the script) |
| 3. Stata results | `do validation/run_stata.do` in Stata | `ssc install sjive`, and Poi's `jive` from the Stata Journal |
| 4. Compare | `python validation/compare.py` | numpy, pandas |

`data/` (three small CSVs) and `results_r.csv` are committed so the comparison, and `tests/test_r_reference.py`,
run without R or Stata. `results_stata.csv` is added once the do-file has been run.

## What we found (coefficient on `t`, all three data sets)

| Comparison | Coefficient | Standard error |
|---|---|---|
| our `UJIVE1` vs R `jive()` | equal to ~1e-15 | equal to ~1e-15 |
| our `UJIVE1` vs Stata `jive, ujive1 robust` | equal to ~3e-8 | equal to ~3e-8 |
| our `UJIVE2` vs Stata `jive, ujive2 robust` | equal to ~4e-8 | equal to ~3e-8 |
| our `JIVE1` / `JIVE2` vs Stata `jive, jive1` / `jive2` | equal to ~3e-8 | **differ** (0.4-4%): Stata's `jive.ado` has a bug in this routine, see below |
| R `ujive()` (Kolesar 2013) | not implemented here; equals `P'Y / P'T` exactly | |
| Stata `sjive, noshrink` | not implemented here; equals an IV of leave-one-out-residualised `Y` on `T` (1e-8) | |

Other Stata numbers that match ours for `UJIVE1`: Root MSE, R-squared, overall (Wald) F, first-stage F(10, 289).
### Why the `JIVE1` / `JIVE2` standard errors differ from Stata's

Poi's `jive.ado` (Stata Journal package st0108, version 1.0.2, 17 Apr 2006) computes the residual for the variance with
`matrix colnames beta = end1 exog one` followed by `matrix score`. `CalcUJIVE` defines `one` (`tempvar one`, line 295), so
the constant column is named correctly. `CalcJIVE` (line 337) uses `` `one' `` but never defines it, so the name list is one
name short and the constant's coefficient ends up on the last regressor. The residual is then `Y - (b_t + b_cons) T` with no
controls, or `Y - b_t T - ... - (b_last + b_cons) W_last` with controls, instead of `Y - X b`. Its Root MSE, R-squared,
default and robust standard errors all follow from that residual: the rule reproduces all 12 `jive1`/`jive2` standard
errors (three data sets, default and robust) to 1e-8, and the Root MSE 3.5927 that Stata prints. The code's own comment
says the intent is `Y - X b` ("Use endogenous vars for residuals, not the predicted vars"), which is what we compute.
We do not copy Poi's source into this repository; download it with `net install st0108, from(http://www.stata-journal.com/software/sj6-3)`.

See the "Naming" section of the top-level README for why "UJIVE" means different things in Stata, R and the papers.

## R package caveat

`jive()` errors on a model with no fixed effects at all (inside `block_diag_hatvalues`, a `Matrix` class that no longer
exists). `run_r.R` works around this with a constant fixed effect `one`, which is equivalent to the intercept.

## Stata run notes

The Stata run (`results_stata.csv`, full log in `results_stata.log` with local directory names redacted) used Stata 19 MP, with `sjive` from SSC and Poi's
`jive` (Stata Journal package st0108). The option names in `run_stata.do` matched `jive`'s help file. `sjive` with
`noshrink` crashed on every data set because of a bug in the package (a matrix `QZt` is defined only in the shrinkage
branch but used either way); the run worked around it by defining that matrix unconditionally, which does not change
the method.
