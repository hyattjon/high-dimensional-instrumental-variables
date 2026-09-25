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
| our `JIVE1` / `JIVE2` vs Stata `jive, jive1` / `jive2` | equal to ~3e-8 | **differ** (0.4-4% vs Stata `robust`) |
| R `ujive()` (Kolesar 2013) | not implemented here; equals `P'Y / P'T` exactly | |
| Stata `sjive, noshrink` | not implemented here; equals an IV of leave-one-out-residualised `Y` on `T` (1e-8) | |

Other Stata numbers that match ours for `UJIVE1`: Root MSE, R-squared, overall (Wald) F, first-stage F(10, 289).
The Stata `jive1`/`jive2` Root MSE and R-squared match none of the residual definitions we tried, and its standard
errors for that pair are not reproduced; that would need Poi's `jive.ado`.

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
