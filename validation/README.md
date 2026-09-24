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

## What we found (R, coefficient on `t`, all three data sets)

- **Our `UJIVE1` equals R's `jive()`** in both coefficient and standard error, to at least six decimals.
  R's `jive()` is the Angrist, Imbens, Krueger (1999) JIVE1 with controls partialled out.
- R's `ujive()` (Kolesar 2013) matches none of our estimators. It is not implemented here.
- Our `JIVE1`, `JIVE2`, `UJIVE2` have no R counterpart (the R package has no JIVE2).

See the "Naming" section of the top-level README for why "UJIVE" means different things in Stata, R and the papers.

## R package caveat

`jive()` errors on a model with no fixed effects at all (inside `block_diag_hatvalues`, a `Matrix` class that no longer
exists). `run_r.R` works around this with a constant fixed effect `one`, which is equivalent to the intercept.
