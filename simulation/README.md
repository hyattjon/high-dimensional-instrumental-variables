# Monte Carlo study

Does each estimator do what it is supposed to with many, possibly weak, instruments? The checks against R and Stata in
[`validation/`](../validation/) show that our estimators compute the same numbers as other software. They do not show that
those numbers are *good*. This study measures bias and confidence-interval coverage against a known truth.

**Status: written, not yet run.** The design and the hypotheses below were fixed before any results existed.

## Design

The data generating process (`simulate` in `monte_carlo.py`), for `n` observations, `k` instruments and `G` controls:

```
x = 0.5 + c * (z_1 + ... + z_k) + 0.5 * (w_1 + ... + w_G) + v
y = 1 + 1 * x                   + 0.5 * (w_1 + ... + w_G) + eps
```

`z` and `w` are standard normal; `(eps, v)` have unit variances and correlation `rho`; the true coefficient on `x` is 1.
All instruments have the same coefficient `c = sqrt(mu2 / (n k))`, so the **concentration parameter** is `mu2`, and the
expected first-stage F-statistic is about `1 + mu2 / k`. Holding `mu2` fixed while `k` grows is the classic
many-instruments design: the same total information is spread over more and more instruments, so 2SLS overfits.

| Axis | `full` grid | `quick` grid |
|---|---|---|
| sample size `n` | 500 | 200 |
| instruments `k` | 5, 20, 50, 100 | 5, 20 |
| concentration `mu2` | 25, 100, 300 | 25, 100 |
| error correlation `rho` | 0.5 | 0.5 |
| controls | 0, 2 | 0, 2 |
| heteroskedastic error | no (`--hetero no yes` adds yes) | no |

The observed first-stage F therefore ranges from about 1.25 (`k = 100`, `mu2 = 25`) to about 61 (`k = 5`, `mu2 = 300`).
All axes can be overridden on the command line; see `python simulation/monte_carlo.py --help`.

**Estimators:** OLS, 2SLS, `JIVE1`, `JIVE2`, `UJIVE1`, `UJIVE2`. OLS and 2SLS use the same robust (HC0) sandwich and
t interval as the jackknife estimators, so differences come from the point estimates.

**Reported per cell and estimator** (coefficient on `x`): median and mean bias, median absolute error (robust to heavy
tails, which the jackknife estimators can have), RMSE, coverage of the 95% interval, median interval width, the ratio of the
median reported standard error to the IQR-based standard deviation of the estimates, and how often the estimator raised an
error.

## Running it

```
python simulation/monte_carlo.py --dry-run                     # prints the design and the number of model fits; simulates nothing
python simulation/monte_carlo.py --grid quick --reps 50        # small trial run: checks the pipeline and times your machine
python simulation/monte_carlo.py --grid full --reps 2000 --workers 4
python simulation/report.py                                    # tables (report.md) and figures (needs matplotlib)
```

Runs are reproducible: replication `r` of cell `i` is seeded by `(seed, i, r)`, so results do not depend on the number of
workers. Finished cells are saved as they complete and skipped on re-running, so an interrupted run resumes. Output goes to
`simulation/results/` (git-ignored). We have not timed the study; run the `quick` grid first and scale up. The cost of a model
fit grows with `n (k + G)^2`, and the `full` grid is 24 cells x `--reps` x 6 estimators.

## What we expect to see (written before running)

These are predictions, several of them uncertain, and any of them can turn out wrong.

1. **OLS** is biased upward by about `rho / (1 + mu2 / n)` (roughly 0.3 to 0.5) in every cell, and its coverage is far below 95%. This is a check of the design, not a finding.
2. **2SLS** median bias grows with `k` and shrinks with `mu2`, roughly like `rho * k / (k + mu2)` (a Nagar-type approximation, for `k` above 2). It should be near OLS's bias in the weakest cells and small in the strongest. Its coverage should fall as bias grows relative to the standard error.
3. **`UJIVE1` and `UJIVE2`** (the Angrist-Imbens-Krueger JIVE in IV form) should have median bias near zero wherever the first stage is not hopelessly weak, at the price of larger spread than 2SLS (higher median absolute error in the weak cells) and occasional extreme estimates. Their intervals should cover close to 95% in the moderate and strong cells and under-cover in the weakest.
4. **`JIVE1` and `JIVE2`** (regress `Y` on the leave-one-out fitted values) are the open question. No paper we know of defines them. The leave-one-out fitted value is a noisy version of the true first-stage fit, and regressing on a noisy regressor biases the coefficient toward zero. So we expect a **downward** bias (below 1), growing with `k` at fixed `mu2`. If instead they behave like `UJIVE1`, the regression form is fine and we were wrong. Their standard errors are also untested.
5. **`UJIVE1` vs `UJIVE2`** should be close (they differ only by a different denominator, `1 - h` against `1 - 1/N`).
6. **Standard errors:** the ratio of reported to actual spread should be near 1 for the estimators with good coverage and well below 1 where coverage fails.
7. **Failures:** errors (a leverage of 1, and so on) should be rare with continuous instruments.

## Limits

One sample size, normal errors, equal-strength instruments, and no fixed effects, clustering or judge-style dummy instruments.
It does not include Kolesár's UJIVE, which we do not implement. Results are for the coefficient on the endogenous variable only.
