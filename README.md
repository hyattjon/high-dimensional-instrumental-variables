# high-dimensional-instrumental-variables
Jackknife IV estimators for instrumental variables regression with many instruments.

Two stage least squares is biased when there are many instruments, and the problem gets worse when the instruments are weak. This package implements four jackknife instrumental variable estimators that reduce that bias: **JIVE1**, **JIVE2**, **UJIVE1** and **UJIVE2**. They are written in plain NumPy / SciPy, never form the N x N projection matrix, and return heteroskedasticity-robust standard errors, t-tests and confidence intervals.

## Installation
```
git clone https://github.com/hyattjon/high-dimensional-instrumental-variables.git
cd high-dimensional-instrumental-variables
pip install .        # needs numpy, scipy and pandas
```

The package is named `high-dimensional-instrumental-variables` but is imported as `high_dimensional_instrumental_variables` (Python module names can't contain hyphens):

```python
from high_dimensional_instrumental_variables.jive1 import JIVE1
```

## Jackknife IV estimators: JIVE1, JIVE2, UJIVE1, UJIVE2

These four estimators reduce the many-instruments bias of 2SLS. Each one replaces the usual first-stage fitted value $\hat{x}_i = Z_i\hat{\pi}$ with a version that leaves observation $i$ out of its own first stage. The names follow the convention of Stata's `jive` command (Poi 2006), in which `ujive1` / `ujive2` are the estimators of Angrist, Imbens, and Krueger (1999). **That differs from the R package and from Kolesár (2013), where "UJIVE" is a different estimator** - see [Naming](#naming-ujive-means-different-things-in-different-software) below.

### Usage

```python
import numpy as np
from high_dimensional_instrumental_variables.jive1 import JIVE1      # also: jive2.JIVE2, ujive1.UJIVE1, ujive2.UJIVE2

rng = np.random.default_rng(0)
N, K = 500, 20
Z = rng.normal(size=(N, K))                    # instruments
u = rng.normal(size=N)
X = Z @ np.full(K, 0.2) + u                    # endogenous regressor
Y = 1 + 2 * X + u + rng.normal(size=N)

result = JIVE1(Y, X, Z)                        # or JIVE1(Y, X, Z, W=controls)
result.summary()                               # regression table
result.beta                                    # [constant, coefficient on X, (controls...)]
```

| Argument | Shape | Notes |
|---|---|---|
| `Y` | `(N,)` | Outcome. Must be one-dimensional. |
| `X` | `(N,)` or `(N, L)` | Endogenous regressors. Do **not** include a constant. |
| `Z` | `(N,)` or `(N, K)` | Excluded instruments, `K >= L`. Do **not** include a constant. |
| `W` | `(N,)` or `(N, G)` | Optional exogenous controls (the fourth argument). They are added to both `X` and `Z`. |
| `talk` | bool | Keyword-only. `True` prints step-by-step debugging output. |

NumPy arrays, pandas Series / DataFrames and plain lists are all accepted, and are converted to `float64`. A constant is always added, so any constant columns in `X`, `Z` or `W` are dropped first (a column counts as constant only if it is constant up to floating-point noise, so an instrument with a large mean and a small spread is kept).

All four estimators return the same kind of result object (`JIVE1Result`, `JIVE2Result`, `UJIVE1Result`, `UJIVE2Result`, all built on one base class). Coefficients are ordered `[constant, endogenous regressors..., controls...]`.

| Attribute | Meaning |
|---|---|
| `beta` | Coefficients, shape `(P,)`. |
| `se` | Standard errors, `sqrt(diag(vcov))`, shape `(P,)`. |
| `vcov` | Robust covariance matrix of `beta`, shape `(P, P)`. |
| `tstats`, `pvals` | t-statistics and two-sided p-values, arrays of shape `(P,)`. |
| `cis` | 95% confidence intervals (lower, upper), shape `(P, 2)`. |
| `f_stat`, `f_pval` | **Robust Wald test** that all coefficients except the constant are zero, reported as an F-statistic with `(P-1, N-P)` degrees of freedom. |
| `first_stage_f`, `first_stage_f_pval` | Partial first-stage F-test that the excluded instruments jointly explain each endogenous regressor, given the constant and controls (classical, homoskedastic). A float for one endogenous regressor, otherwise one entry per regressor. F below 10 is the usual rule of thumb for weak instruments, and `summary()` warns in that case. |
| `root_mse` | `sqrt(RSS / (N - P))` from the structural residuals `Y - X beta`. |
| `r_squared`, `adjusted_r_squared` | `1 - RSS/TSS` from the same residuals. **For IV this is not a goodness-of-fit measure**: it can be negative and is not comparable to an OLS R-squared. |
| `leverage`, `fitted_values` | Leverage of each observation, and the ordinary first-stage fitted values of the endogenous regressors. |

The result can also be indexed like a dictionary, for example `result['beta']`. `result.standard_errors` still works but is a deprecated alias of `vcov` (it was always the covariance matrix, not the standard errors) and emits a `DeprecationWarning`.

### Estimators

Let $h_i$ be the leverage of observation $i$ in the first stage (the $i$th diagonal element of $P = Z(Z'Z)^{-1}Z'$), where $Z$ contains the instruments, the constant and any controls. The jackknife fitted value for the endogenous regressors is:

| Estimator | $\tilde{x}_i$ |
|---|---|
| JIVE1 | $\dfrac{Z_i\hat{\pi} - h_i X_i}{1 - h_i}$ |
| UJIVE1 | $\dfrac{Z_i\hat{\pi} - h_i X_i}{1 - h_i}$ |
| JIVE2 | $\dfrac{Z_i\hat{\pi} - h_i X_i}{1 - 1/N}$ |
| UJIVE2 | $\dfrac{Z_i\hat{\pi} - h_i X_i}{1 - 1/N}$ |

Write $\tilde{X}$ for the matrix that stacks $\tilde{x}_i$ together with the constant and controls. The two families differ in the second step:

- **JIVE1 / JIVE2** regress $Y$ on $\tilde{X}$: $\hat\beta = (\tilde{X}'\tilde{X})^{-1}\tilde{X}'Y$.
- **UJIVE1 / UJIVE2** use $\tilde{X}$ as the instrument for $X$: $\hat\beta = (\tilde{X}'X)^{-1}\tilde{X}'Y$.

Standard errors are heteroskedasticity-robust sandwich estimates following Poi (2006), $B^{-1}\big(\sum_i \hat{e}_i^2\,\tilde{x}_i\tilde{x}_i'\big)B^{-T}$, where $B$ is $\tilde{X}'\tilde{X}$ (JIVE) or $\tilde{X}'X$ (UJIVE) and $\hat{e} = Y - X\hat\beta$. The t-tests and confidence intervals use $N - (\text{number of coefficients})$ degrees of freedom.

**Inference conventions.** The covariance has no finite-sample correction, and tests and intervals use a t distribution with $N - P$ degrees of freedom ($P$ = number of coefficients). R's `jive()` uses z-based inference and an optional small-sample correction (`ssc`), and Stata's `jive` reports homoskedastic standard errors by default, so compare coefficients and standard errors (as [`validation/`](validation/) does) rather than p-values and intervals.

### Naming: "UJIVE" means different things in different software

We use Stata's `jive` naming (Poi 2006). Other software and papers use "UJIVE" for a different estimator, so results are only comparable when you match the right pair:

| This package | What it computes | Comparable to |
|---|---|---|
| `UJIVE1` | Angrist, Imbens, Krueger (1999) **JIVE1**: instrument `X̃` with `(Zπ̂ − hX)/(1−h)` | Stata `jive, ujive1` (the default); **R `jive::jive()`** (Kyle Butts) |
| `UJIVE2` | Angrist, Imbens, Krueger (1999) **JIVE2**: same with `1 − 1/N` | Stata `jive, ujive2` |
| `JIVE1`, `JIVE2` | Regress `Y` on the jackknife fit `X̃` (`X̃'X̃`) | Stata `jive, jive1` / `jive2` (to be confirmed) |
| *(not implemented)* | **Kolesár (2013) UJIVE**: leave-one-out fit on `[Z, W]` minus a leave-one-out fit on the controls `W` alone | R `jive::ujive()`; Stata `sjive, noshrink`; Frandsen et al. |

In short: **our `UJIVE1` is the R package's `jive()`, not its `ujive()`**. In Kolesár's terminology our `UJIVE1` is "JIVE". The two differ only in how the controls are handled; with no controls other than the constant they are still not identical, because the constant is itself a control.

**Checked against R.** On the simulated data in [`validation/`](validation/), our `UJIVE1` reproduces R's `jive()` (coefficient and standard error) to at least six decimals, with and without controls, at N = 300 and N = 60 (`tests/test_r_reference.py`). R's `ujive()` matches none of our estimators. The comparison with Stata's `jive` and `sjive` is set up in `validation/run_stata.do` but has not been run yet, so the Stata column above is based on Kolesár's documentation, not on our own results.

### How it is computed (no projection matrix)

The $N \times N$ projection matrix $P$ is never built, and $Z'Z$ is never inverted. Instead we take one reduced QR decomposition of $Z$ and reuse it everywhere:

```python
Q, R = np.linalg.qr(Z, mode="reduced")
fit = Q @ (Q.T @ X)                    # first-stage fitted values, P @ X
leverage = np.sum(Q**2, axis=1)        # diag(P)
```

The JIVE coefficients come from `np.linalg.lstsq`, the UJIVE coefficients from `np.linalg.solve`, and the sandwich variance from two `np.linalg.solve` calls. The residual "meat" term is computed as one matrix product, not a loop over observations. Memory use is $O(NK)$ instead of $O(N^2)$, and the results agree with the explicit-inverse formulas to about $10^{-15}$ (this is checked in `tests/`).

### Small samples and error messages

The estimators work for small $N$ as long as the first stage is identified. They raise a `ValueError` with an explanation when the inputs cannot support an estimate:

- **`N` must be larger than the number of columns of `Z`** (instruments + constant + controls). With as many columns as observations every leverage is exactly 1 and the jackknife is undefined.
- **`Y`, `X`, `Z` or `W` contains `NaN` or `inf`, or is not numeric.** Nothing is dropped or imputed silently.
- **Fewer excluded instruments than endogenous regressors** (under-identified). The estimate would be meaningless, so this is an error.
- **`Z` is rank deficient.** A collinear instrument or control (for example a duplicated column or a full set of dummies plus the constant) is detected from the QR factor. Drop the redundant column.
- **A leverage is (numerically) 1.** This happens when one observation is the only one with a given instrument or control value, such as a dummy that is 1 for a single row.

With few observations per instrument the leverages are large, so the jackknife fitted values and standard errors are noisy. Expect wide confidence intervals in that case.

### Running the tests

```
pip install pytest
pytest
```

### Changes in 0.2.0

- **Fixed:** a positional `W` was silently ignored (the unused `G` argument is removed; `W` is the fourth argument and `talk` is keyword-only); real instruments with a large mean and small spread were dropped as "constant"; the first-stage F was wrong when there were controls; `JIVE1` had no intercept without controls and `JIVE2` ignored `W`.
- **Input checks:** `NaN` / `inf`, non-numeric data and under-identified models now raise clear errors; lists are accepted.
- **Changed:** `f_stat` is now a robust Wald F-statistic (it used an OLS-style formula that is invalid for IV); `pvals`, `tstats` and `cis` are always NumPy arrays; `standard_errors` is deprecated in favour of `vcov` and `se`.
- **Added:** `first_stage_f` and `first_stage_f_pval` on every estimator; `se`, `vcov`, `f_pval`.
- **Internal:** the four estimators share one implementation, so a fix applies to all of them.

### Other estimators

Earlier versions of this repository also contained 2SLS, LIML, HFUL, SJIVE, IJIVE, CJIVE, Anderson-Rubin and Lagrange multiplier tests, and the Monte Carlo simulation code and results. They were removed from `main` to keep this repository focused, but they are still in the git history and on the `archive/full-package` branch.
