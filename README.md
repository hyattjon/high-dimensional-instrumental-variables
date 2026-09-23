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

These four estimators reduce the many-instruments bias of 2SLS. Each one replaces the usual first-stage fitted value $\hat{x}_i = Z_i\hat{\pi}$ with a version that leaves observation $i$ out of its own first stage. JIVE1 and JIVE2 follow Blomquist and Dahlberg (1999). UJIVE1 and UJIVE2 follow Angrist, Imbens, and Krueger (1999).

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
| `W` | `(N,)` or `(N, G)` | Optional exogenous controls. They are added to both `X` and `Z`. |
| `talk` | bool | `True` prints step-by-step debugging output. |

Pass `W` by keyword. NumPy arrays and pandas Series / DataFrames are both accepted. A constant is always added, and any constant columns in `X` or `Z` are dropped first.

The result object has `beta`, `standard_errors` (the full robust covariance matrix, so use `np.sqrt(np.diag(...))` for the standard errors), `tstats`, `pvals`, `cis`, `r_squared`, `adjusted_r_squared`, `f_stat`, `root_mse`, `leverage` and `fitted_values`. It can also be indexed like a dictionary, for example `result['beta']`. `UJIVE2` additionally returns `first_stage_f`.

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

### How it is computed (no projection matrix)

The $N \times N$ projection matrix $P$ is never built, and $Z'Z$ is never inverted. Instead we take one reduced QR decomposition of $Z$ and reuse it everywhere:

```python
Q, R = np.linalg.qr(Z, mode="reduced")
fit = Q @ (Q.T @ X)                    # first-stage fitted values, P @ X
leverage = np.sum(Q**2, axis=1)        # diag(P)
```

The JIVE coefficients come from `np.linalg.lstsq`, the UJIVE coefficients from `np.linalg.solve`, and the sandwich variance from two `np.linalg.solve` calls. The residual "meat" term is computed as one matrix product, not a loop over observations. Memory use is $O(NK)$ instead of $O(N^2)$, and the results agree with the explicit-inverse formulas to about $10^{-15}$ (this is checked in `tests/`).

### Small samples and error messages

The estimators work for small $N$ as long as the first stage is identified. They raise a `ValueError` with an explanation when it is not:

- **`N` must be larger than the number of columns of `Z`** (instruments + constant + controls). With as many columns as observations every leverage is exactly 1 and the jackknife is undefined.
- **`Z` is rank deficient.** A collinear instrument or control (for example a duplicated column or a full set of dummies plus the constant) is detected from the QR factor. Drop the redundant column.
- **A leverage is (numerically) 1.** This happens when one observation is the only one with a given instrument or control value, such as a dummy that is 1 for a single row.

With few observations per instrument the leverages are large, so the jackknife fitted values and standard errors are noisy. Expect wide confidence intervals in that case.

### Running the tests

```
pip install pytest
pytest
```

### Other estimators

Earlier versions of this repository also contained 2SLS, LIML, HFUL, SJIVE, IJIVE, CJIVE, Anderson-Rubin and Lagrange multiplier tests, and the Monte Carlo simulation code and results. They were removed from `main` to keep this repository focused, but they are still in the git history and on the `archive/full-package` branch.
