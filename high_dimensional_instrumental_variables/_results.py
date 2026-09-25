"""Result object shared by JIVE1, JIVE2, UJIVE1 and UJIVE2."""
import warnings
from dataclasses import dataclass, field, fields
from typing import ClassVar

import numpy as np

from ._common import first_stage_lines


@dataclass
class IVResult:
    """Results of a jackknife IV regression.

    The coefficient vector is ordered ``[constant, endogenous regressors..., controls...]``.

    Attributes
    ----------
    beta : ndarray, shape (P,)
        Estimated coefficients.
    vcov : ndarray, shape (P, P)
        Heteroskedasticity-robust covariance matrix of ``beta`` (a sandwich estimator following Poi (2006), with no
        finite-sample correction).
    leverage : ndarray, shape (N, 1)
        Leverage (diagonal of the first-stage projection matrix) of each observation.
    fitted_values : ndarray, shape (N, L)
        Ordinary first-stage fitted values of the endogenous regressors, before the jackknife correction.
    tstats, pvals : ndarray, shape (P,)
        t-statistics and two-sided p-values, using a t distribution with ``N - P`` degrees of freedom.
    cis : ndarray, shape (P, 2)
        95% confidence intervals (lower, upper), from the same t distribution.
    root_mse : float
        Root mean squared error, ``sqrt(RSS / (N - P))``, with residuals ``Y - X beta`` from the structural equation.
    r_squared, adjusted_r_squared : float
        ``1 - RSS / TSS`` and its degrees-of-freedom adjustment, computed from the structural residuals. **For IV this
        is not a goodness-of-fit measure**: it can be negative and its value is not comparable to an OLS R-squared.
    f_stat, f_pval : float
        Heteroskedasticity-robust Wald test that all coefficients except the constant are zero, reported as an
        F-statistic (Wald / (P - 1)) with (P - 1, N - P) degrees of freedom.
    first_stage_f, first_stage_f_pval : float or ndarray
        Partial first-stage F-test (classical, homoskedastic) that the excluded instruments jointly explain each
        endogenous regressor, given the constant and controls. A float for one endogenous regressor, otherwise one
        entry per regressor. The usual rule of thumb is that F < 10 signals weak instruments.

    Notes
    -----
    ``se`` is the vector of standard errors, ``sqrt(diag(vcov))``. ``standard_errors`` is a deprecated alias of ``vcov``
    (it has always been the covariance matrix, not the standard errors).
    """

    name: ClassVar[str] = "IV"

    beta: np.ndarray
    vcov: np.ndarray
    leverage: np.ndarray = field(repr=False)
    fitted_values: np.ndarray = field(repr=False)
    tstats: np.ndarray
    pvals: np.ndarray
    cis: np.ndarray
    root_mse: float
    r_squared: float
    adjusted_r_squared: float
    f_stat: float
    f_pval: float
    first_stage_f: float | np.ndarray
    first_stage_f_pval: float | np.ndarray

    @property
    def se(self):
        """Standard errors of ``beta``: ``sqrt(diag(vcov))``."""
        return np.sqrt(np.diag(self.vcov))

    @property
    def standard_errors(self):
        """Deprecated alias of ``vcov`` (the covariance matrix). Use ``se`` for standard errors."""
        warnings.warn("`standard_errors` is the covariance matrix; use `.vcov` for it or `.se` for the standard errors.",
                      DeprecationWarning, stacklevel=2)
        return self.vcov

    def __getitem__(self, key: str):
        """Dictionary-style access to the attributes, e.g. ``result['beta']``."""
        valid = [f.name for f in fields(self)] + ["se", "standard_errors"]
        if key not in valid:
            raise KeyError(f"Invalid key '{key}'. Valid keys are " + ", ".join(f"'{k}'" for k in valid) + ".")
        return getattr(self, key)

    def summary(self):
        """Print a regression table (coefficients, robust standard errors, t-tests, confidence intervals) and fit statistics."""
        import pandas as pd

        table = pd.DataFrame({
            "Coefficient": self.beta,
            "Std. Error": self.se,
            "t-stat": self.tstats,
            "P>|t|": self.pvals,
            "Conf. Int. Low": self.cis[:, 0],
            "Conf. Int. High": self.cis[:, 1],
        })
        print(f"\n{self.name} Regression Results")
        print("=" * 80)
        print(table.round(6).to_string(index=False))
        print("-" * 80)
        print(f"R-squared (IV; not a goodness-of-fit measure): {self.r_squared:.6f}")
        print(f"Adjusted R-squared: {self.adjusted_r_squared:.6f}")
        print(f"Robust Wald F-statistic: {self.f_stat:.6f}  (p = {self.f_pval:.6f})")
        print(f"Root MSE: {self.root_mse:.6f}")
        for line in first_stage_lines(self.first_stage_f, self.first_stage_f_pval):
            print(line)
        print("=" * 80)
