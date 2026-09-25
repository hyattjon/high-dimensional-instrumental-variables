# UJIVE2 estimator
import logging

import numpy as np
from numpy.typing import NDArray

from ._common import prepare_inputs, talk_option
from ._core import fit_jackknife_iv
from ._results import IVResult

# Debug output (talk=True) is switched on per call by @talk_option; see _common.talk_logging.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class UJIVE2Result(IVResult):
    """Results of the UJIVE2 estimator. See :class:`~high_dimensional_instrumental_variables._results.IVResult` for the attributes."""

    name = "UJIVE2"


@talk_option(logger)
def UJIVE2(Y: NDArray[np.float64], X: NDArray[np.float64], Z: NDArray[np.float64], W: NDArray[np.float64] | None = None, *, talk: bool = False) -> UJIVE2Result:
    """
    Calculates the UJIVE2 estimator: the Angrist, Imbens, and Krueger (1999) JIVE2 in IV form, beta = (X~'X)^-1 X~'Y, where the jackknife fit
    X~_i = (Z_i pi_hat - h_i X_i) / (1 - 1/N) is used as the instrument for X.

    Args:
        Y (NDArray[np.float64]): A 1-D array of the dependent variable (N,).
        X (NDArray[np.float64]): The endogenous regressors, (N,) or (N x L). Do not include the constant.
        Z (NDArray[np.float64]): The excluded instruments, (N,) or (N x K), where K >= L. Do not include the constant.
        W (NDArray[np.float64]): Optional exogenous controls, (N,) or (N x G). Do not include the constant. Added to both X and Z.
        talk (bool): Keyword-only. If True, prints step-by-step output for teaching / debugging. Default is False.

    Returns:
        UJIVE2Result: coefficients ``beta`` ordered [constant, endogenous regressors, controls], the robust covariance ``vcov`` and
        standard errors ``se``, t-statistics, p-values, confidence intervals, R-squared, a robust Wald F-statistic, root MSE, leverage,
        first-stage fitted values and the first-stage F-statistic. See :class:`~high_dimensional_instrumental_variables._results.IVResult`.

    Raises:
        ValueError: If Y, X, Z or W contain NaN / inf or non-numeric data, have inconsistent dimensions, if there are fewer excluded
            instruments than endogenous regressors, if N is not larger than the number of columns of Z, or if Z is rank deficient.

    Notes:
        - The N x N projection matrix Z(Z'Z)^-1 Z' is never formed. The first stage uses a reduced QR decomposition of Z
          (fit = Q Q'X, leverage = row sums of Q squared), and the coefficients and variance come from solve / lstsq
          instead of an explicit inverse. N must be larger than the number of columns of Z (instruments + constant + controls).
        - Standard errors are heteroskedasticity-robust sandwich estimates (Poi 2006) with no finite-sample correction; tests and
          confidence intervals use a t distribution with N minus the number of coefficients degrees of freedom.
        - first_stage_f / first_stage_f_pval: the partial first-stage F-test (classical, homoskedastic) that the excluded instruments jointly
          explain each endogenous regressor, given the constant and controls. The usual rule of thumb is that F < 10 signals weak instruments.
        - Naming follows Stata's jive command (Poi 2006): UJIVE2 is the Angrist, Imbens, and Krueger (1999) JIVE2 in IV form (denominator 1 - 1/N).
          It is NOT Kolesar's (2013) UJIVE, which this package does not implement. See the README section on naming.

    Example:
        >>> import numpy as np
        >>> from high_dimensional_instrumental_variables.ujive2 import UJIVE2
        >>> rng = np.random.default_rng(0)
        >>> Z = rng.normal(size=(100, 5))
        >>> u = rng.normal(size=100)
        >>> X = Z @ np.full(5, 0.5) + u
        >>> Y = 1 + 2 * X + u + rng.normal(size=100)
        >>> result = UJIVE2(Y, X, Z)
        >>> print(result.beta)
    """
    Y, X, Z, W = prepare_inputs(Y, X, Z, W, logger)
    return UJIVE2Result(**fit_jackknife_iv(Y, X, Z, W, leverage_denominator="n", iv_form=True, logger=logger, label="UJIVE2"))
