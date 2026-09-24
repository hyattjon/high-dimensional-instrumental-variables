# JIVE2 estimator
import numpy as np
import logging
from numpy.typing import NDArray
from scipy.stats import t
from ._common import prepare_inputs, first_stage_f, first_stage_lines, talk_option

# Debug output (talk=True) is switched on per call by @talk_option; see _common.talk_logging.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class JIVE2Result:
    """
    Stores results for the JIVE2 estimator.

    Attributes
    ----------
    beta : NDArray[np.float64]
        Estimated coefficients for the JIVE2 model.
    leverage : NDArray[np.float64]
        Leverage values for each observation.
    fitted_values : NDArray[np.float64]
        Fitted values from the first pass of the JIVE2 estimator.
    r_squared : float
        R-squared value of the model.
    adjusted_r_squared : float
        Adjusted R-squared value of the model.
    f_stat : float
        F-statistic of the model.
    standard_errors : NDArray[np.float64]
        Robust standard errors for the estimated coefficients.
    root_mse : float
        Root mean squared error of the model.
    pvals : list of float
        p-values for the estimated coefficients.
    tstats : list of float
        t-statistics for the estimated coefficients.
    cis : list of tuple
        Confidence intervals for the estimated coefficients.
    """
    def __init__(self, 
                 beta: NDArray[np.float64], 
                 leverage: NDArray[np.float64], 
                 fitted_values: NDArray[np.float64],
                 r_squared: float, 
                 adjusted_r_squared: float, 
                 f_stat: float,
                 standard_errors: NDArray[np.float64],
                 root_mse: float | None = None,
                 pvals: NDArray[np.float64] | None = None,
                 tstats: NDArray[np.float64] | None = None,
                 cis: NDArray[np.float64] | None = None,
                 first_stage_f: float | None = None,
                 first_stage_f_pval: float | None = None):
        self.beta = beta
        self.leverage = leverage
        self.fitted_values = fitted_values
        self.r_squared = r_squared
        self.adjusted_r_squared = adjusted_r_squared
        self.f_stat = f_stat
        self.standard_errors = standard_errors
        self.root_mse = root_mse
        self.pvals = pvals
        self.tstats = tstats
        self.cis = cis
        self.first_stage_f = first_stage_f
        self.first_stage_f_pval = first_stage_f_pval

    def __getitem__(self, key: str):
        """
        Allows dictionary-like access to JIVE2Result attributes.

        Parameters
        ----------
        key : str
            The attribute name to retrieve.

        Returns
        -------
        The value of the requested attribute.

        Raises
        ------
        KeyError
            If the key is not a valid attribute name.
        """
        if key == 'beta':
            return self.beta
        elif key == 'leverage':
            return self.leverage
        elif key == 'fitted_values':
            return self.fitted_values
        elif key == 'r_squared':
            return self.r_squared
        elif key == 'adjusted_r_squared':
            return self.adjusted_r_squared
        elif key == 'f_stat':
            return self.f_stat
        elif key == 'standard_errors':
            return self.standard_errors
        elif key == 'root_mse':
            return self.root_mse
        elif key == 'pvals':
            return self.pvals
        elif key == 'tstats':
            return self.tstats
        elif key == 'cis':
            return self.cis
        elif key == 'first_stage_f':
            return self.first_stage_f
        elif key == 'first_stage_f_pval':
            return self.first_stage_f_pval
        else:
            raise KeyError(f"Invalid key '{key}'. Valid keys are 'beta', 'leverage', 'fitted_values', 'r_squared', 'adjusted_r_squared', 'f_stat', 'standard_errors', 'root_mse', 'pvals', 'tstats', 'cis', 'first_stage_f', or 'first_stage_f_pval'.")


    def __repr__(self):
        return f"JIVE2Result(beta={self.beta}, leverage={self.leverage}, fitted_values={self.fitted_values}, r_squared={self.r_squared}, adjusted_r_squared={self.adjusted_r_squared}, f_stat={self.f_stat}, standard_errors={self.standard_errors}, root_mse={self.root_mse}, pvals={self.pvals}, tstats={self.tstats}, cis={self.cis}, first_stage_f={self.first_stage_f}, first_stage_f_pval={self.first_stage_f_pval})"

    def summary(self):
        """
        Prints a summary of the JIVE2 results in a tabular format similar to statsmodels OLS and UJIVE1.
        """
        import pandas as pd
        import numpy as np

        summary_df = pd.DataFrame({
            "Coefficient": self.beta.flatten(),
            "Std. Error": np.sqrt(np.diag(self.standard_errors)) if self.standard_errors is not None else np.nan,
            "t-stat": self.tstats,
            "P>|t|": self.pvals,
            "Conf. Int. Low": [ci[0] for ci in self.cis] if self.cis is not None else np.nan,
            "Conf. Int. High": [ci[1] for ci in self.cis] if self.cis is not None else np.nan
        })

        print("\nJIVE2 Regression Results")
        print("=" * 80)
        print(summary_df.round(6).to_string(index=False))
        print("-" * 80)
        print(f"R-squared: {self.r_squared:.6f}" if self.r_squared is not None else "R-squared: N/A")
        print(f"Adjusted R-squared: {self.adjusted_r_squared:.6f}" if self.adjusted_r_squared is not None else "Adjusted R-squared: N/A")
        print(f"F-statistic: {self.f_stat:.6f}" if self.f_stat is not None else "F-statistic: N/A")
        print(f"Root MSE: {self.root_mse:.6f}" if self.root_mse is not None else "Root MSE: N/A")
        for line in first_stage_lines(self.first_stage_f, self.first_stage_f_pval):
            print(line)
        print("=" * 80)


@talk_option(logger)
def JIVE2(Y: NDArray[np.float64], X: NDArray[np.float64], Z: NDArray[np.float64], W: NDArray[np.float64] | None = None, *, talk: bool = False) -> JIVE2Result:
    """
    Calculates the JIVE2 jackknife IV estimator: Y is regressed on the leave-one-out first-stage fitted values of the endogenous regressors (Stata jive naming; see the README section on naming).

    Args:
        Y (NDArray[np.float64]): A 1-D numpy array of the dependent variable (N,).
        X (NDArray[np.float64]): A 2-D numpy array of the endogenous regressors (N x L). Do not include the constant.
        Z (NDArray[np.float64]): A 2-D numpy array of the instruments (N x K), where K >= L. Do not include the constant.
        W (NDArray[np.float64]): A 2-D numpy array of the exogenous controls (N x G). Do not include the constant. Optional (default None).
        talk (bool): If True, provides detailed output for teaching purposes. Default is False.

    Returns:
        JIVE2Result: An object containing the following attributes:
            - beta (NDArray[np.float64]): The estimated coefficients for the model.
            - leverage (NDArray[np.float64]): The leverage values for each observation.
            - fitted_values (NDArray[np.float64]): The fitted values from the first pass of the JIVE2 estimator.
            - r_squared (float): The R-squared value for the model.
            - adjusted_r_squared (float): The adjusted R-squared value for the model.
            - f_stat (float): The F-statistic for the model.
            - standard_errors (NDArray[np.float64]): The robust standard errors for the estimated coefficients.
            - root_mse, pvals, tstats, cis: Root MSE, p-values, t-statistics and 95% confidence intervals.

    Raises:
        ValueError: If Y, X, Z or W contain NaN / inf or non-numeric data, have inconsistent dimensions, if there are fewer excluded
            instruments than endogenous regressors, if N is not larger than the number of columns of Z, or if Z is rank deficient.

    Notes:
        - The JIVE2 estimator is a jackknife-based instrumental variable estimator designed to reduce bias in the presence of many instruments.
        - The function performs a two-pass estimation:
            1. The first pass calculates fitted values and leverage values using the instruments.
            2. The second pass removes the ith observation to calculate unbiased estimates.
        - Additional statistics such as R-squared, adjusted R-squared, F-statistics, and robust standard errors are calculated for model evaluation.
        - first_stage_f / first_stage_f_pval: the partial first-stage F-test (classical, homoskedastic) that the excluded instruments jointly explain
          each endogenous regressor, given the constant and controls. A float for one endogenous regressor, an array otherwise. The usual rule of
          thumb is that F < 10 signals weak instruments.
        - The N x N projection matrix Z(Z'Z)^-1 Z' is never formed. The first stage uses a reduced QR decomposition of Z
          (fit = Q Q'X, leverage = row sums of Q squared), and the coefficients and variance come from lstsq / solve
          instead of an explicit inverse. N must be larger than the number of columns of Z (instruments + constant + controls).
        - JIVE2 regresses Y on the jackknife fit (X~'X~), following Stata's jive command naming (Poi 2006). The IV form (X~'X) is UJIVE2 in this package. See the README section on naming.

    Example:
        >>> import numpy as np
        >>> from high_dimensional_instrumental_variables.jive2 import JIVE2
        >>> rng = np.random.default_rng(0)
        >>> Z = rng.normal(size=(100, 5))
        >>> u = rng.normal(size=100)
        >>> X = Z @ np.full(5, 0.5) + u
        >>> Y = 1 + 2 * X + u + rng.normal(size=100)
        >>> result = JIVE2(Y, X, Z)
        >>> print(result.beta)
    """

    # Convert to float arrays; check for NaN / inf, shapes and identification; drop constant columns
    Y, X, Z, W = prepare_inputs(Y, X, Z, W, logger)
    N = Y.shape[0]


    # Add the constant and the controls
    k = X.shape[1]
    ones = np.ones((N, 1))
    X = np.hstack((ones, X))
    Z = np.hstack((ones, Z))
    if W is not None:
        X = np.hstack((X, W))
        Z = np.hstack((Z, W))
        logger.debug("Controls W have been added to both X and Z.\n")

    # First pass to get fitted values and leverage. We use a QR decomposition of Z instead of forming
    # the N x N projection matrix P = Z(Z'Z)^-1 Z' (or inverting Z'Z).
    if Z.shape[1] >= N:
        raise ValueError(f"N must be larger than the number of columns of Z (instruments + constant + controls). Got N = {N} and {Z.shape[1]} columns.")
    Qz, Rz = np.linalg.qr(Z, mode="reduced")
    diag_R = np.abs(np.diag(Rz))
    if diag_R.min() <= diag_R.max() * max(Z.shape) * np.finfo(float).eps:
        raise ValueError("Z (with the constant and controls) is rank deficient. Remove collinear instruments or controls.")
    fit = Qz @ (Qz.T @ X)
    logger.debug(f"Fitted values obtained.\n")

    # First-stage F: partial F-test of the excluded instruments, given the constant and controls
    fs_F, fs_F_pval = first_stage_f(X[:, 1:1 + k], Qz, np.hstack((ones, W)) if W is not None else ones)

    # Leverage is the main diagonal of the projection matrix: the row sums of Q squared
    leverage = np.sum(Qz**2, axis=1)
    if np.any(leverage >= 1 - 1e-10):
        raise ValueError("Leverage values must be strictly less than 1 to avoid division by zero. An observation is the only one with its instrument / control values.")
    logger.debug(f"Leverage values obtained.\n")

    # Reshape leverage to an Nx1 vector
    leverage = leverage.reshape(-1, 1)
    logger.debug(f"First pass complete.\n")

    # Second pass to remove ith row and reduce bias
    fit = fit[:, 1:1+k]
    X = X[:,1:1+k]
    X_jive2 = (fit - leverage * X) / (1 - (1/N))
    logger.debug(f"Second pass complete.\n")

    if W is not None:
        X_jive2 = np.hstack((ones, X_jive2, W))
        X = np.hstack((ones, X, W))
    else:
        X_jive2 = np.hstack((ones, X_jive2))
        X = np.hstack((ones, X))

    # Calculate the JIVE2 estimates
    beta_jive2, *_ = np.linalg.lstsq(X_jive2, Y, rcond=None)
    logger.debug(f"JIVE2 Estimates:\n{beta_jive2}\n")

    #Now, lets get standard errors and do a t-test. We follow Poi (2006).
    resid = Y - X @ beta_jive2
    midsum = (X_jive2 * (resid**2)[:, None]).T @ X_jive2
    A = X_jive2.T @ X_jive2
    left = np.linalg.solve(A, midsum)
    robust_v = np.linalg.solve(A, left.T).T


    #Lets do a hypothesis test that B1=0
    pvals = []
    tstats = []
    cis = []

    K = X.shape[1]
    dof = N - K
    for i in range(K):
        t_stat_i = (beta_jive2[i])/((robust_v[i,i])**.5)
        pval_i = 2 * (1 - t.cdf(np.abs(t_stat_i), df=dof))
        t_crit_i = t.ppf(0.975, df=dof)

        ci_lower = beta_jive2[i] - t_crit_i * (robust_v[i,i])**.5
        ci_upper = beta_jive2[i] + t_crit_i * (robust_v[i,i])**.5
        ci_i = (ci_lower, ci_upper)
        tstats.append(float(t_stat_i))
        pvals.append(float(pval_i))
        cis.append(ci_i)  

    #Grab the R^2 for the model:
    yfit = X @ beta_jive2
    ybar = np.mean(Y)
    r2 = 1 - np.sum((Y-yfit)**2) / np.sum((Y-ybar)**2)
    
    #Overall F-stat for the model:
    q = X.shape[1]
    e = Y-yfit
    F = ((np.sum((yfit-ybar)**2)) / (q-1)) / ((e.T @ e)/(N-q))

    #Mean-square error:
    root_mse = float(((1/(N-q)) * (np.sum((Y - yfit)**2)))**.5)

    #Adjustred R2
    ar2 = 1 - (((1-r2)*(N-1))/(N-q))

    return JIVE2Result(beta=beta_jive2,
                       leverage=leverage,
                       fitted_values=fit,
                       r_squared=r2,
                       adjusted_r_squared=ar2,
                       f_stat=F,
                       standard_errors=robust_v,
                       root_mse=root_mse,
                       pvals=pvals,
                       tstats=tstats,
                       cis=cis,
                       first_stage_f=fs_F,
                       first_stage_f_pval=fs_F_pval)
