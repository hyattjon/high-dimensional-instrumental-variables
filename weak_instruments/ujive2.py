# JIVE2 Estimator
import numpy as np
import logging
from numpy.typing import NDArray
from scipy.stats import t


# Set up the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Default logging level
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')  # Simple format for teaching purposes
handler.setFormatter(formatter)
logger.addHandler(handler)

class UJIVE2Result:
    """
    Stores results for the UJIVE2 estimator.

    Attributes:
        beta: Estimated coefficients.
        leverage: Leverage values for each observation.
        fitted_values: Fitted values from the first pass.
        r_squared: R-squared value.
        adjusted_r_squared: Adjusted R-squared value.
        f_stat: F-statistic for overall model significance.
        standard_errors: Robust standard errors.
        root_mse: Root mean squared error.
        pvals: p-values for coefficients.
        tstats: t-statistics for coefficients.
        cis: Confidence intervals for coefficients.
        first_stage_f: F-statistic for first-stage regression (weak instruments test).
        first_stage_f_pval: P-value for first-stage F-statistic.
    """
    def __init__(self, 
                 beta: NDArray[np.float64], 
                 leverage: NDArray[np.float64], 
                 fitted_values: NDArray[np.float64], 
                 r_squared: float, 
                 adjusted_r_squared: float, 
                 f_stat: float,
                 standard_errors: NDArray[np.float64],
                 root_mse: float,
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
        return f"UJIVE2Result(beta={self.beta}, leverage={self.leverage}, fitted_values={self.fitted_values}, r_squared={self.r_squared}, adjusted_r_squared={self.adjusted_r_squared}, f_stat={self.f_stat}, standard_errors={self.standard_errors}, root_mse={self.root_mse}, pvals={self.pvals}, tstats={self.tstats}, cis={self.cis}, first_stage_f={self.first_stage_f}, first_stage_f_pval={self.first_stage_f_pval})"

    def summary(self):
        """
        Prints a summary of the UJIVE2 results in a tabular format similar to statsmodels OLS.
        """
        import pandas as pd

        summary_df = pd.DataFrame({
            "Coefficient": self.beta.flatten(),
            "Std. Error": np.sqrt(np.diag(self.standard_errors)),
            "t-stat": self.tstats,
            "P>|t|": self.pvals,
            "Conf. Int. Low": [ci[0] for ci in self.cis],
            "Conf. Int. High": [ci[1] for ci in self.cis]
        })

        print("\nUJIVE2 Regression Results")
        print("=" * 80)
        print(summary_df.round(6).to_string(index=False))
        print("-" * 80)
        print(f"R-squared: {self.r_squared:.6f}")
        print(f"Adjusted R-squared: {self.adjusted_r_squared:.6f}")
        print(f"F-statistic: {self.f_stat:.6f}")
        print(f"Root MSE: {self.root_mse:.6f}")
        
        # Print first-stage F-statistic if available
        if self.first_stage_f is not None:
            print("-" * 80)
            print(f"First-stage F-statistic: {self.first_stage_f:.6f}")
            if self.first_stage_f_pval is not None:
                print(f"First-stage F p-value: {self.first_stage_f_pval:.6f}")
            
            # Add warnings for weak instruments based on Stock-Yogo critical values
            if self.first_stage_f < 10:
                print("WARNING: First-stage F < 10, indicating potentially weak instruments")
        print("=" * 80)

def UJIVE2(Y: NDArray[np.float64], X: NDArray[np.float64], Z: NDArray[np.float64], G: NDArray[np.float64] | None = None, W: NDArray[np.float64] | None = None, talk: bool = False) -> UJIVE2Result:
    """
    Calculates the UJIVE2 estimator using a two-pass approach recommended by Angrist, Imbens, and Kreuger (1999) in Jackknife IV estimation.

    Args:
        Y (NDArray[np.float64]): A 1-D numpy array of the dependent variable (N x 1).
        X (NDArray[np.float64]): A 2-D numpy array of the endogenous regressors (N x L). Do not inlude the constant.
        Z (NDArray[np.float64]): A 2-D numpy array of the instruments (N x K), where K > L. Do not include the constant.
        W (NDArray[np.float64]): A 2-D numpy array of the exogenous controls (N x G). Do not include the constant. These are not necessary for the function. 
        talk (bool): If True, provides detailed output for teaching / debugging purposes. Default is False.

    Returns:
        UJIVE1Result: An object containing the following attributes:
            - beta (NDArray[np.float64]): The estimated coefficients for the model.
            - leverage (NDArray[np.float64]): The leverage values for each observation.
            - fitted_values (NDArray[np.float64]): The fitted values from the first pass of the UJIVE1 estimator.
            - r_squared (float): The R-squared value for the model.
            - adjusted_r_squared (float): The adjusted R-squared value for the model.
            - f_stat (float): The F-statistic for the model.
            - standard_errors (NDArray[np.float64]): The robust standard errors for the estimated coefficients.

    Raises:
        ValueError: If the dimensions of Y, X, or Z are inconsistent or invalid.
        RuntimeWarning: If the number of instruments (columns in Z) is not greater than the number of regressors (columns in X).

    Notes:
        - The UJIVE2 estimator is a jackknife-based instrumental variable estimator designed to reduce bias in the presence of many instruments.
        - The function performs a two-pass estimation:
            1. The first pass calculates fitted values and leverage values using the instruments.
            2. The second pass removes the ith observation to calculate unbiased estimates.
        - Additional statistics such as R-squared, adjusted R-squared, and F-statistics are calculated for model evaluation.
        - If the number of endogenous regressors is 1, first-stage statistics (R-squared and F-statistic) are also computed.
        - The N x N projection matrix Z(Z'Z)^-1 Z' is never formed. The first stage uses a reduced QR decomposition of Z
          (fit = Q Q'X, leverage = row sums of Q squared), and the coefficients and variance come from solve
          instead of an explicit inverse. N must be larger than the number of columns of Z (instruments + constant + controls).

    Example:
        >>> import numpy as np
        >>> from weak_instruments.ujive2 import UJIVE2
        >>> rng = np.random.default_rng(0)
        >>> Z = rng.normal(size=(100, 5))
        >>> u = rng.normal(size=100)
        >>> X = Z @ np.full(5, 0.5) + u
        >>> Y = 1 + 2 * X + u + rng.normal(size=100)
        >>> result = UJIVE2(Y, X, Z)
        >>> print(result.beta)
    """

    # Convert pandas DataFrames/Series to numpy arrays
    if hasattr(Y, "values"):
        Y = Y.values
    if hasattr(X, "values"):
        X = X.values
    if hasattr(Z, "values"):
        Z = Z.values
    if G is not None and hasattr(G, "values"):
        G = G.values
    if W is not None and hasattr(W, "values"):
        W = W.values

    # Adjust logging level based on the `talk` parameter
    if talk:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

    # Check if Y is a one-dimensional array
    if Y.ndim != 1:
        raise ValueError(f"Y must be a one-dimensional array, but got shape {Y.shape}.")
    # Check if Z is at least a one-dimensional array
    if Z.ndim < 1:
        raise ValueError(f"Z must be at least a one-dimensional array, but got shape {Z.shape}.")
    
    # If X/Z is a single vector:
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    
    # Check that Y, X, and Z have consistent dimensions
    N = Y.shape[0]
    if X.shape[0] != N:
        raise ValueError(f"X and Y must have the same number of rows. Got X.shape[0] = {X.shape[0]} and Y.shape[0] = {N}.")
    if Z.shape[0] != N:
        raise ValueError(f"Z and Y must have the same number of rows. Got Z.shape[0] = {Z.shape[0]} and Y.shape[0] = {N}.")


    logger.debug(f"Y has {Y.shape[0]} rows.\n")
    logger.debug(f"X has {X.shape[0]} rows and {X.shape[1]} columns.\n")
    logger.debug(f"Z has {Z.shape[0]} rows and {Z.shape[1]} columns.\n")

    # Drop constant columns from X
    constant_columns_X = np.all(np.isclose(X, X[0, :], atol=1e-8), axis=0)
    if np.any(constant_columns_X):  # Check if there are any constant columns
        logger.debug(f"X has constant columns. Dropping columns: {np.where(constant_columns_X)[0]}")
        X = X[:, ~constant_columns_X]  # Keep only non-constant columns

    # Drop constant columns from Z
    constant_columns_Z = np.all(np.isclose(Z, Z[0, :], atol=1e-8), axis=0)
    if np.any(constant_columns_Z):  # Check if there are any constant columns
        logger.debug(f"Z has constant columns. Dropping columns: {np.where(constant_columns_Z)[0]}")
        Z = Z[:, ~constant_columns_Z]  # Keep only non-constant columns

    logger.debug(f"X shape after dropping constant columns: {X.shape}")
    logger.debug(f"Z shape after dropping constant columns: {Z.shape}")

    # Add the constant
    k = X.shape[1]
    ones = np.ones((N, 1))
    X = np.hstack((ones, X))
    Z = np.hstack((ones, Z))

    #Add the controls:
    if W is not None:
        if W.ndim == 1:
            W = W.reshape(-1, 1)
        if W.shape[0] != N:
            raise ValueError(f"W must have the same number of rows as Y. Got G.shape[0] = {W.shape[0]} and Y.shape[0] = {N}.")
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
    X_jive2 = (fit - leverage * X) / (1 - (1 / N))
    logger.debug(f"Second pass complete.\n")

    if W is not None:
        X_jive2 = np.hstack((ones, X_jive2, W))
        X = np.hstack((ones, X, W))
    else:
        X_jive2 = np.hstack((ones, X_jive2))
        X = np.hstack((ones, X))

    # Calculate the UJIVE2 estimates
    A = X_jive2.T @ X
    beta_jive2 = np.linalg.solve(A, X_jive2.T @ Y)
    logger.debug(f"UJIVE2 Estimates:\n{beta_jive2}\n")

    # Now, let's get standard errors and do a t-test. We follow Poi (2006).
    resid = Y - X @ beta_jive2
    midsum = (X_jive2 * (resid ** 2)[:, None]).T @ X_jive2
    left = np.linalg.solve(A, midsum)
    robust_v = np.linalg.solve(A, left.T).T

    # Hypothesis test that B1 = 0
    pvals = []
    tstats = []
    cis = []

    K = X.shape[1]
    dof = N - K
    for i in range(K):
        t_stat_i = beta_jive2[i] / ((robust_v[i, i]) ** 0.5)
        pval_i = 2 * (1 - t.cdf(np.abs(t_stat_i), df=dof))
        t_crit_i = t.ppf(0.975, df=dof)

        ci_lower = beta_jive2[i] - t_crit_i * (robust_v[i, i]) ** 0.5
        ci_upper = beta_jive2[i] + t_crit_i * (robust_v[i, i]) ** 0.5
        ci_i = (ci_lower, ci_upper)
        tstats.append(t_stat_i)
        pvals.append(pval_i)
        cis.append(ci_i)

    # Grab the R^2 for the model:
    yfit = X @ beta_jive2
    ybar = np.mean(Y)
    r2 = 1 - np.sum((Y - yfit) ** 2) / np.sum((Y - ybar) ** 2)

    # Overall F-stat for the model:
    q = X.shape[1]
    e = Y - yfit
    F = ((np.sum((yfit - ybar) ** 2)) / (q - 1)) / ((e.T @ e) / (N - q))

    # Mean-square error:
    root_mse = ((1 / (N - q)) * (np.sum((Y - yfit) ** 2))) ** 0.5

    # Adjusted R^2
    ar2 = 1 - (((1 - r2) * (N - 1)) / (N - q))

    # First stage statistics if the number of endogenous regressors is 1
    if X.ndim == 2:
        X_fs = X[:, 1]
        fs_fit = Qz @ (Qz.T @ X_fs)
        xbar = np.mean(X_fs)

        # First Stage R^2
        fs_r2 = 1 - np.sum((X_fs - fs_fit) ** 2) / np.sum((X_fs - xbar) ** 2)

        # First stage F-stat
        q_fs = Z.shape[1]
        e_fs = X_fs - fs_fit
        fs_F = ((np.sum((fs_fit - xbar) ** 2)) / (q_fs - 1)) / ((e_fs.T @ e_fs) / (N - q_fs))

    return UJIVE2Result(beta=beta_jive2, 
                        leverage=leverage, 
                        fitted_values=fit, 
                        r_squared=r2, 
                        adjusted_r_squared=ar2, 
                        f_stat=F, 
                        standard_errors=robust_v, 
                        root_mse=root_mse, 
                        pvals=np.array(pvals), 
                        tstats=np.array(tstats), 
                        cis=np.array(cis), 
                        first_stage_f=fs_F if 'fs_F' in locals() else None, 
                        first_stage_f_pval=None)
