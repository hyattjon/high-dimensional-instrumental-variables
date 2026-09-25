"""The computation shared by JIVE1, JIVE2, UJIVE1 and UJIVE2.

The four estimators differ in two places only:

* the denominator of the jackknife fitted value, ``1 - h_i`` (JIVE1 / UJIVE1) or ``1 - 1/N`` (JIVE2 / UJIVE2), and
* the second step: regress Y on the jackknife fit, ``(X~'X~)^-1 X~'Y`` (JIVE1 / JIVE2), or use the jackknife fit as an
  instrument for X, ``(X~'X)^-1 X~'Y`` (UJIVE1 / UJIVE2).

The N x N projection matrix is never formed: a reduced QR decomposition of Z gives both the first-stage fit and the
leverage.
"""
import numpy as np
from scipy.stats import f as f_dist
from scipy.stats import t as t_dist

from ._common import first_stage_f


def fit_jackknife_iv(Y, X, Z, W, *, leverage_denominator, iv_form, logger, label):
    """Estimate one of the four jackknife IV estimators.

    Parameters
    ----------
    Y, X, Z, W
        The output of :func:`prepare_inputs`: float arrays with constant columns already dropped. ``X`` holds only the
        endogenous regressors and ``Z`` only the excluded instruments; ``W`` (controls) may be ``None``.
    leverage_denominator : {"leverage", "n"}
        ``"leverage"`` uses ``1 - h_i`` (JIVE1 / UJIVE1); ``"n"`` uses ``1 - 1/N`` (JIVE2 / UJIVE2).
    iv_form : bool
        ``True`` for ``(X~'X)^-1 X~'Y`` (UJIVE1 / UJIVE2), ``False`` for ``(X~'X~)^-1 X~'Y`` (JIVE1 / JIVE2).
    logger, label
        Logger for the ``talk=True`` messages, and the estimator name to use in them.

    Returns
    -------
    dict
        The fields of :class:`IVResult`.
    """
    N, k = X.shape

    # Add the constant and the controls to both the regressors and the instruments
    ones = np.ones((N, 1))
    W_const = ones if W is None else np.hstack((ones, W))
    X = np.hstack((ones, X) if W is None else (ones, X, W))
    Z = np.hstack((ones, Z) if W is None else (ones, Z, W))
    if W is not None:
        logger.debug("Controls W have been added to both X and Z.\n")

    # First pass: fitted values and leverage from a QR decomposition of Z, instead of forming the N x N projection
    # matrix P = Z(Z'Z)^-1 Z' (or inverting Z'Z).
    if Z.shape[1] >= N:
        raise ValueError(f"N must be larger than the number of columns of Z (instruments + constant + controls). Got N = {N} and {Z.shape[1]} columns.")
    Qz, Rz = np.linalg.qr(Z, mode="reduced")
    diag_R = np.abs(np.diag(Rz))
    if diag_R.min() <= diag_R.max() * max(Z.shape) * np.finfo(float).eps:
        raise ValueError("Z (with the constant and controls) is rank deficient. Remove collinear instruments or controls.")
    fit = Qz @ (Qz.T @ X)
    logger.debug("Fitted values obtained.\n")

    endog = X[:, 1:1 + k]
    fs_F, fs_F_pval = first_stage_f(endog, Qz, W_const)

    # Leverage is the main diagonal of the projection matrix: the row sums of Q squared
    leverage = np.sum(Qz ** 2, axis=1)
    if np.any(leverage >= 1 - 1e-10):
        raise ValueError("Leverage values must be strictly less than 1 to avoid division by zero. An observation is the only one with its instrument / control values.")
    logger.debug("Leverage values obtained.\n")
    leverage = leverage.reshape(-1, 1)

    # Second pass: leave observation i out of its own first stage
    fit_endog = fit[:, 1:1 + k]
    denominator = (1 - leverage) if leverage_denominator == "leverage" else (1 - 1 / N)
    X_tilde = (fit_endog - leverage * endog) / denominator
    X_tilde = np.hstack((ones, X_tilde) if W is None else (ones, X_tilde, W))
    logger.debug("Second pass complete.\n")

    # Coefficients and robust (sandwich) covariance, following Poi (2006)
    if iv_form:
        A = X_tilde.T @ X
        beta = np.linalg.solve(A, X_tilde.T @ Y)
    else:
        A = X_tilde.T @ X_tilde
        beta, *_ = np.linalg.lstsq(X_tilde, Y, rcond=None)
    logger.debug(f"{label} Estimates:\n{beta}\n")

    resid = Y - X @ beta
    meat = (X_tilde * (resid ** 2)[:, None]).T @ X_tilde
    left = np.linalg.solve(A, meat)
    vcov = np.linalg.solve(A, left.T).T

    # Inference: t distribution with N - P degrees of freedom
    P = X.shape[1]
    dof = N - P
    se = np.sqrt(np.diag(vcov))
    tstats = beta / se
    pvals = 2 * t_dist.sf(np.abs(tstats), df=dof)
    t_crit = t_dist.ppf(0.975, df=dof)
    cis = np.column_stack((beta - t_crit * se, beta + t_crit * se))

    # Fit statistics from the structural residuals. For IV the R-squared is not a goodness-of-fit measure (see IVResult).
    rss = float(resid @ resid)
    tss = float(np.sum((Y - Y.mean()) ** 2))
    r2 = 1 - rss / tss
    adj_r2 = 1 - (1 - r2) * (N - 1) / dof
    root_mse = float(np.sqrt(rss / dof))

    # Overall test: robust Wald test that every coefficient except the constant is zero
    slopes = beta[1:]
    wald = float(slopes @ np.linalg.solve(vcov[1:, 1:], slopes))
    f_stat = wald / (P - 1)
    f_pval = float(f_dist.sf(f_stat, P - 1, dof))

    return dict(beta=beta, vcov=vcov, leverage=leverage, fitted_values=fit_endog,
                tstats=tstats, pvals=pvals, cis=cis, root_mse=root_mse,
                r_squared=float(r2), adjusted_r_squared=float(adj_r2), f_stat=f_stat, f_pval=f_pval,
                first_stage_f=fs_F, first_stage_f_pval=fs_F_pval)
