"""Input checks and first-stage diagnostics shared by the jackknife IV estimators."""
import contextlib
import functools
import logging

import numpy as np
from scipy.stats import f as f_dist


def _as_float_array(a, name):
    """Convert lists / pandas objects to a float64 array and reject NaN / inf."""
    if hasattr(a, "values"):
        a = a.values
    try:
        a = np.asarray(a, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} could not be converted to a numeric array: {e}") from None
    n_bad = int(np.sum(~np.isfinite(a)))
    if n_bad:
        raise ValueError(f"{name} contains {n_bad} missing or infinite value(s) (NaN / inf). Remove or impute them before estimating.")
    return a


def _as_2d(a, name, n_rows):
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.ndim != 2:
        raise ValueError(f"{name} must be a one- or two-dimensional array, but got shape {a.shape}.")
    if a.shape[0] != n_rows:
        raise ValueError(f"{name} and Y must have the same number of rows. Got {name}.shape[0] = {a.shape[0]} and Y.shape[0] = {n_rows}.")
    return a


def drop_constant_columns(A, name, logger):
    """Drop columns that are constant up to floating-point noise.

    A column counts as constant only if its range is a few ulps of its largest entry. (A relative tolerance like
    np.isclose's default would treat, say, an instrument with mean 1e6 and sd 0.5 as constant.)
    """
    constant = np.ptp(A, axis=0) <= 4 * np.finfo(np.float64).eps * np.max(np.abs(A), axis=0)
    if np.any(constant):
        logger.debug(f"{name} has constant columns. Dropping columns: {np.where(constant)[0]}")
        A = A[:, ~constant]
    return A


def prepare_inputs(Y, X, Z, W, logger):
    """Validate and standardise the inputs of an estimator.

    Returns float64 arrays Y (N,), X (N, L), Z (N, K) and W (N, G) or None, where the constant columns of X, Z and W
    have been dropped (the estimators add their own constant).

    Raises
    ------
    ValueError
        For NaN / inf, non-numeric data, wrong dimensions, mismatched row counts, or fewer excluded instruments
        than endogenous regressors.
    """
    Y = _as_float_array(Y, "Y")
    X = _as_float_array(X, "X")
    Z = _as_float_array(Z, "Z")
    if W is not None:
        W = _as_float_array(W, "W")

    if Y.ndim != 1:
        raise ValueError(f"Y must be a one-dimensional array, but got shape {Y.shape}.")
    N = Y.shape[0]
    if N == 0:
        raise ValueError("Y is empty.")

    X = _as_2d(X, "X", N)
    Z = _as_2d(Z, "Z", N)
    logger.debug(f"Y has {N} rows.\n")
    logger.debug(f"X has {X.shape[0]} rows and {X.shape[1]} columns.\n")
    logger.debug(f"Z has {Z.shape[0]} rows and {Z.shape[1]} columns.\n")

    X = drop_constant_columns(X, "X", logger)
    Z = drop_constant_columns(Z, "Z", logger)
    logger.debug(f"X shape after dropping constant columns: {X.shape}")
    logger.debug(f"Z shape after dropping constant columns: {Z.shape}")

    if W is not None:
        W = drop_constant_columns(_as_2d(W, "W", N), "W", logger)
        if W.shape[1] == 0:
            W = None

    L, K = X.shape[1], Z.shape[1]
    if L == 0:
        raise ValueError("X has no non-constant columns, so there is no endogenous regressor to estimate.")
    if K < L:
        raise ValueError(f"Need at least as many excluded instruments as endogenous regressors. Got {K} instrument column(s) "
                         f"(after dropping constant columns) and {L} endogenous regressor(s).")
    return Y, X, Z, W


def first_stage_f(endog, Qz, W_const):
    """Partial first-stage F-test that the excluded instruments jointly explain each endogenous regressor.

    Parameters
    ----------
    endog : (N, L) array
        The endogenous regressors.
    Qz : (N, 1 + G + K) array
        Orthonormal basis (reduced QR) of the full instrument set [constant, controls, instruments].
    W_const : (N, 1 + G) array
        The constant and the controls, i.e. the restricted first stage.

    Returns
    -------
    (F, p) : floats if L == 1, otherwise arrays of length L. This is the classical (homoskedastic) F-statistic with
    (K, N - 1 - G - K) degrees of freedom, the one behind the Staiger-Stock rule of thumb F < 10.
    """
    Qw, _ = np.linalg.qr(W_const, mode="reduced")
    rss_full = np.sum((endog - Qz @ (Qz.T @ endog)) ** 2, axis=0)
    rss_restricted = np.sum((endog - Qw @ (Qw.T @ endog)) ** 2, axis=0)
    df1 = Qz.shape[1] - Qw.shape[1]
    df2 = endog.shape[0] - Qz.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        F = ((rss_restricted - rss_full) / df1) / (rss_full / df2)
    p = f_dist.sf(F, df1, df2)
    if F.size == 1:
        return float(F[0]), float(p[0])
    return F, p


def first_stage_lines(F, pval):
    """Lines describing the first-stage F for an estimator's summary() output."""
    if F is None:
        return []
    F = np.atleast_1d(F)
    p = None if pval is None else np.atleast_1d(pval)
    lines = ["-" * 80]
    for j in range(F.size):
        tag = "" if F.size == 1 else f" (endogenous regressor {j + 1})"
        lines.append(f"First-stage F-statistic{tag}: {F[j]:.6f}")
        if p is not None:
            lines.append(f"First-stage F p-value{tag}: {p[j]:.6f}")
    if np.min(F) < 10:
        lines.append("WARNING: First-stage F < 10, indicating potentially weak instruments")
    return lines


@contextlib.contextmanager
def talk_logging(logger, talk):
    """With talk=True, print the estimator's debug messages to stderr for the duration of the call.

    The handler is attached only inside the call and the logger's level and propagation are restored afterwards, so
    there is no output (and no leaked state) when talk=False, and no duplicate lines if the user has configured
    logging themselves.
    """
    if not talk:
        yield
        return
    handler = logging.StreamHandler()  # created now, so it writes to the current sys.stderr
    handler.setFormatter(logging.Formatter("%(message)s"))
    old_level, old_propagate = logger.level, logger.propagate
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    try:
        yield
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
        logger.propagate = old_propagate


def talk_option(logger):
    """Decorator: honour the keyword argument `talk` of an estimator by wrapping the call in talk_logging."""
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            with talk_logging(logger, kwargs.get("talk", False)):
                return f(*args, **kwargs)
        return wrapper
    return decorator
