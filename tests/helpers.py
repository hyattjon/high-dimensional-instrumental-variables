"""Shared helpers for the JIVE / UJIVE tests.

`reference` is a deliberately naive implementation that builds the N x N projection matrix and uses
explicit inverses. The estimators in the package must agree with it (they use QR / solve instead).
"""
import numpy as np


def make_data(n, k, seed=0, n_endog=1, n_controls=0):
    """Simulate an IV data set with an endogenous regressor, k instruments and optional controls."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, k))
    W = rng.standard_normal((n, n_controls)) if n_controls else None
    e1 = rng.normal(0, 3, n)
    e2 = rng.normal(0, 3, n)
    x = np.column_stack([Z[:, j] * 0.6 + 0.5 * e1 for j in range(n_endog)])
    if W is not None:
        x = x + W[:, [0]]
    Y = 1 + x @ np.arange(2, 2 + n_endog) + e1 + e2
    if W is not None:
        Y = Y + W.sum(axis=1)
    return Y, (x[:, 0] if n_endog == 1 else x), Z, W


def reference(kind, Y, x, Z, W=None):
    """Return (beta, robust_v, leverage) for kind in {"JIVE1", "JIVE2", "UJIVE1", "UJIVE2"}."""
    N = len(Y)
    endog = np.asarray(x).reshape(N, -1)
    k = endog.shape[1]
    ones = np.ones((N, 1))
    X = np.hstack([ones, endog] + ([W] if W is not None else []))
    Zf = np.hstack([ones, Z] + ([W] if W is not None else []))

    P = Zf @ np.linalg.inv(Zf.T @ Zf) @ Zf.T
    h = np.diag(P)[:, None]
    fit = (P @ X)[:, 1:1 + k]
    Xe = X[:, 1:1 + k]
    denom = (1 - h) if kind in ("JIVE1", "UJIVE1") else (1 - 1 / N)
    Xj = np.hstack([ones, (fit - h * Xe) / denom] + ([W] if W is not None else []))

    # The JIVE estimators regress Y on the jackknife fit; the UJIVE estimators use it as an instrument for X.
    B = Xj.T @ Xj if kind in ("JIVE1", "JIVE2") else Xj.T @ X
    Binv = np.linalg.inv(B)
    beta = Binv @ Xj.T @ Y
    resid = Y - X @ beta
    meat = sum(resid[i] ** 2 * np.outer(Xj[i], Xj[i]) for i in range(N))
    return beta, Binv @ meat @ Binv.T, h
