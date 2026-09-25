"""The estimators against the literal definitions of Angrist, Imbens, Krueger (1999, section 3).

The estimators use a shortcut, (Z_i pi_hat - h_i X_i) / (1 - h_i) for JIVE1 and (Z_i pi_hat - h_i X_i) / (1 - 1/N) for
JIVE2, so that the leave-one-out first stage does not have to be run N times. This test runs it N times, exactly as the
paper defines it, on small data sets, and checks that the shortcut gives the same estimates.
"""
import numpy as np
import pytest

from high_dimensional_instrumental_variables.jive1 import JIVE1
from high_dimensional_instrumental_variables.jive2 import JIVE2
from high_dimensional_instrumental_variables.ujive1 import UJIVE1
from high_dimensional_instrumental_variables.ujive2 import UJIVE2
from tests.helpers import make_data


def literal_jackknife_regressors(Y, x, Z, W, kind):
    """X~ from N separate regressions. Returns X~ = [1, fit, W] and X = [1, x, W]."""
    n = len(Y)
    ones = np.ones((n, 1))
    controls = np.zeros((n, 0)) if W is None else W
    Zf, X = np.hstack([ones, Z, controls]), np.column_stack([ones, x, controls])
    ZtZ_inv, ZtX = np.linalg.inv(Zf.T @ Zf), Zf.T @ X
    fit = np.empty(n)
    for i in range(n):
        if kind == "jive1":     # pi(i) = (Z(i)'Z(i))^-1 Z(i)'X(i): refit without observation i (paper eq. 4)
            Zo, Xo = np.delete(Zf, i, axis=0), np.delete(X, i, axis=0)
            pi_i = np.linalg.solve(Zo.T @ Zo, Zo.T @ Xo)
        else:                   # JIVE2: pi(i) = (Z'Z)^-1 (Z'X - Z_i'X_i) * N / (N - 1)   (paper eq. 6)
            pi_i = ZtZ_inv @ (ZtX - np.outer(Zf[i], X[i])) * (n / (n - 1))
        fit[i] = (Zf[i] @ pi_i)[1]
    return np.column_stack([ones, fit, controls]), X


@pytest.mark.parametrize("n, k, n_controls", [(60, 5, 0), (60, 5, 2), (100, 12, 2)])
@pytest.mark.parametrize("estimator, kind, iv_form", [(JIVE1, "jive1", False), (UJIVE1, "jive1", True),
                                                       (JIVE2, "jive2", False), (UJIVE2, "jive2", True)],
                         ids=["JIVE1", "UJIVE1", "JIVE2", "UJIVE2"])
def test_estimates_equal_the_literal_leave_one_out_definition(estimator, kind, iv_form, n, k, n_controls):
    Y, x, Z, W = make_data(n, k, seed=n + k, n_controls=n_controls)
    X_tilde, X = literal_jackknife_regressors(Y, x, Z, W, kind)
    if iv_form:     # beta = (X~'X)^-1 X~'Y  (UJIVE1 / UJIVE2, the AIK estimators)
        beta = np.linalg.solve(X_tilde.T @ X, X_tilde.T @ Y)
    else:           # beta = (X~'X~)^-1 X~'Y  (JIVE1 / JIVE2 in Stata's naming)
        beta = np.linalg.lstsq(X_tilde, Y, rcond=None)[0]
    np.testing.assert_allclose(estimator(Y, x, Z, W=W).beta, beta, rtol=1e-9, atol=1e-11)
