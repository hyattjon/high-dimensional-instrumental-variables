import numpy as np
import pandas as pd
import pytest

from high_dimensional_instrumental_variables.jive2 import JIVE2
from tests.helpers import make_data, reference


@pytest.mark.parametrize("n, k, n_endog, n_controls", [(200, 10, 1, 0), (200, 10, 1, 2), (200, 8, 2, 0), (40, 5, 1, 0)])
def test_matches_explicit_projection_matrix(n, k, n_endog, n_controls):
    Y, x, Z, W = make_data(n, k, n_endog=n_endog, n_controls=n_controls)
    res = JIVE2(Y, x, Z, W=W)
    beta, robust_v, leverage = reference("JIVE2", Y, x, Z, W)
    np.testing.assert_allclose(res.beta, beta, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(res.vcov, robust_v, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(res.leverage, leverage, rtol=1e-8, atol=1e-12)


def test_output_shapes_and_inference():
    Y, x, Z, W = make_data(100, 5, n_controls=2)
    res = JIVE2(Y, x, Z, W=W)
    n_coef = 1 + 1 + 2  # constant + endogenous + controls
    assert res.beta.shape == (n_coef,)
    assert res.vcov.shape == (n_coef, n_coef)
    assert res.leverage.shape == (100, 1)
    assert len(res.pvals) == len(res.tstats) == len(res.cis) == n_coef
    assert np.all(np.isfinite(res.beta)) and np.isfinite(res.f_stat)
    assert 0 < res.leverage.min() and res.leverage.max() < 1


def test_small_sample():
    Y, x, Z, _ = make_data(12, 3)
    res = JIVE2(Y, x, Z)
    assert np.all(np.isfinite(res.beta))
    assert np.all(np.isfinite(res.se))


def test_pandas_input_and_summary(capsys):
    Y, x, Z, W = make_data(100, 5, n_controls=1)
    res = JIVE2(pd.Series(Y), pd.Series(x), pd.DataFrame(Z), W=pd.DataFrame(W))
    np.testing.assert_allclose(res.beta, JIVE2(Y, x, Z, W=W).beta)
    res.summary()
    assert "JIVE2 Regression Results" in capsys.readouterr().out


def test_too_few_observations():
    Y, x, Z, _ = make_data(6, 5)  # constant + 5 instruments = 6 columns = N
    with pytest.raises(ValueError, match="larger than the number of columns"):
        JIVE2(Y, x, Z)


def test_rank_deficient_instruments():
    Y, x, Z, _ = make_data(50, 4)
    Z = np.column_stack([Z, Z[:, 0]])  # duplicated instrument
    with pytest.raises(ValueError, match="rank deficient"):
        JIVE2(Y, x, Z)


def test_large_n_does_not_build_projection_matrix():
    # An N x N float64 matrix at N = 50,000 would be 20 GB, so this only passes if we never form it.
    Y, x, Z, _ = make_data(50_000, 20)
    res = JIVE2(Y, x, Z)
    assert abs(res.beta[1] - 2) < 0.5
