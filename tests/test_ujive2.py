import numpy as np
import pandas as pd
import pytest

from high_dimensional_instrumental_variables.ujive2 import UJIVE2
from tests.helpers import make_data, reference


@pytest.mark.parametrize("n, k, n_endog, n_controls", [(200, 10, 1, 0), (200, 10, 1, 2), (200, 8, 2, 0), (40, 5, 1, 0)])
def test_matches_explicit_projection_matrix(n, k, n_endog, n_controls):
    Y, x, Z, W = make_data(n, k, n_endog=n_endog, n_controls=n_controls)
    res = UJIVE2(Y, x, Z, W=W)
    beta, robust_v, leverage = reference("UJIVE2", Y, x, Z, W)
    np.testing.assert_allclose(res.beta, beta, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(res.standard_errors, robust_v, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(res.leverage, leverage, rtol=1e-8, atol=1e-12)


def test_output_shapes_and_inference():
    Y, x, Z, W = make_data(100, 5, n_controls=2)
    res = UJIVE2(Y, x, Z, W=W)
    n_coef = 1 + 1 + 2  # constant + endogenous + controls
    assert res.beta.shape == (n_coef,)
    assert res.standard_errors.shape == (n_coef, n_coef)
    assert res.leverage.shape == (100, 1)
    assert len(res.pvals) == len(res.tstats) == len(res.cis) == n_coef
    assert np.all(np.isfinite(res.beta)) and np.isfinite(res.f_stat)
    assert 0 < res.leverage.min() and res.leverage.max() < 1


def test_small_sample():
    Y, x, Z, _ = make_data(12, 3)
    res = UJIVE2(Y, x, Z)
    assert np.all(np.isfinite(res.beta))
    assert np.all(np.isfinite(np.diag(res.standard_errors)))


def test_pandas_input_and_summary(capsys):
    Y, x, Z, W = make_data(100, 5, n_controls=1)
    res = UJIVE2(pd.Series(Y), pd.Series(x), pd.DataFrame(Z), W=pd.DataFrame(W))
    np.testing.assert_allclose(res.beta, UJIVE2(Y, x, Z, W=W).beta)
    res.summary()
    assert "UJIVE2 Regression Results" in capsys.readouterr().out


def test_too_few_observations():
    Y, x, Z, _ = make_data(6, 5)  # constant + 5 instruments = 6 columns = N
    with pytest.raises(ValueError, match="larger than the number of columns"):
        UJIVE2(Y, x, Z)


def test_rank_deficient_instruments():
    Y, x, Z, _ = make_data(50, 4)
    Z = np.column_stack([Z, Z[:, 0]])  # duplicated instrument
    with pytest.raises(ValueError, match="rank deficient"):
        UJIVE2(Y, x, Z)


def test_large_n_does_not_build_projection_matrix():
    # An N x N float64 matrix at N = 50,000 would be 20 GB, so this only passes if we never form it.
    Y, x, Z, _ = make_data(50_000, 20)
    res = UJIVE2(Y, x, Z)
    assert abs(res.beta[1] - 2) < 0.5
