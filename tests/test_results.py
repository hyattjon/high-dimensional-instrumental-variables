"""The result object shared by JIVE1, JIVE2, UJIVE1 and UJIVE2."""
import warnings

import numpy as np
import pytest
from scipy.stats import f as f_dist
from scipy.stats import t as t_dist

from high_dimensional_instrumental_variables._results import IVResult
from high_dimensional_instrumental_variables.jive1 import JIVE1, JIVE1Result
from high_dimensional_instrumental_variables.jive2 import JIVE2, JIVE2Result
from high_dimensional_instrumental_variables.ujive1 import UJIVE1, UJIVE1Result
from high_dimensional_instrumental_variables.ujive2 import UJIVE2, UJIVE2Result
from tests.helpers import make_data, reference

CASES = [(JIVE1, JIVE1Result), (JIVE2, JIVE2Result), (UJIVE1, UJIVE1Result), (UJIVE2, UJIVE2Result)]
ids = [f.__name__ for f, _ in CASES]


@pytest.mark.parametrize("f, cls", CASES, ids=ids)
def test_returns_the_estimators_own_result_class(f, cls):
    Y, x, Z, W = make_data(150, 5, n_controls=1)
    res = f(Y, x, Z, W=W)
    assert type(res) is cls and isinstance(res, IVResult)
    assert res.name == f.__name__


@pytest.mark.parametrize("f, cls", CASES, ids=ids)
def test_se_is_the_vector_of_standard_errors(f, cls):
    Y, x, Z, W = make_data(200, 6, n_controls=2)
    res = f(Y, x, Z, W=W)
    _, ref_v, _ = reference(f.__name__, Y, x, Z, W)
    assert res.se.shape == (4,)
    np.testing.assert_allclose(res.se, np.sqrt(np.diag(ref_v)), rtol=1e-8)
    np.testing.assert_allclose(res.se, np.sqrt(np.diag(res.vcov)))


@pytest.mark.parametrize("f, cls", CASES, ids=ids)
def test_standard_errors_is_a_deprecated_alias_of_vcov(f, cls):
    Y, x, Z, _ = make_data(100, 4)
    res = f(Y, x, Z)
    with pytest.warns(DeprecationWarning, match="covariance matrix"):
        old = res.standard_errors
    np.testing.assert_array_equal(old, res.vcov)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res.vcov, res.se  # the new names do not warn


def test_all_estimators_return_the_same_types_and_fields():
    Y, x, Z, W = make_data(200, 6, n_controls=2)
    results = [f(Y, x, Z, W=W) for f, _ in CASES]
    for res in results:
        assert isinstance(res.pvals, np.ndarray) and res.pvals.shape == (4,)
        assert isinstance(res.tstats, np.ndarray) and res.tstats.shape == (4,)
        assert isinstance(res.cis, np.ndarray) and res.cis.shape == (4, 2)
        assert all(isinstance(getattr(res, a), float) for a in ("root_mse", "r_squared", "adjusted_r_squared", "f_stat", "f_pval"))
    assert len({tuple(sorted(vars(r))) for r in results}) == 1


@pytest.mark.parametrize("f, cls", CASES, ids=ids)
def test_inference_uses_t_distribution_with_n_minus_p_dof(f, cls):
    Y, x, Z, W = make_data(80, 5, n_controls=1)
    res = f(Y, x, Z, W=W)
    dof = 80 - 3
    np.testing.assert_allclose(res.tstats, res.beta / res.se)
    np.testing.assert_allclose(res.pvals, 2 * t_dist.sf(np.abs(res.tstats), dof))
    crit = t_dist.ppf(0.975, dof)
    np.testing.assert_allclose(res.cis, np.column_stack([res.beta - crit * res.se, res.beta + crit * res.se]))


@pytest.mark.parametrize("f, cls", CASES, ids=ids)
def test_f_stat_is_the_robust_wald_test_of_all_slopes(f, cls):
    Y, x, Z, W = make_data(200, 6, n_controls=2)
    res = f(Y, x, Z, W=W)
    b, V = res.beta[1:], res.vcov[1:, 1:]
    wald = float(b @ np.linalg.inv(V) @ b)
    assert res.f_stat == pytest.approx(wald / 3, rel=1e-8)
    assert res.f_pval == pytest.approx(f_dist.sf(res.f_stat, 3, 200 - 4), rel=1e-6)


@pytest.mark.parametrize("f, cls", CASES, ids=ids)
def test_with_one_slope_the_wald_f_is_the_squared_t_statistic(f, cls):
    Y, x, Z, _ = make_data(150, 5)
    res = f(Y, x, Z)
    assert res.f_stat == pytest.approx(res.tstats[1] ** 2, rel=1e-8)


@pytest.mark.parametrize("f, cls", CASES, ids=ids)
def test_dictionary_access(f, cls):
    Y, x, Z, _ = make_data(100, 4)
    res = f(Y, x, Z)
    np.testing.assert_array_equal(res["beta"], res.beta)
    np.testing.assert_array_equal(res["se"], res.se)
    np.testing.assert_array_equal(res["first_stage_f"], res.first_stage_f)
    with pytest.raises(KeyError, match="Invalid key 'nope'"):
        res["nope"]


def test_repr_is_short_even_for_large_n():
    Y, x, Z, _ = make_data(2000, 5)
    assert len(repr(UJIVE1(Y, x, Z))) < 1500  # leverage and fitted values are not printed


def test_summary_layout(capsys):
    Y, x, Z, W = make_data(150, 5, n_controls=1)
    UJIVE2(Y, x, Z, W=W).summary()
    out = capsys.readouterr().out
    for text in ("UJIVE2 Regression Results", "Std. Error", "Robust Wald F-statistic", "Root MSE", "First-stage F-statistic"):
        assert text in out
