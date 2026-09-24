"""Regression tests for the silent-wrong-answer bugs: positional controls, near-constant columns,
first-stage F with controls, and missing / unusable input."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import f as f_dist

from high_dimensional_instrumental_variables.jive1 import JIVE1
from high_dimensional_instrumental_variables.jive2 import JIVE2
from high_dimensional_instrumental_variables.ujive1 import UJIVE1
from high_dimensional_instrumental_variables.ujive2 import UJIVE2
from tests.helpers import make_data

ESTIMATORS = [JIVE1, JIVE2, UJIVE1, UJIVE2]
ids = [f.__name__ for f in ESTIMATORS]


@pytest.mark.parametrize("f", ESTIMATORS, ids=ids)
def test_controls_can_be_passed_positionally(f):
    Y, x, Z, W = make_data(200, 6, n_controls=2)
    by_position, by_keyword = f(Y, x, Z, W), f(Y, x, Z, W=W)
    assert by_position.beta.shape == (4,)  # constant + endogenous + 2 controls
    np.testing.assert_allclose(by_position.beta, by_keyword.beta)


@pytest.mark.parametrize("f", ESTIMATORS, ids=ids)
def test_unused_G_argument_is_gone(f):
    Y, x, Z, W = make_data(100, 4, n_controls=1)
    with pytest.raises(TypeError):
        f(Y, x, Z, G=W)


@pytest.mark.parametrize("f", ESTIMATORS, ids=ids)
def test_large_mean_small_variance_instrument_is_kept(f):
    Y, x, Z, _ = make_data(300, 8)
    z_big = 1e6 + np.random.default_rng(3).normal(0, 0.5, 300)
    kept = f(Y, x, np.column_stack([Z, z_big]))
    centred = f(Y, x, np.column_stack([Z, z_big - z_big.mean()]))  # same column space, so same leverage
    assert kept.leverage.sum() == pytest.approx(10.0)  # constant + 8 instruments + the new one
    np.testing.assert_allclose(kept.leverage, centred.leverage, atol=1e-6)


@pytest.mark.parametrize("f", ESTIMATORS, ids=ids)
def test_constant_columns_are_still_dropped(f):
    Y, x, Z, W = make_data(150, 5, n_controls=2)
    base = f(Y, x, Z, W=W)
    with_const = f(Y, x, np.column_stack([Z, np.full(150, 7.0)]), W=np.column_stack([W, np.ones(150)]))
    np.testing.assert_allclose(with_const.beta, base.beta)


@pytest.mark.parametrize("f", ESTIMATORS, ids=ids)
@pytest.mark.parametrize("where", ["Y", "X", "Z", "W"])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_missing_or_infinite_values_raise(f, where, bad):
    Y, x, Z, W = make_data(100, 4, n_controls=1)
    args = {"Y": Y.copy(), "X": x.copy(), "Z": Z.copy(), "W": W.copy()}
    args[where][(5,) + (0,) * (args[where].ndim - 1)] = bad  # poison exactly one element
    with pytest.raises(ValueError, match=f"{where} contains 1 missing or infinite"):
        f(args["Y"], args["X"], args["Z"], W=args["W"])


@pytest.mark.parametrize("f", ESTIMATORS, ids=ids)
def test_fewer_instruments_than_endogenous_regressors_raises(f):
    Y, X, Z, _ = make_data(200, 4, n_endog=2)
    with pytest.raises(ValueError, match="at least as many excluded instruments"):
        f(Y, X, Z[:, :1])


@pytest.mark.parametrize("f", ESTIMATORS, ids=ids)
def test_lists_are_accepted(f):
    Y, x, Z, W = make_data(100, 4, n_controls=1)
    from_lists = f(list(Y), list(x), Z.tolist(), W=W.tolist())
    np.testing.assert_allclose(from_lists.beta, f(Y, x, Z, W=W).beta)


def partial_f(endog, Z, W=None):
    """Independent first-stage F: instruments jointly, given the constant and controls."""
    N = len(endog)
    restricted = np.column_stack([np.ones(N)] + ([W] if W is not None else []))
    full = np.column_stack([restricted, Z])
    rss = lambda A: float(((endog - A @ np.linalg.lstsq(A, endog, rcond=None)[0]) ** 2).sum())
    df1, df2 = full.shape[1] - restricted.shape[1], N - full.shape[1]
    F = ((rss(restricted) - rss(full)) / df1) / (rss(full) / df2)
    return F, f_dist.sf(F, df1, df2)


@pytest.mark.parametrize("f", ESTIMATORS, ids=ids)
@pytest.mark.parametrize("n_controls", [0, 2])
def test_first_stage_f_is_the_partial_f_for_the_instruments(f, n_controls):
    Y, x, Z, W = make_data(300, 8, n_controls=n_controls)
    res = f(Y, x, Z, W=W)
    F, p = partial_f(x, Z, W)
    assert res.first_stage_f == pytest.approx(F, rel=1e-8)
    assert res.first_stage_f_pval == pytest.approx(p, rel=1e-6, abs=1e-12)


@pytest.mark.parametrize("f", ESTIMATORS, ids=ids)
def test_first_stage_f_one_per_endogenous_regressor(f):
    Y, X, Z, W = make_data(300, 8, n_endog=2, n_controls=1)
    res = f(Y, X, Z, W=W)
    expected = [partial_f(X[:, j], Z, W)[0] for j in range(2)]
    np.testing.assert_allclose(res.first_stage_f, expected, rtol=1e-8)


def test_summary_reports_first_stage_f(capsys):
    Y, x, Z, W = make_data(200, 5, n_controls=1)
    for f in ESTIMATORS:
        f(Y, x, Z, W=W).summary()
    assert capsys.readouterr().out.count("First-stage F-statistic") == 4


# ---- talk=True logging ----------------------------------------------------------------------------------------
import logging

from high_dimensional_instrumental_variables import jive1 as jive1_module


@pytest.mark.parametrize("f", ESTIMATORS, ids=ids)
def test_talk_prints_step_by_step_output_once(f, capsys):
    Y, x, Z, _ = make_data(100, 4)
    f(Y, x, Z, talk=True)
    assert capsys.readouterr().err.count("Fitted values obtained") == 1


@pytest.mark.parametrize("f", ESTIMATORS, ids=ids)
def test_no_output_without_talk(f, capsys):
    Y, x, Z, _ = make_data(100, 4)
    f(Y, x, Z)
    captured = capsys.readouterr()
    assert captured.err == "" and captured.out == ""


def test_talk_leaves_logger_state_untouched_and_does_not_duplicate_with_root_logging(capsys):
    lg = jive1_module.logger
    before = (lg.level, lg.propagate, list(lg.handlers))
    Y, x, Z, _ = make_data(100, 4)
    root_handler = logging.StreamHandler()  # what logging.basicConfig() would set up in a user's script
    logging.getLogger().addHandler(root_handler)
    try:
        jive1_module.JIVE1(Y, x, Z, talk=True)
    finally:
        logging.getLogger().removeHandler(root_handler)
    assert capsys.readouterr().err.count("Fitted values obtained") == 1  # not printed a second time by the root handler
    assert (lg.level, lg.propagate, list(lg.handlers)) == before
