"""Tests for the building blocks of simulation/monte_carlo.py (the design, the estimators, the summaries, the CLI).

None of these runs the simulation study itself.
"""
import numpy as np
import pytest

from simulation import monte_carlo as mc
from tests.helpers import make_data


def test_simulate_shapes_and_reproducibility():
    cell = mc.Cell(n=100, k=7, mu2=70.0, rho=0.5, controls=2, hetero=False)
    y, x, Z, W = mc.simulate(cell, np.random.default_rng(1))
    assert y.shape == x.shape == (100,) and Z.shape == (100, 7) and W.shape == (100, 2)
    again = mc.simulate(cell, np.random.default_rng(1))
    for a, b in zip((y, x, Z, W), again):
        np.testing.assert_array_equal(a, b)
    assert mc.simulate(mc.Cell(50, 3, 15.0, 0.5, 0, False), np.random.default_rng(1))[3] is None


def test_first_stage_strength_is_calibrated_to_the_target_f():
    from high_dimensional_instrumental_variables.ujive1 import UJIVE1
    cell = mc.Cell(n=2000, k=10, mu2=100.0, rho=0.5, controls=0, hetero=False)  # expected F = 1 + 100 / 10 = 11
    fs = []
    for r in range(20):
        y, x, Z, W = mc.simulate(cell, np.random.default_rng([7, r]))
        fs.append(UJIVE1(y, x, Z, W=W).first_stage_f)
    assert 9.5 < np.mean(fs) < 12.5  # expected first-stage F is about the target (sd of the mean is about 0.5)


def test_endogeneity_moves_ols_by_about_rho():
    # The OLS bias is cov(x, eps) / var(x) = rho / (1 + mu2 / n), about rho here because mu2 / n is tiny
    cell = mc.Cell(n=40000, k=5, mu2=50.0, rho=0.5, controls=0, hetero=False)
    y, x, Z, W = mc.simulate(cell, np.random.default_rng(3))
    assert abs(mc.fit_ols(y, x, Z, W).beta - (mc.BETA0 + 0.5)) < 0.03


def test_heteroskedastic_error_keeps_unit_variance_but_depends_on_the_first_instrument():
    cell = mc.Cell(n=200000, k=3, mu2=3.0, rho=0.0, controls=0, hetero=True)
    y, x, Z, _ = mc.simulate(cell, np.random.default_rng(5))
    eps = y - 1.0 - mc.BETA0 * x
    assert abs(eps.var() - 1.0) < 0.02
    assert abs(np.corrcoef(eps ** 2, Z[:, 0] ** 2)[0, 1]) > 0.1


def test_ols_and_2sls_match_explicit_formulas():
    y, x, Z, W = make_data(150, 6, n_controls=2)
    X = np.column_stack([np.ones(150), x, W])
    Zf = np.column_stack([np.ones(150), Z, W])
    ols = mc.fit_ols(y, x, Z, W)
    np.testing.assert_allclose(ols.beta, np.linalg.lstsq(X, y, rcond=None)[0][1])
    # 2SLS with the projection matrix written out, including the robust sandwich and the t interval
    P = Zf @ np.linalg.inv(Zf.T @ Zf) @ Zf.T
    Xh = P @ X
    b = np.linalg.inv(Xh.T @ X) @ Xh.T @ y
    e = y - X @ b
    bread = np.linalg.inv(Xh.T @ X)
    V = bread @ (Xh.T * e ** 2) @ Xh @ bread.T
    tsls = mc.fit_2sls(y, x, Z, W)
    np.testing.assert_allclose(tsls.beta, b[1], rtol=1e-8)
    np.testing.assert_allclose(tsls.se, np.sqrt(V[1, 1]), rtol=1e-8)
    assert tsls.lo < tsls.beta < tsls.hi


def test_fit_all_returns_every_estimator_and_records_failures_as_nan():
    y, x, Z, W = make_data(100, 4)
    fits = mc.fit_all(y, x, Z, W)
    assert list(fits) == mc.ESTIMATORS and all(np.isfinite(f.beta) for f in fits.values())
    fits = mc.fit_all(y, x, np.column_stack([Z, Z[:, 0]]), W)  # duplicated instrument: the jackknife estimators must raise
    assert all(np.isnan(fits[name].beta) for name in ("JIVE1", "JIVE2", "UJIVE1", "UJIVE2"))


def test_summarize_known_values():
    est = np.array([0.5, 1.0, 1.5, 2.0])
    lo, hi = est - 0.6, est + 0.6           # intervals contain 1.0 for est = 0.5, 1.0, 1.5 but not for 2.0
    s = mc.summarize(est, np.full(4, 0.3), lo, hi)
    assert s["n_failed"] == 0 and s["median_bias"] == pytest.approx(0.25) and s["mean_bias"] == pytest.approx(0.25)
    assert s["mad"] == pytest.approx(0.5) and s["coverage"] == pytest.approx(0.75) and s["median_ci_width"] == pytest.approx(1.2)
    assert s["rmse"] == pytest.approx(np.sqrt((0.25 + 0 + 0.25 + 1) / 4))


def test_summarize_ignores_failed_replications_and_handles_all_failed():
    est = np.array([1.0, np.nan, 1.2, 0.8])
    s = mc.summarize(est, np.array([0.1, np.nan, 0.1, 0.1]), est - 0.3, est + 0.3)
    assert s["n_failed"] == 1 and s["n_reps"] == 4 and s["coverage"] == 1.0
    nothing = mc.summarize(np.full(3, np.nan), np.full(3, np.nan), np.full(3, np.nan), np.full(3, np.nan))
    assert nothing["n_failed"] == 3 and np.isnan(nothing["coverage"])


def test_cell_key_is_stable_and_depends_on_every_input():
    c = mc.Cell(500, 20, 5.0, 0.5, 2, False)
    base = mc.cell_key(3, c, 2000, 1)
    assert base == mc.cell_key(3, c, 2000, 1)
    assert len({base, mc.cell_key(4, c, 2000, 1), mc.cell_key(3, mc.Cell(500, 20, 5.0, 0.5, 2, True), 2000, 1),
                mc.cell_key(3, c, 1999, 1), mc.cell_key(3, c, 2000, 2)}) == 5


def test_dry_run_prints_the_design_and_simulates_nothing(tmp_path, capsys):
    out = tmp_path / "results"
    assert mc.main(["--dry-run", "--grid", "quick", "--reps", "10", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "cells x 10 replications x 6 estimators" in text and "nothing simulated" in text
    assert not out.exists()


@pytest.mark.parametrize("grid, expected", [("quick", 8), ("full", 24)])
def test_preset_grids_have_the_documented_number_of_cells(grid, expected):
    assert len(mc.build_cells(mc.GRIDS[grid])) == expected


def test_command_line_overrides_change_the_grid():
    args = mc.parse_args(["--grid", "full", "--k", "10", "20", "--mu2", "100", "--controls", "0", "--hetero", "no", "yes"])
    cells = mc.build_cells(mc.grid_from_args(args))
    assert len(cells) == 1 * 2 * 1 * 1 * 1 * 2 and {c.hetero for c in cells} == {False, True} and {c.k for c in cells} == {10, 20}
