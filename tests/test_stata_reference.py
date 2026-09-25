"""Our estimators against Stata's jive command (Poi 2006, Stata Journal package st0108).

The data and Stata's results are committed under validation/ (Stata 19 MP; regenerate with validation/run_stata.do),
so this test does not need Stata.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from high_dimensional_instrumental_variables.jive1 import JIVE1
from high_dimensional_instrumental_variables.jive2 import JIVE2
from high_dimensional_instrumental_variables.ujive1 import UJIVE1
from high_dimensional_instrumental_variables.ujive2 import UJIVE2

VALIDATION = Path(__file__).resolve().parents[1] / "validation"
STATA = pd.read_csv(VALIDATION / "results_stata.csv").set_index(["dataset", "source"])
DATASETS = ["n300_k10_nocontrols", "n300_k10_controls", "n60_k8_controls"]
ESTIMATORS = {"jive1": JIVE1, "jive2": JIVE2, "ujive1": UJIVE1, "ujive2": UJIVE2}


def ours(estimator, dataset):
    df = pd.read_csv(VALIDATION / "data" / f"{dataset}.csv")
    W = df[["w1", "w2"]] if "w1" in df else None
    res = ESTIMATORS[estimator](df["y"], df["t"], df.filter(regex=r"^z\d+$"), W=W)
    return res.beta[1], res.se[1]


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("option", ESTIMATORS)
def test_coefficient_equals_stata_jive(option, dataset):
    beta, _ = ours(option, dataset)
    np.testing.assert_allclose(beta, STATA.loc[(dataset, f"Stata jive, {option}")].beta, rtol=1e-6)


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("option", ["ujive1", "ujive2"])
def test_ujive_robust_standard_error_equals_stata(option, dataset):
    _, se = ours(option, dataset)
    np.testing.assert_allclose(se, STATA.loc[(dataset, f"Stata jive, {option} robust")].se, rtol=1e-6)


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("option", ["jive1", "jive2"])
def test_jive_standard_error_is_close_to_but_not_equal_to_stata(option, dataset):
    """Ours differ from Stata's jive1/jive2 standard errors on purpose: Stata's jive.ado has a bug in CalcJIVE.

    We use the intended residual Y - X b. The test at the bottom of this file reproduces Stata's numbers from the bug.
    This one only pins down how far apart the two are, so an unrelated change that moves ours is noticed.
    """
    _, se = ours(option, dataset)
    stata = STATA.loc[(dataset, f"Stata jive, {option} robust")].se
    assert abs(se - stata) / stata < 0.05


def stata_jive_standard_errors(estimator, dataset):
    """Reproduce Stata's jive1 / jive2 standard errors, bug included (see validation/README.md).

    Poi's jive.ado names the coefficient columns with a macro (`one`) that its CalcJIVE routine never defines, so the
    constant's coefficient is attached to the last regressor when the residual is formed: e = Y - X b with
    b_last += b_constant, instead of e = Y - X b with the constant as an intercept. Default and robust standard errors
    are then computed from that residual.
    """
    df = pd.read_csv(VALIDATION / "data" / f"{dataset}.csv")
    W = df[["w1", "w2"]].values if "w1" in df else np.empty((len(df), 0))
    Y, T = df["y"].values, df["t"].values
    res = ESTIMATORS[estimator](df["y"], df["t"], df.filter(regex=r"^z\d+$"), W=W if W.size else None)
    n, k = len(Y), len(res.beta)
    h = res.leverage[:, 0]
    denominator = (1 - h) if estimator == "jive1" else (1 - 1 / n)
    X_tilde = np.column_stack([(res.fitted_values[:, 0] - h * T) / denominator, W, np.ones(n)])  # Stata orders [T, controls, constant]
    b = np.concatenate([[res.beta[1]], res.beta[2:], [res.beta[0]]])
    regressors = np.column_stack([T, W])
    coef = b[:-1].copy()
    coef[-1] += b[-1]                                  # the bug: the constant's coefficient lands on the last regressor
    e = Y - regressors @ coef
    bread = np.linalg.inv(X_tilde.T @ X_tilde)
    sigsq = np.var(e, ddof=1) * (n - 1) / (n - k)
    default = np.sqrt(sigsq * bread[0, 0])
    robust = np.sqrt((bread @ ((X_tilde * (e ** 2)[:, None]).T @ X_tilde) @ bread)[0, 0])
    return default, robust


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("option", ["jive1", "jive2"])
def test_stata_jive_standard_errors_are_explained_by_the_jive_ado_bug(option, dataset):
    default, robust = stata_jive_standard_errors(option, dataset)
    np.testing.assert_allclose(default, STATA.loc[(dataset, f"Stata jive, {option}")].se, rtol=1e-6)
    np.testing.assert_allclose(robust, STATA.loc[(dataset, f"Stata jive, {option} robust")].se, rtol=1e-6)
