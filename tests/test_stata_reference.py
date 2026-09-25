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
    """Known gap: Stata's jive1/jive2 use a residual scale we could not reproduce (see README, section on naming).

    This pins down how far off we are, so a change that makes it worse is noticed. If the formula is ever matched,
    tighten this to assert equality and move `jive1` / `jive2` up into the test above.
    """
    _, se = ours(option, dataset)
    stata = STATA.loc[(dataset, f"Stata jive, {option} robust")].se
    assert abs(se - stata) / stata < 0.05
