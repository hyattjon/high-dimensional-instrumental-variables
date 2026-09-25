"""Our UJIVE1 must reproduce Kyle Butts' R package `jive` (its jive() function).

The data and the R results are committed under validation/ (regenerate with validation/make_data.py and
validation/run_r.R), so this test does not need R. R's jive() is the Angrist, Imbens, Krueger (1999) JIVE1 with
the controls partialled out, which is what this package calls UJIVE1 (Stata's jive, ujive1 naming).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from high_dimensional_instrumental_variables.ujive1 import UJIVE1

VALIDATION = Path(__file__).resolve().parents[1] / "validation"
R_RESULTS = pd.read_csv(VALIDATION / "results_r.csv")
R_JIVE = R_RESULTS[R_RESULTS["source"] == "R jive()"]


@pytest.mark.parametrize("row", list(R_JIVE.itertuples()), ids=lambda r: r.dataset)
def test_ujive1_matches_r_jive(row):
    df = pd.read_csv(VALIDATION / "data" / f"{row.dataset}.csv")
    W = df[["w1", "w2"]] if "w1" in df else None
    res = UJIVE1(df["y"], df["t"], df.filter(regex=r"^z\d+$"), W=W)
    np.testing.assert_allclose(res.beta[1], row.beta, rtol=1e-6)
    np.testing.assert_allclose(res.se[1], row.se, rtol=1e-6)
