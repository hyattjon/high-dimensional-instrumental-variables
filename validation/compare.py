"""Compare our estimators with the R (and, if present, Stata) results on the shared data sets.

Run from the repo root after `python validation/make_data.py` and `Rscript validation/run_r.R`
(and optionally `do validation/run_stata.do` in Stata, which writes validation/results_stata.csv):

    python validation/compare.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so it runs from a clone without installing

from high_dimensional_instrumental_variables.jive1 import JIVE1
from high_dimensional_instrumental_variables.jive2 import JIVE2
from high_dimensional_instrumental_variables.ujive1 import UJIVE1
from high_dimensional_instrumental_variables.ujive2 import UJIVE2

HERE = Path(__file__).parent
ESTIMATORS = {"JIVE1": JIVE1, "JIVE2": JIVE2, "UJIVE1": UJIVE1, "UJIVE2": UJIVE2}


def ours(name):
    df = pd.read_csv(HERE / "data" / f"{name}.csv")
    Z = df.filter(regex=r"^z\d+$")
    W = df[["w1", "w2"]] if "w1" in df else None
    rows = []
    for label, f in ESTIMATORS.items():
        r = f(df["y"], df["t"], Z, W=W)
        rows.append({"dataset": name, "source": f"ours {label}", "beta": r.beta[1], "se": np.sqrt(r.standard_errors[1, 1])})
    return rows


def main():
    rows = [row for name in sorted(p.stem for p in (HERE / "data").glob("*.csv")) for row in ours(name)]
    external = [pd.read_csv(p) for p in (HERE / "results_r.csv", HERE / "results_stata.csv") if p.exists()]
    table = pd.concat([pd.DataFrame(rows)] + external, ignore_index=True)
    pd.options.display.float_format = "{:.6f}".format
    for name, grp in table.groupby("dataset", sort=False):
        print(f"\n== {name}")
        print(grp[["source", "beta", "se"]].to_string(index=False))
    return table


if __name__ == "__main__":
    main()
