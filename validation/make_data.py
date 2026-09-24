"""Write the shared simulated data sets used to compare our estimators with R and Stata.

Each CSV has columns y, t (endogenous regressor), z1..zK (instruments) and, where present, w1, w2 (controls).
Run from anywhere: python validation/make_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "data"

# name: (N, K instruments, controls?)
CASES = {
    "n300_k10_nocontrols": (300, 10, False),
    "n300_k10_controls": (300, 10, True),
    "n60_k8_controls": (60, 8, True),
}


def make(n, k, controls, seed):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, k))
    W = rng.standard_normal((n, 2)) if controls else None
    e1 = rng.normal(0, 3, n)
    e2 = rng.normal(0, 3, n)
    t = Z @ np.full(k, 0.4) + 0.5 * e1 + (W[:, 0] if controls else 0)
    y = 1 + 2 * t + e1 + e2 + (W.sum(axis=1) if controls else 0)
    df = pd.DataFrame({"y": y, "t": t})
    for j in range(k):
        df[f"z{j + 1}"] = Z[:, j]
    if controls:
        df["w1"], df["w2"] = W[:, 0], W[:, 1]
    return df


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for i, (name, (n, k, controls)) in enumerate(CASES.items()):
        make(n, k, controls, seed=100 + i).to_csv(OUT / f"{name}.csv", index=False)
        print("wrote", OUT / f"{name}.csv")
