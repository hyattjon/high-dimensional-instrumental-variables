"""Monte Carlo study of the jackknife IV estimators: bias and confidence-interval coverage with many, possibly weak, instruments.

For every cell of a design grid (sample size, number of instruments K, instrument strength as a concentration parameter,
endogeneity, controls, heteroskedasticity) the script simulates `--reps` data sets and fits OLS, 2SLS, JIVE1, JIVE2, UJIVE1 and UJIVE2 to each.
It records, per cell and estimator, the bias, error, robust spread and 95% confidence-interval coverage of the coefficient
on the endogenous regressor. See simulation/README.md for the design and the hypotheses it is meant to test.

    python simulation/monte_carlo.py --dry-run                      # print the design and the amount of work; simulate nothing
    python simulation/monte_carlo.py --grid quick --reps 50         # small trial run (times the machine)
    python simulation/monte_carlo.py --grid full --reps 2000 --workers 4

Results are written to simulation/results/ (summary.csv, plus one small file per finished cell so an interrupted run resumes).
Then run `python simulation/report.py`.
"""
import argparse
import dataclasses
import hashlib
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# One BLAS thread per process: we parallelise over cells instead, so extra BLAS threads would only oversubscribe the CPU.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a clone without installing the package

from high_dimensional_instrumental_variables.jive1 import JIVE1
from high_dimensional_instrumental_variables.jive2 import JIVE2
from high_dimensional_instrumental_variables.ujive1 import UJIVE1
from high_dimensional_instrumental_variables.ujive2 import UJIVE2

HERE = Path(__file__).resolve().parent
BETA0 = 1.0  # true coefficient on the endogenous regressor
ESTIMATORS = ["OLS", "2SLS", "JIVE1", "JIVE2", "UJIVE1", "UJIVE2"]

# name -> axes of the design grid. Each axis is a list; the grid is their Cartesian product.
GRIDS = {
    "quick": dict(n=[200], k=[5, 20], mu2=[25.0, 100.0], rho=[0.5], controls=[0, 2], hetero=[False]),
    "full": dict(n=[500], k=[5, 20, 50, 100], mu2=[25.0, 100.0, 300.0], rho=[0.5], controls=[0, 2], hetero=[False]),
}


@dataclasses.dataclass(frozen=True)
class Cell:
    """One point of the design grid.

    n : sample size.  k : number of excluded instruments.  mu2 : concentration parameter, the total strength of the
    instruments (the first-stage F-statistic is about 1 + mu2 / k, so at fixed mu2 adding instruments dilutes them: the
    classic many-instruments design).  rho : correlation between the structural error and the first-stage error (endogeneity).
    controls : number of exogenous controls (0 or more, on top of the constant).  hetero : heteroskedastic structural error.
    """
    n: int
    k: int
    mu2: float
    rho: float
    controls: int
    hetero: bool


def build_cells(grid):
    axes = [grid[a] for a in ("n", "k", "mu2", "rho", "controls", "hetero")]
    return [Cell(*values) for values in itertools.product(*axes)]


# --------------------------------------------------------------------------------------------------------------------
# Data generating process
# --------------------------------------------------------------------------------------------------------------------
def simulate(cell, rng):
    """Draw one data set from the design.

    y = 1 + BETA0 * x + 0.5 * sum(W) + eps        x = 0.5 + c * sum(Z) + 0.5 * sum(W) + v

    Z (n x k) and W (n x controls) are standard normal. (eps, v) have unit variances and correlation `rho`. Every
    instrument has the same coefficient c = sqrt(mu2 / (n * k)), so the concentration parameter n * k * c^2 equals `mu2`
    and the expected first-stage F-statistic is about 1 + mu2 / k. With `hetero`, the structural error is scaled by
    sqrt((0.25 + Z_1^2) / 1.25), which keeps its variance at 1 but makes it depend on the first instrument.

    Returns y (n,), x (n,), Z (n, k) and W (n, controls) or None.
    """
    n, k = cell.n, cell.k
    Z = rng.standard_normal((n, k))
    W = rng.standard_normal((n, cell.controls)) if cell.controls else None
    v = rng.standard_normal(n)
    eps = cell.rho * v + np.sqrt(1 - cell.rho ** 2) * rng.standard_normal(n)
    if cell.hetero:
        eps = eps * np.sqrt((0.25 + Z[:, 0] ** 2) / 1.25)
    c = np.sqrt(cell.mu2 / (n * k))
    x = 0.5 + c * Z.sum(axis=1) + v
    y = 1.0 + BETA0 * x + eps
    if W is not None:
        x = x + 0.5 * W.sum(axis=1)
        y = y + 0.5 * W.sum(axis=1)
    return y, x, Z, W


# --------------------------------------------------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class Fit:
    beta: float
    se: float
    lo: float
    hi: float
    first_stage_f: float = np.nan


FAILED = Fit(np.nan, np.nan, np.nan, np.nan)


def _linear_iv(y, X, X_inst):
    """Coefficient on column 1 of X with instruments X_inst, and its robust (HC0 sandwich) 95% t-interval.

    OLS is X_inst = X; 2SLS is X_inst = the projection of X on the instruments. Same variance form as the jackknife
    estimators, so the comparison is about the point estimates and not about the variance formula.
    """
    A = X_inst.T @ X
    beta = np.linalg.solve(A, X_inst.T @ y)
    e = y - X @ beta
    meat = (X_inst * (e ** 2)[:, None]).T @ X_inst
    left = np.linalg.solve(A, meat)
    V = np.linalg.solve(A, left.T).T
    se = float(np.sqrt(V[1, 1]))
    crit = t_dist.ppf(0.975, len(y) - X.shape[1])
    return Fit(float(beta[1]), se, float(beta[1] - crit * se), float(beta[1] + crit * se))


def fit_ols(y, x, Z, W):
    X = np.column_stack([np.ones(len(y)), x] + ([W] if W is not None else []))
    return _linear_iv(y, X, X)


def fit_2sls(y, x, Z, W):
    n = len(y)
    X = np.column_stack([np.ones(n), x] + ([W] if W is not None else []))
    Q, _ = np.linalg.qr(np.column_stack([np.ones(n), Z] + ([W] if W is not None else [])), mode="reduced")
    return _linear_iv(y, X, Q @ (Q.T @ X))


def _fit_package(estimator):
    def fit(y, x, Z, W):
        r = estimator(y, x, Z, W=W)
        return Fit(float(r.beta[1]), float(r.se[1]), float(r.cis[1, 0]), float(r.cis[1, 1]), float(np.atleast_1d(r.first_stage_f)[0]))
    return fit


FITTERS = {"OLS": fit_ols, "2SLS": fit_2sls, "JIVE1": _fit_package(JIVE1), "JIVE2": _fit_package(JIVE2),
           "UJIVE1": _fit_package(UJIVE1), "UJIVE2": _fit_package(UJIVE2)}


def fit_all(y, x, Z, W):
    """Fit every estimator to one data set. An estimator that raises (for example a leverage of 1) counts as a failure."""
    out = {}
    for name, fitter in FITTERS.items():
        try:
            out[name] = fitter(y, x, Z, W)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            out[name] = FAILED
    return out


# --------------------------------------------------------------------------------------------------------------------
# One cell of the grid, and its summary
# --------------------------------------------------------------------------------------------------------------------
def summarize(estimates, ses, los, his, beta0=BETA0):
    """Summary statistics of one estimator over the replications of one cell (NaN marks a failed replication)."""
    est, se, lo, hi = (np.asarray(a, dtype=float) for a in (estimates, ses, los, his))
    ok = np.isfinite(est) & np.isfinite(se)
    n_ok = int(ok.sum())
    out = dict(n_reps=len(est), n_failed=int(len(est) - n_ok))
    if n_ok == 0:
        return {**out, **{k: np.nan for k in ("median_bias", "mean_bias", "mad", "rmse", "iqr_sd", "coverage", "median_ci_width", "se_ratio")}}
    err = est[ok] - beta0
    q25, q75 = np.percentile(est[ok], [25, 75])
    iqr_sd = (q75 - q25) / 1.349  # robust standard deviation: the estimators can have very heavy tails
    out.update(
        median_bias=float(np.median(err)),
        mean_bias=float(np.mean(err)),
        mad=float(np.median(np.abs(err))),
        rmse=float(np.sqrt(np.mean(err ** 2))),
        iqr_sd=float(iqr_sd),
        coverage=float(np.mean((lo[ok] <= beta0) & (beta0 <= hi[ok]))),
        median_ci_width=float(np.median(hi[ok] - lo[ok])),
        se_ratio=float(np.median(se[ok]) / iqr_sd) if iqr_sd > 0 else np.nan,  # ~1 if the reported SE is right
    )
    return out


def cell_key(index, cell, reps, seed):
    digest = hashlib.md5(json.dumps([dataclasses.astuple(cell), reps, seed]).encode()).hexdigest()[:8]
    return f"cell_{index:03d}_{digest}"


def run_cell(index, cell, reps, seed, save_draws=False):
    """Run all replications of one cell. Replication r uses its own generator, seeded by (seed, cell index, r)."""
    est = {name: {f: np.full(reps, np.nan) for f in ("beta", "se", "lo", "hi")} for name in ESTIMATORS}
    fs = np.full(reps, np.nan)
    for r in range(reps):
        rng = np.random.default_rng([seed, index, r])
        fits = fit_all(*simulate(cell, rng))
        for name, fit in fits.items():
            for f in ("beta", "se", "lo", "hi"):
                est[name][f][r] = getattr(fit, f)
        fs[r] = next((fit.first_stage_f for fit in fits.values() if np.isfinite(fit.first_stage_f)), np.nan)
    rows = []
    for name in ESTIMATORS:
        row = dict(cell_index=index, **dataclasses.asdict(cell), expected_first_stage_f=1 + cell.mu2 / cell.k, estimator=name,
                   **summarize(est[name]["beta"], est[name]["se"], est[name]["lo"], est[name]["hi"]),
                   median_first_stage_f=float(np.nanmedian(fs)) if np.isfinite(fs).any() else np.nan)
        rows.append(row)
    draws = None
    if save_draws:
        draws = pd.DataFrame([dict(cell_index=index, rep=r, estimator=name, beta=est[name]["beta"][r], se=est[name]["se"][r],
                                   lo=est[name]["lo"][r], hi=est[name]["hi"][r]) for name in ESTIMATORS for r in range(reps)])
    return pd.DataFrame(rows), draws


def _worker(args):
    index, cell, reps, seed, save_draws, outdir = args
    t0 = time.time()
    summary, draws = run_cell(index, cell, reps, seed, save_draws)
    key = cell_key(index, cell, reps, seed)
    if draws is not None:
        (Path(outdir) / "draws").mkdir(exist_ok=True)
        draws.to_csv(Path(outdir) / "draws" / f"{key}.csv", index=False)
    summary.to_csv(Path(outdir) / "cells" / f"{key}.csv", index=False)  # written last: its presence marks the cell as finished
    return index, time.time() - t0


# --------------------------------------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", choices=sorted(GRIDS), default="full", help="preset design grid (default: full)")
    p.add_argument("--n", type=int, nargs="+", help="override: sample sizes")
    p.add_argument("--k", type=int, nargs="+", help="override: numbers of instruments")
    p.add_argument("--mu2", type=float, nargs="+", help="override: concentration parameters (instrument strength; first-stage F is about 1 + mu2 / k)")
    p.add_argument("--rho", type=float, nargs="+", help="override: error correlations (endogeneity)")
    p.add_argument("--controls", type=int, nargs="+", help="override: numbers of controls")
    p.add_argument("--hetero", choices=["no", "yes"], nargs="+", help="override: homoskedastic and/or heteroskedastic errors")
    p.add_argument("--reps", type=int, default=2000, help="replications per cell (default 2000)")
    p.add_argument("--seed", type=int, default=20260924, help="master seed (default 20260924)")
    p.add_argument("--workers", type=int, default=1, help="processes to run cells in parallel (default 1)")
    p.add_argument("--out", type=Path, default=HERE / "results", help="output directory (default simulation/results)")
    p.add_argument("--save-draws", action="store_true", help="also save every replication's estimate, SE and interval (large)")
    p.add_argument("--dry-run", action="store_true", help="print the design and the amount of work, then stop without simulating")
    return p.parse_args(argv)


def grid_from_args(args):
    grid = {k: list(v) for k, v in GRIDS[args.grid].items()}
    for axis in ("n", "k", "mu2", "rho", "controls"):
        if getattr(args, axis) is not None:
            grid[axis] = getattr(args, axis)
    if args.hetero is not None:
        grid["hetero"] = [h == "yes" for h in args.hetero]
    return grid


def main(argv=None):
    args = parse_args(argv)
    cells = build_cells(grid_from_args(args))
    table = pd.DataFrame([dataclasses.asdict(c) for c in cells])
    n_fits = len(cells) * args.reps * len(ESTIMATORS)
    if args.dry_run:
        print(table.to_string())
        print(f"\n{len(cells)} cells x {args.reps} replications x {len(ESTIMATORS)} estimators = {n_fits:,} model fits")
        print(f"estimators: {', '.join(ESTIMATORS)}   true coefficient: {BETA0}   master seed: {args.seed}   workers: {args.workers}")
        print(f"output: {args.out}   (nothing simulated: --dry-run)")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cells").mkdir(exist_ok=True)
    pending = [(i, c, args.reps, args.seed, args.save_draws, str(args.out)) for i, c in enumerate(cells)
               if not (args.out / "cells" / f"{cell_key(i, c, args.reps, args.seed)}.csv").exists()]
    print(f"{len(cells)} cells, {len(cells) - len(pending)} already finished, {len(pending)} to run ({args.reps} replications each)")
    (args.out / "config.json").write_text(json.dumps(dict(args={k: str(v) for k, v in vars(args).items()}, cells=[dataclasses.asdict(c) for c in cells]), indent=2))

    t0 = time.time()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_worker, job) for job in pending]
            for done, fut in enumerate(as_completed(futures), 1):
                index, seconds = fut.result()
                print(f"[{done}/{len(pending)}] cell {index} {cells[index]} finished in {seconds:.0f}s (elapsed {time.time() - t0:.0f}s)", flush=True)
    else:
        for done, job in enumerate(pending, 1):
            index, seconds = _worker(job)
            print(f"[{done}/{len(pending)}] cell {index} {cells[index]} finished in {seconds:.0f}s (elapsed {time.time() - t0:.0f}s)", flush=True)

    files = [args.out / "cells" / f"{cell_key(i, c, args.reps, args.seed)}.csv" for i, c in enumerate(cells)]
    summary = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    summary.to_csv(args.out / "summary.csv", index=False)
    print(f"wrote {args.out / 'summary.csv'} ({len(summary)} rows). Next: python simulation/report.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
