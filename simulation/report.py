"""Tables and figures from simulation/results/summary.csv (written by simulation/monte_carlo.py).

    python simulation/report.py [--results simulation/results]

Prints one table per metric (rows: controls / instruments K / concentration mu2; columns: estimators, plus the median observed first-stage F) and writes them to
report.md. If matplotlib is installed it also saves figures (median bias and coverage against K).
"""
import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ORDER = ["OLS", "2SLS", "JIVE1", "JIVE2", "UJIVE1", "UJIVE2"]
METRICS = [
    ("median_bias", "Median bias of the coefficient (truth = 1). Closer to 0 is better; OLS shows the size of the endogeneity problem.", "{:+.3f}"),
    ("mad", "Median absolute error. Robust to heavy tails, unlike RMSE.", "{:.3f}"),
    ("coverage", "Coverage of the nominal 95% confidence interval. Closer to 0.95 is better.", "{:.3f}"),
    ("se_ratio", "Median reported standard error / robust (IQR-based) standard deviation of the estimates. Near 1 means the standard error is right.", "{:.2f}"),
    ("median_ci_width", "Median width of the 95% confidence interval.", "{:.2f}"),
    ("n_failed", "Replications in which the estimator raised an error (out of --reps).", "{:.0f}"),
]


def load(results):
    summary = pd.read_csv(Path(results) / "summary.csv")
    for extra in ("hetero",):
        summary[extra] = summary[extra].astype(bool)
    return summary


def table(summary, metric, fmt):
    parts = []
    for (n, rho, hetero), grp in summary.groupby(["n", "rho", "hetero"]):
        wide = grp.pivot_table(index=["controls", "k", "mu2"], columns="estimator", values=metric)
        wide = wide[[c for c in ORDER if c in wide.columns]]
        first_f = grp.groupby(["controls", "k", "mu2"])["median_first_stage_f"].first().rename("F_obs")
        wide = wide.join(first_f)
        text = wide.to_string(float_format=lambda v: fmt.format(v))
        parts.append(f"N = {n}, error correlation = {rho}, heteroskedastic = {hetero}\n\n```\n{text}\n```\n")
    return "\n".join(parts)


def figures(summary, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping figures")
        return
    for metric, ylabel, ref in [("median_bias", "median bias", 0.0), ("coverage", "95% CI coverage", 0.95)]:
        for (n, rho, hetero, controls), grp in summary.groupby(["n", "rho", "hetero", "controls"]):
            fs = sorted(grp["mu2"].unique())
            fig, axes = plt.subplots(1, len(fs), figsize=(4 * len(fs), 3.4), sharey=True, squeeze=False)
            for ax, f in zip(axes[0], fs):
                sub = grp[grp["mu2"] == f]
                for est in ORDER:
                    line = sub[sub["estimator"] == est].sort_values("k")
                    if len(line):
                        ax.plot(line["k"], line[metric], marker="o", label=est)
                ax.axhline(ref, color="grey", lw=0.8, ls="--")
                ax.set_title(f"concentration mu2 = {f:g}")
                ax.set_xlabel("number of instruments K")
            axes[0][0].set_ylabel(ylabel)
            axes[0][-1].legend(fontsize=7)
            fig.suptitle(f"N = {n}, rho = {rho}, controls = {controls}, heteroskedastic = {hetero}", fontsize=9)
            fig.tight_layout()
            name = f"{metric}_n{n}_rho{rho}_controls{controls}_het{int(hetero)}.png"
            fig.savefig(out / name, dpi=130)
            plt.close(fig)
    print(f"figures saved in {out}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=HERE / "results")
    args = p.parse_args(argv)
    summary = load(args.results)
    reps = int(summary["n_reps"].iloc[0])
    md = [f"# Monte Carlo results ({reps} replications per cell)\n"]
    for metric, description, fmt in METRICS:
        md.append(f"## {metric}\n\n{description}\n")
        md.append(table(summary, metric, fmt))
    text = "\n".join(md)
    (args.results / "report.md").write_text(text)
    print(text)
    figures(summary, args.results)


if __name__ == "__main__":
    main()
