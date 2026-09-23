# Import necessary libraries
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
font_path = '/Library/Fonts/latinmodern-math.otf' 
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Latin Modern Math'

# Create data
def create_data(n=1000, perc=0.5, strength=0.01):
    """
    Create a dataset with endogenous variables and instruments.
    
    Parameters:
    n (int): Number of observations.
    
    Returns:
    DataFrame: A DataFrame containing the generated data.
    """
    # Create many instruments: half as many as n
    num_instruments = int(n * perc )
    Z = np.random.randn(n, num_instruments)
    column_of_ones = np.ones((Z.shape[0], 1))
    Z = np.hstack((column_of_ones, Z))

    # Parameter vectors:
    # Make the coefficients on the instruments (alpha) very small to make instruments weak
    α = np.full(num_instruments + 1, strength)  # All coefficients are 0.01 (weak instruments)
    β = np.array([1, 2])

    # Error terms:
    e1 = np.random.normal(0, 6, n)
    e2 = np.random.normal(0, 6, n)
    δ = np.random.normal(0, 1)
    ε = 2 * e1 - 2 * e2 + δ

    # Making our endogenous variable:
    x = np.dot(Z, α) + .2 * e1
    X = np.column_stack((column_of_ones, x))

    # Outcome vector:
    Y = np.dot(X, β) + ε

    return Y, X, Z



from ujive1 import *
from ujive2 import *
from tsls import *
from ijive import *
from hful import *
from jive1 import *
from jive2 import *
from lagrange_multiplier import *
from liml import *
from sjive import *
warnings.filterwarnings("ignore")

def run_monte_carlo(n_runs, n_sim, n, true_beta, perc, strength, figures_dir="../figures"):
    results = []

    for run in range(n_runs):
        beta_ols_list = []
        beta_2sls_list = []
        beta_ujive1_list = []
        beta_ujive2_list = []
        beta_jive1_list = []
        beta_jive2_list = []
        beta_ijive_list = []
        beta_hful_list = []
        beta_liml_list = []
        beta_sjive_list = []
        beta_lm_list = []

        for sim in range(n_sim):
            Y, X, Z = create_data(n=n, perc=perc, strength=strength)

            # OLS
            bhat_ols = np.linalg.inv(X.T @ X) @ X.T @ Y
            beta_ols_list.append(bhat_ols[1])

            # 2SLS
            Zt_Z = Z.T @ Z
            Zt_Z_inv = np.linalg.inv(Zt_Z)
            pz = Z @ Zt_Z_inv @ Z.T
            proj_x = pz @ X
            first = np.linalg.inv(proj_x.T @ X)
            second = proj_x.T @ Y
            bhat_2sls = first @ second
            beta_2sls_list.append(bhat_2sls[1])

            # UJIVE1
            ujive1 = UJIVE1(Y, X, Z, talk=False)
            beta_ujive1_list.append(ujive1.beta[1])

            # UJIVE2
            ujive2 = UJIVE2(Y, X, Z, talk=False)
            beta_ujive2_list.append(ujive2.beta[1])

            # JIVE1
            jive1 = JIVE1(Y, X, Z, talk=False)
            beta_jive1_list.append(jive1.beta[1])

            # JIVE2
            jive2 = JIVE2(Y, X, Z, talk=False)
            beta_jive2_list.append(jive2.beta[1])

            # HFUL
            hful = HFUL(Y, X, Z, talk=False)
            beta_hful_list.append(hful.betas[1])

            # SJIVE
            #sjive = SJIVE(Y, X, Z, talk=False)
            #beta_sjive_list.append(sjive.beta[1])

        # Compute statistics for this run
        for name, estimates in zip(
            ['OLS', '2SLS', 'UJIVE1', 'UJIVE2', 'JIVE1', 'JIVE2', 'HFUL'], # missing SJIVE
            [beta_ols_list, beta_2sls_list, beta_ujive1_list, beta_ujive2_list, beta_jive1_list, beta_jive2_list, beta_hful_list]
        ):
            estimates = np.array(estimates)
            bias = np.mean(estimates - true_beta)
            variance = np.var(estimates)
            mse = np.mean((estimates - true_beta) ** 2)
            results.append({
                'run': run + 1,
                'estimator': name,
                'bias': bias,
                'variance': variance,
                'mse': mse
            })

    results_df = pd.DataFrame(results)

    import os
    # Save LaTeX tables
    os.makedirs(figures_dir, exist_ok=True)
    for metric in ['bias', 'variance', 'mse']:
        summary = results_df.groupby('estimator')[metric].describe().drop(columns=['count'])
        summary = summary.applymap(lambda x: f"{x:.4f}")
        summary.columns = [col.upper() for col in summary.columns]
        summary.index.name = summary.index.name.capitalize() if summary.index.name else None
        n_cols = summary.shape[1] + 1
        col_format = 'l' + 'r' * (n_cols - 1)
        table_title = f"{metric} summary by estimator".title()
        latex_table = f"\\begin{{table}}[ht]\n\\centering\n\\caption{{{table_title}}}\n"
        latex_str = summary.to_latex(escape=False, column_format=col_format)
        latex_str = latex_str.replace('%', '\\%')
        latex_table += latex_str
        latex_table += "\\end{table}\n"
        fname = f"{figures_dir}/{metric}_table_perc{perc}_strength{strength}.tex"
        with open(fname, "w") as f:
            f.write(latex_table)

    # Plotting
    import seaborn as sns

    plt.figure(figsize=(12, 6))
    estimators = results_df['estimator'].unique()
    palette = sns.color_palette("tab10", n_colors=len(estimators))
    color_map = {est: palette[i] for i, est in enumerate(estimators)}
    all_biases = results_df['bias']
    bin_count = 200
    bin_edges = np.histogram_bin_edges(all_biases, bins=bin_count)

    for estimator in estimators:
        subset = results_df[results_df['estimator'] == estimator]
        color = color_map[estimator]
        plt.hist(
            subset['bias'],
            bins=bin_edges,
            alpha=0.6,
            label=estimator,
            edgecolor='black',
            linewidth=1,
            density=True,
            color=color
        )
        sns.kdeplot(
            subset['bias'],
            label=f"{estimator} KDE",
            linewidth=2,
            color=color,
            fill=True,
        )

    plt.axvline(0, color='k', linestyle='dashed', linewidth=2, label='True Bias (0)')
    plt.xlabel('Bias')
    plt.ylabel('Density')
    plt.title(f'Histogram and KDE of Bias (perc={perc}, strength={strength})')
    plt.xlim(all_biases.min()-0.75, all_biases.max()+0.5)
    plt.legend(frameon=True, loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/bias_plot_perc{perc}_strength{strength}.png", dpi=900, bbox_inches='tight')
    plt.close()

    # Black and white histogram
    plt.figure(figsize=(12, 6))
    hatch_patterns = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
    for i, estimator in enumerate(results_df['estimator'].unique()):
        subset = results_df[results_df['estimator'] == estimator]
        plt.hist(
            subset['bias'],
            bins=100,
            range=(all_biases.min(), all_biases.max()),
            alpha=1.0,
            label=estimator,
            color='white',
            edgecolor='black',
            linewidth=1.5,
            hatch=hatch_patterns[i % len(hatch_patterns)]
        )

    plt.axvline(0, color='black', linestyle='dashed', linewidth=2, label='True Bias (0)')
    plt.xlabel('Bias')
    plt.xlim(all_biases.min() - 1, all_biases.max()+0.5)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Bias (perc={perc}, strength={strength})')
    plt.legend(frameon=True, loc='upper left', fancybox=True, shadow=True, edgecolor='black')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/bias_plot_bw_perc{perc}_strength{strength}.png", dpi=300, bbox_inches='tight')
    plt.close()

# Now, loop over values of perc and strength
perc_values = [0.2, 0.3, 0.5]
strength_values = [0.3, 0.5, 0.7, 0.9]

for perc in perc_values:
    for strength in strength_values:
        print(f"Running simulation for perc={perc}, strength={strength}")
        run_monte_carlo(
            n_runs=10,
            n_sim=10,
            n=1000,
            true_beta=2,
            perc=perc,
            strength=strength,
            figures_dir="../figures"
        )