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
import warnings
warnings.filterwarnings("ignore")

def create_data(n=1000, num_instruments=None, strength=0.1):
    """
    Create a dataset with endogenous variables and instruments.

    Parameters:
    n (int): Number of observations.
    num_instruments (int or None): Number of instruments (excluding constant). If None, defaults to n // 2.
    strength (float or array): Strength of instruments (alpha coefficient). Can be a scalar or array of length num_instruments+1.

    Returns:
    tuple: (Y, X, Z)
    """
    if num_instruments is None:
        num_instruments = n // 2

    Z = np.random.randn(n, num_instruments)
    column_of_ones = np.ones((Z.shape[0], 1))
    Z = np.hstack((column_of_ones, Z))

    # Parameter vectors:
    if np.isscalar(strength):
        α = np.full(num_instruments + 1, strength)
    else:
        α = np.asarray(strength)
        assert α.shape[0] == num_instruments + 1, "strength array must match number of instruments + 1"

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







# Set up the parameters
n_runs = 10
n_sim = 10
n = 1000
true_beta = 2

# Define the instrument numbers and strengths to loop over
instrument_numbers = [n // d for d in [2, 3, 4, 5, 6, 7, 8, 9, 10]]
strength_values = np.linspace(0.1, 0.9, 9)

results = []

for num_instruments in instrument_numbers:
    print(f"Starting simulations for num_instruments = {num_instruments}")
    for strength in strength_values:
        print(f"  Strength = {strength:.2f}")
        for run in range(n_runs):
            print(f"    Run {run + 1} of {n_runs}...", end="", flush=True)
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

            # New lists for F-statistics
            fstat_2sls_list = []
            fstat_ujive1_list = []
            fstat_ujive2_list = []
            fstat_jive1_list = []
            fstat_jive2_list = []
            fstat_hful_list = []
            fstat_sjive_list = []

            for sim in range(n_sim):
                if sim == 0 or (sim + 1) % 5 == 0 or sim == n_sim - 1:
                    print(f".{sim + 1}", end="", flush=True)
                Y, X, Z = create_data(n=n, num_instruments=num_instruments, strength=strength)

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

                # Calculate basic F-stat for 2SLS manually
                X_endog = X[:, 1:]  # Exclude constant
                X_mean = np.mean(X_endog)
                fitted_vals = Z @ np.linalg.inv(Z.T @ Z) @ Z.T @ X_endog
                rss_restricted = np.sum((X_endog - X_mean)**2)
                rss_unrestricted = np.sum((X_endog - fitted_vals)**2)
                df1 = Z.shape[1] - 1  # Excluding constant
                df2 = n - Z.shape[1]
                fstat_2sls = ((rss_restricted - rss_unrestricted) / df1) / (rss_unrestricted / df2)
                fstat_2sls_list.append(fstat_2sls)

                # UJIVE1
                ujive1 = UJIVE1(Y, X, Z, talk=False)
                beta_ujive1_list.append(ujive1.beta[1])
                fstat_ujive1_list.append(ujive1.f_stat)

                # UJIVE2
                ujive2 = UJIVE2(Y, X, Z, talk=False)
                beta_ujive2_list.append(ujive2.beta[1])
                fstat_ujive2_list.append(ujive2.f_stat)

                # JIVE1
                jive1 = JIVE1(Y, X, Z, talk=False)
                beta_jive1_list.append(jive1.beta[1])
                fstat_jive1_list.append(jive1.f_stat)

                # JIVE2
                jive2 = JIVE2(Y, X, Z, talk=False)
                beta_jive2_list.append(jive2.beta[1])
                fstat_jive2_list.append(jive2.f_stat)

                # HFUL
                hful = HFUL(Y, X, Z, talk=False)
                beta_hful_list.append(hful.betas[1])

                # SJIVE
                sjive = SJIVE(Y, X, Z, talk=False)
                beta_sjive_list.append(sjive.beta)

            print(" done.", flush=True)  # End of run

            # Compute statistics for this run
            for name, estimates, f_stats in zip(
                ['OLS', '2SLS', 'UJIVE1', 'UJIVE2', 'JIVE1', 'JIVE2', 'HFUL', 'SJIVE'],
                [beta_ols_list, beta_2sls_list, beta_ujive1_list, beta_ujive2_list, beta_jive1_list, beta_jive2_list, beta_hful_list, beta_sjive_list],
                [None, fstat_2sls_list, fstat_ujive1_list, fstat_ujive2_list, fstat_jive1_list, fstat_jive2_list, fstat_hful_list, fstat_sjive_list]
            ):
                estimates = np.array(estimates)
                bias = np.mean(estimates - true_beta)
                variance = np.var(estimates)
                mse = np.mean((estimates - true_beta) ** 2)

                # Calculate mean F-statistic (if available)
                if f_stats is not None:
                    f_stats = np.array([f for f in f_stats if f is not None])
                    mean_fstat = np.mean(f_stats) if len(f_stats) > 0 else None
                else:
                    mean_fstat = None

                results.append({
                    'run': run + 1,
                    'estimator': name,
                    'bias': bias,
                    'variance': variance,
                    'mse': mse,
                    'fstat': mean_fstat,
                    'num_instruments': num_instruments,
                    'strength': strength
                })

    import pandas as pd
    # Convert results to DataFrame and save as CSV
    results_df = pd.DataFrame(results)
    # Make the fstat if inf to be 10,000 in the results_df
    results_df.loc[results_df['fstat'] == np.inf, 'fstat'] = 10000
    # same for NaN
    results_df.loc[results_df['fstat'].isna(), 'fstat'] = 10000

    results_df.to_csv(f"simulation_results_numinst{num_instruments}.csv", index=False)
    print(f"Results saved to simulation_results_numinst{num_instruments}.csv")
    results.clear()  # Free up memory for next iteration

print("All simulations complete.")

