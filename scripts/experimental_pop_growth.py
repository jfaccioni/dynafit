"""
pop_growth_modelling.py

Common functions used for population growth modelling.
"""
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

USE_EXPERIMENTAL_PLUS_ONE = True
AVERAGE_EXPERIMENTAL_GR = 0.003  # old value = 0.271717555
AVERAGE_EXPERIMENTAL_GR_STD = 0.004 # old value = 0.335737853


if USE_EXPERIMENTAL_PLUS_ONE:
    SETTINGS = {
        'replicates': 1000,
        'initial_n': 1000,
        'generations': 365,
        'growth_rate': AVERAGE_EXPERIMENTAL_GR,
        'growth_rate_std': AVERAGE_EXPERIMENTAL_GR_STD
    }
else:
    SETTINGS = {
        'replicates': 1000,
        'initial_n': 1000,
        'generations': 20,
        # 320520 cells after 20 generations
        'growth_rate': 320520**(1/20),
        # normalized based on experimental data: GR=0.3357, GR_SD = 0.2717
        'growth_rate_std': (320520**(1/20) / 0.3357) * 0.2717
    }


def main(replicates: int, initial_n: int, generations: int, growth_rate: float, growth_rate_std: float) -> None:
    # extract gaussian/uniform parameters from experimental input distribution
    gaussian_params = {
        'mi': growth_rate,
        'sigma': growth_rate_std,
    }
    uniform_params = {
        'lower': growth_rate - 3 * growth_rate_std,
        'upper': growth_rate + 3 * growth_rate_std,
    }
    # initialize containers for storing data during main loop
    colnames = ['fixed', 'gaussian_1', 'gaussian_2', 'gaussian_3']
    df = pd.DataFrame(columns=colnames)
    gauss_means = []
    gauss_sd = []
    unif_means = []
    unif_sd = []
    # main loop
    for _ in range(replicates):
        # fixed N (should be always the same)
        fixed_final_n = calculate_fixed_final_n(initial_n=initial_n, generations=generations, growth_rate=growth_rate)
        # Gaussian GR distribution
        gaussian_gr_sample_1 = get_gaussian_growth_rate_sample(**gaussian_params, generations=generations)
        gaussian_final_n_1 = calculate_random_final_n(initial_n=initial_n, gr_sample=gaussian_gr_sample_1)
        gaussian_params['sigma'] /= 10
        gaussian_gr_sample_2 = get_gaussian_growth_rate_sample(**gaussian_params, generations=generations)
        gaussian_final_n_2 = calculate_random_final_n(initial_n=initial_n, gr_sample=gaussian_gr_sample_2)
        gaussian_params['sigma'] /= 10
        gaussian_gr_sample_3 = get_gaussian_growth_rate_sample(**gaussian_params, generations=generations)
        gaussian_final_n_3 = calculate_random_final_n(initial_n=initial_n, gr_sample=gaussian_gr_sample_3)
        # Uniform GR distribution
        # uniform_gr_sample = get_uniform_growth_rate_sample(**uniform_params, generations=generations)
        # uniform_final_n = calculate_random_final_n(initial_n=initial_n, gr_sample=uniform_gr_sample)
        # Add values to containers
        s = pd.Series([fixed_final_n, gaussian_final_n_1, gaussian_final_n_2, gaussian_final_n_3], index=colnames)
        df.loc[len(df), :] = s
        # gauss_means.append(gaussian_gr_sample.mean())
        # gauss_sd.append(gaussian_gr_sample.std())
        # unif_means.append(uniform_gr_sample.mean())
        # unif_sd.append(uniform_gr_sample.std())
        gaussian_params['sigma'] *= 10 * 10
    # plot data
    plot_results(df=df, initial_n=initial_n, replicates=replicates, generations=generations,
                 var=str(gaussian_params['sigma']))
    # print(len(df.loc[df.gaussian > fixed_final_n]))
    # print(len(df.loc[df.gaussian <= fixed_final_n]))
    # plot_histograms(gauss_means=gauss_means, gauss_sd=gauss_sd, name='gaussian')
    # plot_histograms(gauss_means=unif_means, gauss_sd=unif_sd, name='uniform')
    # t-tests
    # print("Student's t-test - Fixed vs Gaussian ->", stats.ttest_rel(df.fixed, df.gaussian))
    # print("Student's t-test - Fixed vs Uniform ->", stats.ttest_rel(df.fixed, df.uniform))
    # show plots
    plt.show()


def plot_results(df: pd.DataFrame, initial_n: int, replicates: int, generations: float, var: str) -> None:
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    pal = sns.color_palette('rainbow')
    gauss_color = pal[0]
    # unif_color = pal[-1]
    ax.scatter([3 for i in range(len(df))] + np.random.random(df.gaussian_1.shape) / 2, df.gaussian_1, color=gauss_color, alpha=0.7, s=0.5)
    ax.scatter([2 for i in range(len(df))] + np.random.random(df.gaussian_1.shape) / 2, df.gaussian_2, color=gauss_color, alpha=0.7, s=0.5)
    ax.scatter([1 for i in range(len(df))] + np.random.random(df.gaussian_1.shape) / 2, df.gaussian_3, color=gauss_color, alpha=0.7, s=0.5)
    # ax.scatter(df.fixed, df.uniform, color=unif_color, label='uniform', alpha=0.7)
    ax.axhline(initial_n, label='initial', linestyle='--', color='k', lw=3)
    ax.axhline(df.fixed[0], label='final (fixed)', linestyle='--', color='r', lw=3)
    # ax.scatter(initial_n, initial_n, color='white', edgecolor='black', marker='*', s=200, label='start', zorder=1)
    # plt.plot(plt.xlim(), plt.xlim(), linestyle='--', color='k', lw=3, scalex=False, scaley=False, zorder=0)
    ax.set_xlabel('Final N - Fixed GR (mean of dynamic GR for the same replicate)')
    ax.set_ylabel('Final N - Dynamic GR (fluctuates on each generation)')
    ax.legend()
    fig.suptitle(f'Comparison of fixed and fluctuating GR\n{initial_n=} {replicates=}, {generations=}')


def plot_histograms(gauss_means: List[float], gauss_sd: List[float], name: str) -> None:
    fig, ax = plt.subplots()
    ax.hist(gauss_means, label='Mean', color='blue', alpha=0.5)
    ax.axvline(np.mean(gauss_means), linestyle='--', color='blue', alpha=0.9)
    ax.hist(gauss_sd, label='SD', color='orange', alpha=0.5)
    ax.axvline(np.mean(gauss_sd), linestyle='--', color='orange', alpha=0.9)
    ax.legend()
    fig.suptitle(f'Histogram of {name} means and SD')


def get_gaussian_growth_rate_sample(mi: float, sigma: float, generations: int) -> np.array:
    return np.random.normal(loc=mi, scale=sigma, size=generations)


def get_uniform_growth_rate_sample(lower: float, upper: float, generations: int) -> np.array:
    return np.random.uniform(low=lower, high=upper, size=generations)


def calculate_random_final_n(initial_n: int, gr_sample) -> int:
    n = initial_n
    for gr in gr_sample:
        n = n + n * gr
    return n
    # return initial_n * gr_sample.cumprod()[-1]


def calculate_fixed_final_n(initial_n: int, growth_rate: float, generations: int) -> float:
    n = initial_n
    for i in range(generations):
        n = n + n * growth_rate
    return n


if __name__ == '__main__':
    main(**SETTINGS)
