"""
pop_growth_modelling.py

Common functions used for population growth modelling.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

USE_EXPERIMENTAL_PLUS_ONE = True
AVERAGE_EXPERIMENTAL_GR = 0.271717555
AVERAGE_EXPERIMENTAL_GR_STD = 0.335737853

if USE_EXPERIMENTAL_PLUS_ONE:
    SETTINGS = {
        'replicates': 1000,
        'initial_n': 100000,
        'generations': 20,
        'growth_rate': AVERAGE_EXPERIMENTAL_GR + 1,
        'growth_rate_std': AVERAGE_EXPERIMENTAL_GR_STD + 1
    }
else:
    SETTINGS = {
        'replicates': 1000,
        'initial_n': 100000,
        'generations': 20,
        # 320520 cells after 20 generations
        'growth_rate': 320520**(1/20),
        # normalized based on experimental data: GR=0.3357, GR_SD = 0.2717
        'growth_rate_std': (320520**(1/20) / 0.3357) * 0.2717
    }


def main(replicates: int, initial_n: int, generations: int, growth_rate: float, growth_rate_std: float) -> None:
    colnames = ['fixed', 'gaussian', 'uniform']
    gaussian_params = {
        'mi': growth_rate,
        'sigma': growth_rate_std,
    }
    uniform_params = {
        'lower': growth_rate - 3 * growth_rate_std,
        'upper': growth_rate + 3 * growth_rate_std,
    }
    df = pd.DataFrame(columns=colnames)
    gauss_means = []
    gauss_sd = []
    for _ in range(replicates):
        fixed_final_n = calculate_fixed_final_n(initial_n=initial_n,
                                                generations=generations,
                                                growth_rate=growth_rate)
        gaussian_gr_sample = get_gaussian_growth_rate_sample(**gaussian_params, generations=generations)
        gauss_means.append(gaussian_gr_sample.mean())
        gauss_sd.append(gaussian_gr_sample.std())
        gaussian_final_n = calculate_random_final_n(initial_n=initial_n, gr_sample=gaussian_gr_sample)
        uniform_gr_sample = get_uniform_growth_rate_sample(**uniform_params, generations=generations)
        uniform_final_n = calculate_random_final_n(initial_n=initial_n, gr_sample=uniform_gr_sample)
        df.loc[len(df), :] = pd.Series([fixed_final_n, gaussian_final_n, uniform_final_n], index=colnames)
    plot_results(df=df, initial_n=initial_n, replicates=replicates, generations=generations)
    fig, ax = plt.subplots()
    ax.hist(gauss_means, label='Mean', color='blue', alpha=0.5)
    ax.axvline(np.mean(gauss_means), linestyle='--', alpha=0.9)
    ax.hist(gauss_sd, label='SD', color='orange', alpha=0.5)
    ax.axvline(np.mean(gauss_sd), linestyle='--', alpha=0.9)
    ax.legend()
    fig.suptitle('Histogram of gaussian means and SD')
    print("Student's t-test - Fixed vs Gaussian ->", stats.ttest_rel(df.fixed,df.gaussian))
    print("Student's t-test - Fixed vs Uniform ->", stats.ttest_rel(df.fixed,df.gaussian))
    plt.show()


def plot_results(df: pd.DataFrame, initial_n: int, replicates: int, generations: float) -> None:
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    pal = sns.color_palette('rainbow')
    gauss_color = pal[0]
    unif_color = pal[-1]
    ax.scatter(df.fixed, df.gaussian, color=gauss_color, label='gaussian', alpha=0.7)
    ax.scatter(df.fixed, df.uniform, color=unif_color, label='uniform', alpha=0.7)
    ax.scatter(initial_n, initial_n, color='white', edgecolor='black', marker='*', s=200, label='start', zorder=1)
    plt.plot(plt.xlim(), plt.xlim(), linestyle='--', color='k', lw=3, scalex=False, scaley=False, zorder=0)
    ax.set_xlabel('Final N - Fixed GR (mean of dynamic GR for the same replicate)')
    ax.set_ylabel('Final N - Dynamic GR (fluctuates on each generation)')
    ax.legend()
    fig.suptitle(f'Comparison of fixed and fluctuating GR\n{initial_n=} {replicates=}, {generations=}')


def get_gaussian_growth_rate_sample(mi: float, sigma: float, generations: int) -> np.array:
    return np.random.normal(loc=mi, scale=sigma, size=generations)


def get_uniform_growth_rate_sample(lower: float, upper: float, generations: int) -> np.array:
    return np.random.uniform(low=lower, high=upper, size=generations)


def calculate_random_final_n(initial_n: int, gr_sample) -> int:
    return initial_n * gr_sample.cumprod()[-1]


def calculate_fixed_final_n(initial_n: int, growth_rate: float, generations: int) -> float:
    return initial_n * (growth_rate ** generations)


if __name__ == '__main__':
    main(**SETTINGS)
