"""
experimental_pop_growth.py

Analytical module comparing cell population with gaussian-distributed growth rates to static (mean of gaussians)
and fixed (literature) growth rates.
"""

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_1samp

SETTINGS = {
    'replicates': 100,
    'initial_n': 1000,
    'generations': 20,
    'fixed_growth_rate': 0.351920148690623,
    'growth_rate_std': 0.117034030929312,
}


def main(replicates: int, initial_n: int, generations: int, fixed_growth_rate: float,
         growth_rate_std: float) -> None:
    """Main function of this script."""
    replicate_list = []
    fixed_pop_sizes = calculate_pop_size_from_growth_rate(initial_n=initial_n, generations=generations,
                                                          growth_rate=fixed_growth_rate)
    for i in range(replicates):
        gaussian_growth_rate_sample = sample_from_gaussian(mi=fixed_growth_rate, sigma=growth_rate_std,
                                                           n=generations)
        dynamic_pop_size = calculate_pop_size_from_gr_vector(initial_n=initial_n, gr_vector=gaussian_growth_rate_sample)
        static_growth_rate = calculate_static_gr_from_gaussian_gr(gaussian_gr=gaussian_growth_rate_sample)
        static_pop_size = calculate_pop_size_from_growth_rate(initial_n=initial_n, generations=generations,
                                                              growth_rate=static_growth_rate)
        df = pd.DataFrame(
            {
                'Replicate': i + 1,
                'Generation': range(generations),
                'Fixed_GR': fixed_growth_rate,
                'Fixed_GR_SD': growth_rate_std,
                'Fixed_N': fixed_pop_sizes,
                'Dynamic_GR': gaussian_growth_rate_sample,
                'Dynamic_N': dynamic_pop_size,
                'Static_GR': static_growth_rate,
                'Static_N': static_pop_size,
            }
        )
        replicate_list.append(df)
    data = pd.concat(replicate_list)
    data.to_excel(f'experimental_pop_growth_LS12gen20_MIN.xlsx', index_label=f'{initial_n} initial cells')
    summary_stats = measure_replicate_stats(replicate_list=replicate_list)
    summary_stats.to_excel(f'experimental_pop_growth_stats_LS12gen20_MIN.xlsx', index_label=f'Replicate')


def calculate_pop_size_from_growth_rate(initial_n: int, generations: int, growth_rate: float) -> np.array:
    """Given an initial cell number, a number of generations and a fixed growth rate, returns a vector of
    the number of cells after each generation."""
    n = initial_n
    ns = []
    for i in range(generations):
        n = n + n * growth_rate
        ns.append(n)
    return np.array(ns)


def sample_from_gaussian(mi: float, sigma: float, n: int) -> np.array:
    """Returns a vector with n samples from a gaussian distribution
    of fixed mean(mi) and standard deviation (sigma)."""
    return np.random.normal(loc=mi, scale=sigma, size=n)


def calculate_pop_size_from_gr_vector(initial_n: int, gr_vector: np.array) -> np.array:
    """Given an initial cell number, a number of generations and a vector of growth rates, returns a vector of
    the number of cells after each generation (applying each growth rate sequentially to the sample)."""
    n = initial_n
    ns = []
    for gr in gr_vector:
        n = n + n * gr
        ns.append(n)
    return np.array(ns)


def calculate_static_gr_from_gaussian_gr(gaussian_gr: np.array) -> float:
    return gaussian_gr.mean()


def measure_replicate_stats(replicate_list: List[pd.DataFrame]) -> pd.DataFrame:
    summary_stats_list = []
    for i, replicate in enumerate(replicate_list, 1):
        _, dynamic_pval = ttest_1samp(a=replicate.Dynamic_GR, popmean=replicate.Fixed_GR[0])
        df = pd.DataFrame(
            {
                'Fixed_GR': replicate.Fixed_GR.iloc[0],
                'Fixed_GR_SD': replicate.Fixed_GR_SD.iloc[0],
                'Static_GR': replicate.Static_GR.mean(),
                'Static_GR_Final_N': replicate.Static_N.iloc[-1],
                'Dynamic_GR_Mean': replicate.Dynamic_GR.mean(),
                'Dynamic_GR_SD': replicate.Dynamic_GR.std(),
                'Dynamic_GR_Final_N': replicate.Dynamic_N.iloc[-1],
                'Dynamic_ttest': abs(dynamic_pval) < 0.025,
                'Dynamic_pval': dynamic_pval,
                'Static_Dynamic_Ratio': replicate.Static_N.iloc[-1] / replicate.Dynamic_N.iloc[-1],
            }, index=[i]
        )
        summary_stats_list.append(df)
    return pd.concat(summary_stats_list)


def plot_results(df: pd.DataFrame, initial_n: int, replicates: int, generations: float, var: str) -> None:
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    pal = sns.color_palette('rainbow')
    gauss_color = pal[0]
    # unif_color = pal[-1]
    # ax.scatter([3 for i in range(len(df))] + np.random.random(df.gaussian_1.shape) / 2, df.gaussian_1, color=gauss_color, alpha=0.7, s=0.5)
    # ax.scatter([2 for i in range(len(df))] + np.random.random(df.gaussian_1.shape) / 2, df.gaussian_2, color=gauss_color, alpha=0.7, s=0.5)
    # ax.scatter([1 for i in range(len(df))] + np.random.random(df.gaussian_1.shape) / 2, df.gaussian_3, color=gauss_color, alpha=0.7, s=0.5)
    ax.scatter(df.fixed, df.gaussian_1, color=gauss_color, label='uniform', alpha=0.7)
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


if __name__ == '__main__':
    main(**SETTINGS)
