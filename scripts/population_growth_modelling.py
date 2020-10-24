"""
pop_growth_modelling.py

Common functions used for population growth modelling.
"""

from typing import Any, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


SETTINGS = {
    'save': False,
    'plot': True,
    'model_params': {
        'replicates': 1000,
        'initial_n': 1000,
        'generations': 1000,
        'gr_method': 'gaussian',
    },
    'gaussian_params': {
        'mi': 1.0,
        'sigma': 0.08/3,
    },
    'uniform_params': {
        'lower': 0.92,
        'upper': 1.08,
    },
}


def save_to_excel(gaussian_df: pd.DataFrame, uniform_df: pd.DataFrame, filename: str = 'results') -> None:
    xl = pd.ExcelWriter(f'{filename}.xlsx')
    gaussian_df.to_excel(excel_writer=xl, sheet_name='gaussian')
    uniform_df.to_excel(excel_writer=xl, sheet_name='uniform')
    xl.save()


def plot_results(gaussian_df, uniform_df, model_params: Dict[str, Any]) -> None:
    pal = sns.color_palette('rainbow')
    gauss_color = pal[0]
    unif_color = pal[-1]
    plt.scatter(gaussian_df.fixed_final_n, gaussian_df.random_final_n, color=gauss_color, label='gaussian', alpha=0.7)
    plt.scatter(uniform_df.fixed_final_n, uniform_df.random_final_n, color=unif_color, label='uniform', alpha=0.7)
    plt.plot(plt.xlim(), plt.xlim(), linestyle='--', color='k', lw=3, scalex=False, scaley=False, zorder=0)
    plt.scatter([model_params['initial_n']], [model_params['initial_n']], color='white', edgecolor='black',
                marker='*', s=200, label='start', zorder=1)
    plt.xlabel('Final N - Fixed GR (mean of dynamic GR for the same replicate)')
    plt.ylabel('Final N - Dynamic GR (fluctuates on each generation)')
    plt.title(f'Comparison of fixed and fluctuating GR\nreplicates={model_params["replicates"]} '
              f'initial_n={model_params["initial_n"]} generations={model_params["generations"]}')
    plt.legend()
    plt.ticklabel_format(useOffset=False)
    plt.show()


def pop_growth_modelling(replicates: int, initial_n: int, gr_method: str, generations: int,
                         gaussian_params: Dict[str, float], uniform_params: Dict[str, float]) -> pd.DataFrame:
    colnames = ['fixed_final_n', 'random_final_n', 'fixed/random ratio']
    df = pd.DataFrame(columns=colnames)
    for _ in range(replicates):
        gr_sample = get_growth_rate_sample(gr_method=gr_method, generations=generations,
                                           gaussian_params=gaussian_params, uniform_params=uniform_params)
        fixed_final_n = calculate_fixed_final_n(initial_n=initial_n, gr_sample=gr_sample)
        random_final_n = calculate_random_final_n(initial_n=initial_n, gr_sample=gr_sample)
        perc = fixed_final_n / random_final_n
        df.loc[len(df), :] = pd.Series([fixed_final_n, random_final_n, perc], index=colnames)
    return df


def get_growth_rate_sample(gr_method: str, generations: int, gaussian_params: Dict[str, float],
                           uniform_params: Dict[str, float]) -> np.array:
    if generations < 1:
        return np.array([1.0])
    if gr_method == 'gaussian':
        return get_gaussian_growth_rate_sample(**gaussian_params, generations=generations)
    elif gr_method == 'uniform':
        return get_uniform_growth_rate_sample(**uniform_params, generations=generations)
    raise ValueError('gr_method must be one of: "gaussian", "uniform".')


def get_gaussian_growth_rate_sample(mi: float, sigma: float, generations: int) -> np.array:
    return np.random.normal(loc=mi, scale=sigma, size=generations)


def get_uniform_growth_rate_sample(lower: float, upper: float, generations: int) -> np.array:
    return np.random.uniform(low=lower, high=upper, size=generations)


def calculate_random_final_n(initial_n: int, gr_sample) -> int:
    return initial_n * gr_sample.cumprod()[-1]


def calculate_fixed_final_n(initial_n: int, gr_sample: np.array) -> int:
    return initial_n * (gr_sample.mean()**len(gr_sample))
