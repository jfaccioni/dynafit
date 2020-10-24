from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scripts.population_growth_modelling import pop_growth_modelling

sns.set_context('paper')

SETTINGS = {
    'save': True,
    'show': False,
    'initial_n_list': [100, 250, 500, 750, 1000, 1500],
    'generations_list': [100, 250, 500, 750, 1000, 1500],
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


def main(save: bool, show: bool, initial_n_list: List[int], generations_list: List[int], model_params: Dict[str, Any],
         gaussian_params: Dict[str, float], uniform_params: Dict[str, float]) -> None:
    """Main function of this script"""
    fig_lin, axes_lin = plt.subplots(nrows=len(initial_n_list), ncols=len(generations_list),
                                     figsize=(24, 24))
    fig_log, axes_log = plt.subplots(nrows=len(initial_n_list), ncols=len(generations_list),
                                     figsize=(24, 24))
    figs = {'linear': fig_lin, 'log': fig_log}
    for initial_n, ax_lin_row, ax_log_row in zip(initial_n_list, axes_lin, axes_log):
        model_params['initial_n'] = initial_n
        for generations, ax_lin, ax_log in zip(generations_list, ax_lin_row, ax_log_row):
            model_params['generations'] = generations
            print(f'running with params {initial_n=}, {generations=}')
            gaussian_data = pop_growth_modelling(**model_params, gaussian_params=gaussian_params,
                                                 uniform_params=uniform_params)
            uniform_data = pop_growth_modelling(**model_params, gaussian_params=gaussian_params,
                                                uniform_params=uniform_params)
            plot(gauss=gaussian_data, unif=uniform_data, ax=ax_lin, initial_n=initial_n,
                 generations=generations, log=False)
            plot(gauss=to_log(gaussian_data), unif=to_log(uniform_data), ax=ax_log, initial_n=initial_n,
                 generations=generations, log=True)
            for name, figure in figs.items():
                format_figure(figure=figure, name=name)
    if show:
        plt.show()
    if save:
        for name, figure in figs.items():
            save_figure(figure=figure, name=name)
            save_resized_figure(figure=figure, name=name)


def to_log(data: pd.DataFrame) -> pd.DataFrame:
    data.loc[data['fixed_final_n'] < 0.005]['fixed_final_n'] = 0.5
    data['fixed_final_n'] = np.log(data['fixed_final_n'].astype('float'))
    data.loc[data['random_final_n'] < 0.005].loc[:, 'random_final_n'] = 0.5
    data['random_final_n'] = np.log(data['random_final_n'].astype('float'))
    return data


def plot(gauss: pd.DataFrame, unif: pd.DataFrame, ax: plt.Axes, initial_n: int, generations: int, log: bool) -> None:
    pal = sns.color_palette('rainbow')
    if log:
        ax.set_xscale('log')
        ax.set_yscale('log')
    gauss_color = pal[0]
    unif_color = pal[-1]
    ax.scatter(gauss.fixed_final_n, gauss.random_final_n, color=gauss_color, label='gaussian', alpha=0.7)
    ax.scatter(unif.fixed_final_n, unif.random_final_n, color=unif_color, label='uniform', alpha=0.7)
    start_pos = initial_n if not log else np.log(initial_n)
    ax.scatter(start_pos, start_pos, color='white', edgecolor='black', marker='*', s=200, label='start', zorder=1)
    ax.plot(ax.get_xlim(), ax.get_xlim(), linestyle='--', color='k', lw=3, scalex=False, scaley=False, zorder=0,
            label='$x = y$')
    ax.legend()
    if not log:
        ax.ticklabel_format(useOffset=False)
    ax.set_title(f'{initial_n=} {generations=}')


def format_figure(figure: plt.Figure, name: str) -> None:
    figure.suptitle(f'Comparison of fixed and fluctuating GR, 1000 replicates - {name} scale', size=20)
    figure.text(0.5, 0.04, 'Final N - Fixed GR', ha='center', va='center')
    figure.text(0.06, 0.5, 'Final N - Dynamic GR', ha='center', va='center', rotation='vertical')


def save_figure(figure: plt.Figure, name: str) -> None:
    print(f'saving figure {name}')
    figure.savefig(f'results/batch_results/figure_{name}.png')
    figure.savefig(f'results/batch_results/figure_{name}.pdf')
    figure.savefig(f'results/batch_results/figure_{name}.svg')


def save_resized_figure(figure: plt.Figure, name: str) -> None:
    print(f'saving rescaled figure {name}')
    scale = 0 if name == 'linear' else 1
    for ax in figure.axes:
        ax.set_xlim(left=scale)
        ax.set_ylim(bottom=scale)
    figure.savefig(f'results/batch_results/figure_{name}_scaled.png')
    figure.savefig(f'results/batch_results/figure_{name}_scaled.pdf')
    figure.savefig(f'results/batch_results/figure_{name}_scaled.svg')


if __name__ == '__main__':
    main(**SETTINGS)

