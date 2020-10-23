from typing import Any, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scripts.population_growth_modelling import pop_growth_modelling

SETTINGS = {
    'initial_n_range': (100, 1001, 500),
    'generations_range': (100, 1001, 500),
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


def main(initial_n_range: Tuple[int, int, int], generations_range: Tuple[int, int, int],
         model_params: Dict[str, Any], gaussian_params: Dict[str, float], uniform_params: Dict[str, float]) -> None:
    """Main function of this script"""
    frames = []
    for method in ['gaussian', 'uniform']:
        model_params['gr_method'] = method
        batch_data = produce_batch_data(initial_n_range=initial_n_range, generations_range=generations_range,
                                        model_params=model_params, gaussian_params=gaussian_params,
                                        uniform_params=uniform_params)
        fix_numeric_data(data=batch_data)
        frames.append(batch_data)
    data = pd.concat(frames)
    plot_batch_data(data=data, model_params=model_params)
    plt.show()


def produce_batch_data(initial_n_range: Tuple[int, int, int], generations_range: Tuple[int, int, int],
                       model_params: Dict[str, Any], gaussian_params: Dict[str, float],
                       uniform_params: Dict[str, float]) -> pd.DataFrame:
    df = pd.DataFrame(columns=['fixed_final_n', 'random_final_n'])
    for initial_n in range(*initial_n_range):
        model_params['initial_n'] = initial_n
        for generations in range(*generations_range):
            model_params['generations'] = generations
            out_df = pop_growth_modelling(**model_params, gaussian_params=gaussian_params,
                                          uniform_params=uniform_params).loc[:, ['fixed_final_n', 'random_final_n']]
            out_df['initial_n'] = initial_n
            out_df['generations'] = generations
            out_df['distribution'] = model_params['gr_method']
            df = pd.concat([df, out_df])
    return df


def fix_numeric_data(data: pd.DataFrame) -> None:  # works in-place
    data.loc[data['fixed_final_n'] < 0.005]['fixed_final_n'] = 0.5
    data['fixed_final_n'] = np.log(data['fixed_final_n'].astype('float'))
    data.loc[data['random_final_n'] < 0.005].loc[:, 'random_final_n'] = 0.5
    data['random_final_n'] = np.log(data['random_final_n'].astype('float'))


def plot_batch_data(data: pd.DataFrame, model_params: Dict[str, Any]) -> None:
    g = sns.relplot(data=data, x='fixed_final_n', y='random_final_n', row='initial_n', col='generations',
                    hue='distribution', alpha=0.5)
    for ax_row in g.axes:
        for ax in ax_row:
            ax.plot(ax.get_xlim(), ax.get_xlim(), linestyle='--', color='k', lw=3, scalex=False, scaley=False, zorder=0)
            ax.scatter([model_params['initial_n']], [model_params['initial_n']], color='white', edgecolor='black',
                       marker='*', s=200, zorder=1)  # must check for the value on each axes
    plt.xlabel('Final N - Fixed GR (mean of fluctuating GR for the same replicate)')
    plt.ylabel('Final N - GR fluctuates on each generation')
    plt.ticklabel_format(useOffset=False)
    plt.show()


if __name__ == '__main__':
    main(**SETTINGS)
