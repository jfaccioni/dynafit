"""
pop_growth_oneshot.py

Calculates and plots a single run of the population growth simulation.
"""

from typing import Any, Dict

from scripts.population_growth_modelling import plot_results, pop_growth_modelling, save_to_excel

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


def main(save: bool, plot: bool, model_params: Dict[str, Any], gaussian_params: Dict[str, float],
         uniform_params: Dict[str, float]) -> None:
    """Main function of this script"""
    gaussian_df = pop_growth_modelling(**model_params, gaussian_params=gaussian_params, uniform_params=uniform_params)
    model_params['gr_method'] = 'uniform'
    uniform_df = pop_growth_modelling(**model_params, gaussian_params=gaussian_params, uniform_params=uniform_params)
    if save:
        save_to_excel(gaussian_df=gaussian_df, uniform_df=uniform_df)
    if plot:
        plot_results(gaussian_df=gaussian_df, uniform_df=uniform_df, model_params=model_params)


if __name__ == '__main__':
    main(**SETTINGS)
