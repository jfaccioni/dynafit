"""
create_pop_growth_dataset.py

Used to create a dataset of varying parameters of population growth runs.
"""

import pandas as pd

from typing import Dict, Any
from scripts.population_growth_modelling import pop_growth_modelling

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


def main(model_params: Dict[str, Any], gaussian_params: Dict[str, float], uniform_params: Dict[str, float]) -> None:
    initial_std = 0.01 / 3
    initial_uniform_range = 0.01
    out_df = pd.DataFrame(columns=['fixed_final_n', 'random_final_n', 'distribution', 'variation'])
    for scale in range(1, 101, 20):
        # get gaussian data
        gaussian_params['sigma'] = initial_std * scale
        gaussian_df = pop_growth_modelling(**model_params, gaussian_params=gaussian_params,
                                           uniform_params=uniform_params).loc[:, ['fixed_final_n', 'random_final_n']]
        gaussian_df['distribution'] = 'gaussian'
        gaussian_df['variation'] = gaussian_params['sigma']
        # get uniform data
        model_params['gr_method'] = 'uniform'
        uniform_params['lower'] = 1 - initial_uniform_range * scale
        uniform_params['upper'] = 1 + initial_uniform_range * scale
        uniform_df = pop_growth_modelling(**model_params, gaussian_params=gaussian_params,
                                          uniform_params=uniform_params).loc[:, ['fixed_final_n', 'random_final_n']]
        uniform_df['distribution'] = 'uniform'
        uniform_df['variation'] = uniform_params["lower"]
        out_df = pd.concat([out_df, gaussian_df, uniform_df])
    out_df.to_excel('multiplot_data.xlsx')


if __name__ == '__main__':
    main(model_params=SETTINGS['model_params'], gaussian_params=SETTINGS['gaussian_params'],
         uniform_params=SETTINGS['uniform_params'])
