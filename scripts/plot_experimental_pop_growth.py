import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
sns.set(context='notebook')

SETTINGS = {
    'path': ('/home/juliano/Dropbox/SHARED/DynaFit Experimental Cancer Cell/Submissions/'
             'Cancer Research Experimental/2nd Answer/Juliano/Figures/LS12/'),
    'filenames': {
        'min': 'experimental_pop_growth_stats_LS12gen20_2_MIN.xlsx',
        'avg': 'experimental_pop_growth_stats_LS12gen20_2_AVG.xlsx',
        'max': 'experimental_pop_growth_stats_LS12gen20_2_MAX.xlsx',
    },
    'initial_n': 1000,
    'plot_cutoff': 2000,
    'log': True
}

Color = Tuple[float, float, float, float]


def main(path: str, filenames: Dict[str, str], initial_n: int, plot_cutoff: int, log: bool) -> None:
    """Docstring"""
    dataframes = {
        label: pd.read_excel(os.path.join(path, filename))
        for label, filename in filenames.items()
    }
    print('Running lineplot...')
    plot_lineplot(dataframes=dataframes, initial_n=initial_n)
    print('Running violins...')
    plot_violins(dataframes=dataframes, initial_n=initial_n, plot_cutoff=plot_cutoff, log=log)
    print('Running relative violins...')
    plot_relative_violins(dataframes=dataframes, initial_n=initial_n, plot_cutoff=plot_cutoff, log=log)
    print('Showing...')
    plt.show()


def plot_lineplot(dataframes: Dict[str, pd.DataFrame], initial_n: int) -> None:
    """Docstring"""
    pal = sns.color_palette('rainbow', n_colors=20)
    colors = [pal[2], pal[15], pal[19]]
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    for (label, df), color in zip(reversed(dataframes.items()), reversed(colors)):
        df_gr, df_gr_sd = df.loc[:, 'Fixed_GR'].iloc[0], df.loc[:, 'Fixed_GR_SD'].iloc[0]
        ax.scatter(df.Static_GR_Final_N/initial_n, df.Dynamic_GR_Final_N/initial_n, color=color, s=10, alpha=0.7,
                   label=f'LS12, $\sigma = {round(df_gr_sd, 4)}$ ({label})')
    ax.scatter([1], [1], color='white', edgecolor='black', marker='*', s=200, label='start pos', zorder=2)
    ax.plot(ax.get_xlim(), ax.get_xlim(), lw=3, ls='--', color='k', alpha=0.7, zorder=0, label='$x = y$',
            scalex=False, scaley=False)
    fig.suptitle('Based on experimental data (LS12 untreated)\n'
                 '1000 initial cells, 20 generations, growth rate = '
                 r'$\frac{0.352}{day}$')
    ax.set_xlabel(r'Simulation result ($\frac{final}{initial}$) using fixed growth rate')
    ax.set_ylabel(r'Simulation result ($\frac{final}{initial}$) using dynamic growth rate')
    ax.legend()


def plot_violins(dataframes: Dict[str, pd.DataFrame], initial_n: int, plot_cutoff: int, log: bool) -> None:
    """Docstring"""
    sort_order = []
    for label, df in dataframes.items():
        df_gr_sd = df.loc[:, 'Fixed_GR_SD'].iloc[0]
        xlabel = f'{label} ($\sigma = {round(df_gr_sd, 4)}$)'
        df['Growth rate standard deviation'] = xlabel
        sort_order.append(xlabel)
    merged_df = pd.melt(
        pd.concat(dataframes),
        id_vars=['Growth rate standard deviation'],
        value_vars=['Static_GR_Final_N', 'Dynamic_GR_Final_N'],
        var_name='Method',
        value_name='Final N',
    )
    merged_df['Growth rate strategy'] = merged_df['Method'].apply(lambda x: x.split('_')[0])
    merged_df['N'] = merged_df['Final N'] / initial_n
    if log:
        merged_df['N'] = np.log2(merged_df['N'])
    if len(merged_df) > plot_cutoff:
        merged_df = merged_df.sample(n=plot_cutoff, replace=False)
    pal = sns.color_palette('rainbow', n_colors=20)
    colors = [pal[2], pal[15]]
    colors2 = [pal[4], pal[17]]
    fig, ax = plt.subplots()
    sns.violinplot(ax=ax, data=merged_df, x='Growth rate standard deviation', y='N', hue='Growth rate strategy',
                   palette=colors, split=True, order=sort_order)
    sns.swarmplot(ax=ax, data=merged_df, x='Growth rate standard deviation', y='N', hue='Growth rate strategy',
                  palette=colors2, dodge=True, alpha=0.7, size=3, order=sort_order)
    # ax.axvline(1, lw=2, ls='-', color='k', alpha=0.7)
    ax.set_xlabel(r'Number of cells on each replicate ($\frac{final}{initial}$)')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(title='Growth rate strategy', handles=handles[2:], labels=labels[2:])


def plot_relative_violins(dataframes: Dict[str, pd.DataFrame], initial_n: int, plot_cutoff: int, log: bool) -> None:
    """Docstring"""
    sort_order = []
    for label, df in dataframes.items():
        df_gr_sd = df.loc[:, 'Fixed_GR_SD'].iloc[0]
        xlabel = f'{label} ($\sigma = {round(df_gr_sd, 4)}$)'
        df['Growth rate standard deviation'] = xlabel
        sort_order.append(xlabel)
    merged_df = pd.concat(dataframes)
    merged_df['N'] = (merged_df['Static_GR_Final_N'] / initial_n) / (merged_df['Dynamic_GR_Final_N'] / initial_n)
    if log:
        merged_df['N'] = np.log2(merged_df['N'])
    if len(merged_df) > plot_cutoff:
        merged_df = merged_df.sample(n=plot_cutoff, replace=False)
    pal = sns.color_palette('rainbow', n_colors=20)
    colors = [pal[2], pal[15], pal[17]]
    colors2 = [pal[4], pal[17], pal[19]]
    fig, ax = plt.subplots()
    sns.violinplot(ax=ax, data=merged_df, x='Growth rate standard deviation', y='N', palette=colors, split=True,
                   order=sort_order)
    sns.swarmplot(ax=ax, data=merged_df, x='Growth rate standard deviation', y='N', palette=colors2, dodge=True,
                  edgecolor='gray', alpha=0.7, size=3, order=sort_order)
    ax.axhline(np.log2(1), lw=2, ls='-', color='k', alpha=0.7)
    ax.set_ylabel(r'Final number of cells on each replicate ($\frac{static}{dynamic}$)')


if __name__ == '__main__':
    main(**SETTINGS)
