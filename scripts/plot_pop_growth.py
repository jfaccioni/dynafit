import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import seaborn as sns
from matplotlib.lines import Line2D

SETTINGS = {
    'input_file': 'multiplot_data.xlsx',
    'plot': True
}


def main(input_file: str, plot: bool) -> None:
    """Main function of this script"""
    df = pd.read_excel(input_file, index_col=0)
    gaussian_df = df.loc[df.distribution == 'gaussian']
    gaussian_b = get_b_var_values(data=gaussian_df)
    uniform_df = df.loc[df.distribution == 'uniform']
    uniform_df['variation'] = (1 - uniform_df['variation']) * 2  # fixes calculation of uniform range
    uniform_b = get_b_var_values(data=uniform_df)
    if plot:
        plot_individual_distribution(data=gaussian_df, name='Gaussian')
        plot_individual_distribution(data=uniform_df, name='Uniform')
        plot_lr(gaussian_data=gaussian_b, uniform_data=uniform_b)
        plt.show()


def plot_individual_distribution(data: pd.DataFrame, name: str) -> None:
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    sns.scatterplot([1000], [1000], color='white', edgecolor='black', marker='*', s=200, ax=ax,
                    label='start pos', zorder=2)
    sns.scatterplot(x='fixed_final_n', y='random_final_n', data=data,
                    ax=ax, hue='variation', palette='rainbow', zorder=1, alpha=0.7)
    sns.lineplot(plt.xlim(), plt.xlim(), ax=ax, linestyle='--', color='k', alpha=0.7, lw=3,
                 scalex=False, scaley=False, zorder=0, label='$x = y$')
    fig.suptitle(f'{name} distribution')
    ax.set_xlabel('Final N - Fixed GR (mean of dynamic GR for the same replicate)')
    ax.set_ylabel('Final N - Dynamic GR (fluctuates on each generation)')
    ax.legend().texts[2].set_text("$\sigma$" if name == 'Gaussian' else 'interval width ')


def get_b_var_values(data: pd.DataFrame) -> pd.DataFrame:
    out_df = pd.DataFrame(columns=['x', 'y', 'type', 'dist'])
    for group_value, group in data.groupby('variation'):
        xs = np.log(group.fixed_final_n)
        ys = np.log(group.random_final_n)
        slope, intercept, r_value, p_value, std_err = linregress(xs, ys)
        lr_df = pd.DataFrame({
            'x': [group_value, group_value],
            'y': [intercept, r_value],
            'type': ['intercept', 'r_value'],
            'dist': [group['distribution'][0], group['distribution'][0]]
        }, index=[len(out_df), len(out_df) + 1])
        out_df = pd.concat([out_df, lr_df])
    return out_df


def plot_lr(gaussian_data: pd.DataFrame, uniform_data: pd.DataFrame) -> None:
    pal = sns.color_palette('rainbow')
    gauss_color = pal[0]
    unif_color = pal[-1]
    data = pd.concat([gaussian_data, uniform_data])
    ax1_data = data.loc[data.type == 'intercept']
    ax2_data = data.loc[data.type == 'r_value']
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(ax1_data.loc[ax1_data.dist == 'gaussian'].x, ax1_data.loc[ax1_data.dist == 'gaussian'].y,
             color=gauss_color)
    # ax1.plot(ax1_data.loc[ax1_data.dist == 'uniform'].x, ax1_data.loc[ax1_data.dist == 'uniform'].y,
    #          color=unif_color)
    ax2.plot(ax2_data.loc[ax2_data.dist == 'gaussian'].x, ax2_data.loc[ax2_data.dist == 'gaussian'].y,
             color=gauss_color, linestyle='--')
    # ax2.plot(ax2_data.loc[ax2_data.dist == 'uniform'].x, ax2_data.loc[ax2_data.dist == 'uniform'].y,
    #          color=unif_color, linestyle='--')
    fig.suptitle(f'$b$ and $r$ values as a function of distribution variance')
    ax1.set_xlabel(f'$\sigma$')
    ax1.set_ylabel('$b$')
    ax2.set_ylabel('$r$')
    ax2.set_ylim(0, 1.05)
    ax1.legend(loc='lower left', handles=[
        Line2D([], [], color=gauss_color, linestyle='-', label='gaussian distribution'),
        # Line2D([], [], color=unif_color, linestyle='-', label='uniform distribution'),
        Line2D([], [], color='#232323', linestyle='--', label='$r$'),
        # Line2D([], [], color='#232323', linestyle='--', label='$b$')
    ])


if __name__ == '__main__':
    main(**SETTINGS)
