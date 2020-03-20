from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import Workbook

from src.validator import ExcelValidator


def dynafit(data: Workbook, sheetname: str, cs_start_cell: str, cs_end_cell: str, gr_start_cell: str, gr_end_cell: str,
            max_binned_colony_size: int, bins: int, repeats: int, sample_size: int, fig: plt.Figure, cvp_ax: plt.Axes,
            hist_ax: plt.Axes) -> None:
    """Main function of this script"""
    # Run DynaFit analysis
    df = ExcelValidator(data=data, sheetname=sheetname, cs_start_cell=cs_start_cell, cs_end_cell=cs_end_cell,
                        gr_start_cell=gr_start_cell, gr_end_cell=gr_end_cell).data
    binned_df = add_bins(df=df, max_binned_colony_size=max_binned_colony_size, bins=bins)
    bootstrapped_df = bootstrap_data(df=binned_df, repeats=repeats, sample_size=sample_size)
    # Calculate mean line
    mean_line = get_mean_line_arrays(df=bootstrapped_df)
    # Plot DynaFit results
    plot_bootstrap_scatter(df=bootstrapped_df, ax=cvp_ax)
    plot_supporting_lines(mean_line=mean_line, ax=cvp_ax)
    plot_mean_line(ax=cvp_ax, mean_line=mean_line)
    plot_histogram(df=binned_df, ax=hist_ax)
    # Format resulting plot
    fig.suptitle(get_plot_title(repeats=repeats, sample_size=sample_size))
    cvp_ax.set_xlabel('log2(Colony Size)')
    cvp_ax.set_ylabel('log2(Growth Rate variance)')
    # Return AAC value to GUI
    # return area_above_curve(df=df)


def add_bins(df: pd.DataFrame, max_binned_colony_size: int, bins: int) -> pd.DataFrame:
    """Returns the data DataFrame with a "bins" column, which divides the population of values in
    bins with a close number of instances in them"""
    df['bins'] = df['CS']
    bin_condition = df['bins'] > max_binned_colony_size
    binned_data = pd.qcut(df.loc[bin_condition]['CS'], bins, labels=False)
    binned_data += max_binned_colony_size + 1
    binned_data = pd.concat([binned_data, df.loc[~bin_condition]['bins']])
    return df.assign(bins=binned_data)


def bootstrap_data(df: pd.DataFrame, repeats: int, sample_size: int) -> pd.DataFrame:
    """Performs bootstrapping"""
    columns = ['CS_mean', 'GR_var', 'bins']
    output_df = pd.DataFrame(columns=columns)
    for bin_number, bin_values in df.groupby('bins'):
        for repeat in range(repeats):
            sample = bin_values.sample(n=sample_size, replace=True)
            row = pd.Series([sample['CS'].mean(), sample['GR'].var(), bin_number], index=columns)
            output_df = output_df.append(row, ignore_index=True)
    return output_df.assign(log2_CS_mean=np.log2(output_df['CS_mean']), log2_GR_var=np.log2(output_df['GR_var']))


def plot_bootstrap_scatter(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plots bootstrap result"""
    xs = df['log2_CS_mean']
    ys = df['log2_GR_var']
    ax.scatter(xs, ys, marker='.', edgecolor='k', facecolor='gray')


def get_mean_line_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    xs = df.groupby('bins').mean()['log2_CS_mean']
    ys = df.groupby('bins').mean()['log2_GR_var']
    return xs, ys


def plot_mean_line(ax: plt.Axes, mean_line: Tuple[np.ndarray, np.ndarray]) -> None:
    """Plots mean line for all data in ax."""
    xs, ys = mean_line
    ax.plot(xs, ys, color='green', alpha=0.9, lw=3)


def plot_supporting_lines(ax: plt.Axes, mean_line: Tuple[np.ndarray, np.ndarray]) -> None:
    """Plots the three supporting lines of the CVP."""
    xs, ys = mean_line
    max_x, min_x = xs.max(), xs.min()
    max_y, min_y = ys.max(), ys.min()
    ax.plot([min_x, max_x], [max_y, max_y], color='blue', lw=3)
    ax.plot([min_x, max_x], [max_y, min_y], color='red', lw=3)
    ax.axvline(min_x, color='black', lw=3)


def get_plot_title(repeats, sample_size) -> str:
    """Returns a string for the plot's title."""
    return f'CVP: sampling repeats={repeats}, sample size={sample_size}'


def plot_histogram(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plots a histogram of the colony size, indicating the "cuts" made by the binning process"""
    ax.hist(np.log2(df['CS']))
    grouped_data = df.groupby('bins')
    positions = np.log2(grouped_data.max()['CS'])
    bin_min_labels = np.core.defchararray.add('> ', np.roll(grouped_data.max()['CS'], 1).astype(str))
    bin_min_labels[0] = '> 0'
    bin_max_labels = np.core.defchararray.add('<= ', grouped_data.max()['CS'].astype(str))
    bin_max_labels[-1] = '<= inf'
    number_of_instances = grouped_data.count()['CS']
    for pos, bin_min, bin_max, num in zip(positions, bin_min_labels, bin_max_labels, number_of_instances):
        ax.axvline(pos, c='k')
        text = f'{bin_min}\n{bin_max}\nn={num}'
        ax.text(pos, ax.get_ylim()[1] * 0.5, text)


def area_above_curve() -> None:
    """Returns the area above the curve (mean green line)."""
    # TODO: calculate this
    # triangle_area = max_x * () / 2
    pass
