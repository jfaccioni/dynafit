from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import Workbook

from src.validator import ExcelValidator


def dynafit(data: Workbook, filename: str, sheetname: str, is_raw_colony_sizes: bool, time_delta: float,
            cs_start_cell: str,  cs_end_cell: str, gr_start_cell: str, gr_end_cell: str, max_binned_colony_size: int,
            bins: int, repeats: int, sample_size: int, fig: plt.Figure, cvp_ax: plt.Axes,
            hist_ax: plt.Axes) -> Dict[str, Any]:
    """Main function of this script. Returns a dictionary of calculated CoDy values"""
    # Validate input data
    ev = ExcelValidator(data=data, sheetname=sheetname, cs_start_cell=cs_start_cell, cs_end_cell=cs_end_cell,
                        gr_start_cell=gr_start_cell, gr_end_cell=gr_end_cell)
    df = ev.data
    if is_raw_colony_sizes is True:
        df = calculate_growth_rate(df=df, time_delta=time_delta)
    df = filter_bad_data(df=df)
    # Run DynaFit analysis
    binned_df = add_bins(df=df, max_binned_colony_size=max_binned_colony_size, bins=bins)
    bootstrapped_df = bootstrap_data(df=binned_df, repeats=repeats, sample_size=sample_size)
    add_log_columns(df=bootstrapped_df)
    # Calculate plot parameters
    mean_line = get_mean_line_arrays(df=bootstrapped_df)
    # Plot DynaFit results
    plot_supporting_lines(mean_line=mean_line, ax=cvp_ax)
    plot_bootstrap_scatter(df=bootstrapped_df, ax=cvp_ax, max_binned_colony_size=max_binned_colony_size)
    plot_mean_line(mean_line=mean_line, ax=cvp_ax)
    plot_histogram(df=binned_df, ax=hist_ax)
    # Format labels
    fig.suptitle(f'CVP - Exp: {filename}, Sheet: {sheetname}')
    cvp_ax.set_xlabel('log2(Colony Size)')
    cvp_ax.set_ylabel('log2(Growth Rate variance)')
    # Get results values
    results = {'filename': filename, 'sheet': sheetname, 'max binned colony size': max_binned_colony_size, 'bins': bins,
               'repeats': repeats, 'sample_size': sample_size, 'sample size': sample_size}
    # Get CoDy values
    for i in range(1, 7):
        results[f'CoDy {i}'] = calculate_cody(mean_line=mean_line, cody_n=i)
    results['CoDy inf'] = calculate_cody(mean_line=mean_line, cody_n=None)
    return results


def calculate_growth_rate(df: pd.DataFrame, time_delta: float) -> pd.DataFrame:
    """Calculates GR values from CS1 and CS2"""
    growth_rate = (np.log2(df['GR']) - np.log2(df['CS'])) / (time_delta / 24)
    if growth_rate.isna().any():  # happens if the log of CS1 or CS2 is negative - which doesn't make sense anyway
        raise ValueError('Growth rate could not be calculated from the given colony size ranges. '
                         'Did you mean to select Colony Size and Growth Rate instead? '
                         'Please check your column ranges.')
    return df.assign(GR=growth_rate)


def filter_bad_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter low CS values (colonies with less than 1 cell should not exist)"""
    return df.loc[df['CS'] >= 1]


def add_bins(df: pd.DataFrame, max_binned_colony_size: int, bins: int) -> pd.DataFrame:
    """Returns the data DataFrame with a "bins" column, which divides the population of values in
    bins with a close number of instances in them"""
    bin_condition = df['CS'] > max_binned_colony_size
    single_bins = df.loc[~bin_condition]['CS']
    multiple_bins = pd.qcut(df.loc[bin_condition]['CS'], bins, labels=False) + (max_binned_colony_size + 1)
    single_intervals = pd.interval_range(start=0, end=max_binned_colony_size)
    multiple_intervals = pd.qcut(df.loc[bin_condition]['CS'], bins)
    return df.assign(bins=pd.concat([single_bins, multiple_bins]),
                     intervals=pd.concat([single_intervals, multiple_intervals]))


def bootstrap_data(df: pd.DataFrame, repeats: int, sample_size: int) -> pd.DataFrame:
    """Performs bootstrapping"""
    columns = ['CS_mean', 'GR_var', 'bins']
    output_df = pd.DataFrame(columns=columns)
    for bin_number, bin_values in df.groupby('bins'):
        for repeat in range(repeats):
            sample = bin_values.sample(n=sample_size, replace=True)
            row = pd.Series([sample['CS'].mean(), sample['GR'].var(), bin_number], index=columns)
            output_df = output_df.append(row, ignore_index=True)
    return output_df


def add_log_columns(df: pd.DataFrame) -> None:
    """Adds columns with log2 values in-place"""
    df['log2_CS_mean'] = np.log2(df['CS_mean'])
    df['log2_GR_var'] = np.log2(df['GR_var'])


def plot_bootstrap_scatter(df: pd.DataFrame, ax: plt.Axes, max_binned_colony_size: int) -> None:
    """Plots bootstrap result"""
    xs = df['log2_CS_mean']
    ys = df['log2_GR_var']
    facecolors = df['bins'].apply(lambda curr_bin: 'gray' if curr_bin > max_binned_colony_size else 'red')
    ax.scatter(xs, ys, marker='.', edgecolor='k', facecolor=facecolors, alpha=0.5)


def get_mean_line_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    xs = df.groupby('bins').mean()['log2_CS_mean'].values
    ys = df.groupby('bins').mean()['log2_GR_var'].values
    return xs, ys


def get_start_end_values(line: Tuple[np.ndarray, np.ndarray]) -> Tuple[float, float, float, float]:
    """Returns the max and min XY values of the mean line"""
    xs, ys = line
    start_x, end_x = xs[0], xs[-1]
    start_y, end_y = ys[0], ys[-1]
    return start_x, end_x, start_y, end_y


def plot_mean_line(mean_line: Tuple[np.ndarray, np.ndarray], ax: plt.Axes) -> None:
    """Plots mean line for all data in ax."""
    ax.plot(*mean_line, color='green', alpha=0.9, lw=3)


def plot_supporting_lines(ax: plt.Axes, mean_line: Tuple[np.ndarray, np.ndarray]) -> None:
    """Plots the three supporting lines of the CVP."""
    start_x, end_x, start_y, _ = get_start_end_values(line=mean_line)
    # horizontal blue line (H0)
    ax.plot([start_x, end_x], [start_y, start_y], color='blue', lw=3)
    # diagonal red line (H1)
    linear_y = -1 * end_x + start_y  # y = ax + b, a=1, b=start_y
    ax.plot([start_x, end_x], [start_y, linear_y], color='red', lw=3)
    # vertical black line (Y axis)
    ax.axvline(start_x, color='darkgray', lw=3, zorder=0)


def plot_histogram(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plots a histogram of the colony size, indicating the "cuts" made by the binning process"""
    ax.hist(np.log2(df['CS']))
    grouped_data = df.groupby('bins')
    positions = np.log2(grouped_data.max()['CS'])
    bin_min_labels = np.core.defchararray.add('> ', np.roll(grouped_data.max()['CS'], 1).astype(int).astype(str))
    bin_min_labels[0] = '> 0'
    bin_max_labels = np.core.defchararray.add('<= ', grouped_data.max()['CS'].astype(int).astype(str))
    bin_max_labels[-1] = '<= inf'
    number_of_instances = grouped_data.count()['CS']
    for pos, bin_min, bin_max, num in zip(positions, bin_min_labels, bin_max_labels, number_of_instances):
        ax.axvline(pos, c='k')
        text = f'{bin_min}\n{bin_max}\nn={num}'
        ax.text(pos, ax.get_ylim()[1] * 0.5, text)


def calculate_cody(mean_line: Tuple[np.ndarray, np.ndarray], cody_n: Optional[int]) -> float:
    """Returns the area above the curve (mean green line) up to a maximal x position of 2**cody_n. If rcodiff
     is none, use the whole range of XY values instead"""
    xs, ys = mean_line
    start_x, end_x, start_y, _ = get_start_end_values(line=mean_line)
    if cody_n is not None:  # Cap off CoDy value
        end_x, end_y = cody_n, np.interp(cody_n, xs, ys)
        xs = [x for x in xs if x <= end_x]
        ys = ys[:len(xs)]
    linear_y = -1 * end_x + start_y  # y = ax + b, a=1, b=start_y
    triangle_area = (abs(end_x - start_x) * abs(start_y - linear_y)) / 2
    return trapezium_integration(xs=xs, ys=ys) / triangle_area


def trapezium_integration(xs: np.ndarray, ys: np.ndarray):
    """Performs trapezium integration over the XY series of coordinates"""
    integrated_area = 0
    for i, (x, y) in enumerate(zip(xs, ys)):
        try:
            next_x = xs[i+1]
            next_y = ys[i+1]
        except IndexError:  # Nothing more to add
            return integrated_area
        square = (next_x - x) * (ys[0] - y)
        triangle = (next_x - x) * (y - next_y) / 2
        integrated_area += (square + triangle)
