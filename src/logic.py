from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import Workbook
from scipy.stats import sem, t

from src.validator import ExcelValidator

# Value from which to throw a warning of low N
N_WARNING_LEVEL = 20


def dynafit(data: Workbook, filename: str, sheetname: str, is_raw_colony_sizes: bool, time_delta: float,
            cs_start_cell: str,  cs_end_cell: str, gr_start_cell: str, gr_end_cell: str, max_binned_colony_size: int,
            bins: int, repeats: int, show_violin: bool, show_ci: bool, filter_outliers: bool, confidence: float,
            fig: plt.Figure, cvp_ax: plt.Axes, hist_ax: plt.Axes) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Main function of this script. Returns a dictionary of calculated CoDy values"""
    upp = low = None
    # Validate input data
    ev = ExcelValidator(data=data, sheetname=sheetname, cs_start_cell=cs_start_cell, cs_end_cell=cs_end_cell,
                        gr_start_cell=gr_start_cell, gr_end_cell=gr_end_cell)
    df = ev.data
    if is_raw_colony_sizes is True:
        df = calculate_growth_rate(df=df, time_delta=time_delta)
    df = filter_bad_data(df=df)
    if filter_outliers is True:
        df = perform_outlier_filtering(df=df)
    # Run DynaFit analysis
    binned_df = add_bins(df=df, max_binned_colony_size=max_binned_colony_size, bins=bins)
    bootstrapped_df = bootstrap_data(df=binned_df, repeats=repeats)
    add_log_columns(df=bootstrapped_df)
    # Plot DynaFit results
    plot_supporting_lines(df=bootstrapped_df, ax=cvp_ax)
    if show_ci:
        upp, low = plot_confidence_intervals(df=bootstrapped_df, confidence=confidence, ax=cvp_ax)
        plot_supporting_lines_ci(df=bootstrapped_df, ax=cvp_ax, upper=upp[0], lower=low[0])
    if show_violin:
        plot_bootstrap_violins(df=bootstrapped_df, ax=cvp_ax, max_binned_colony_size=max_binned_colony_size)
    plot_bootstrap_scatter(df=bootstrapped_df, ax=cvp_ax, max_binned_colony_size=max_binned_colony_size)
    plot_mean_line(df=bootstrapped_df, ax=cvp_ax)
    plot_histogram(df=binned_df, ax=hist_ax)
    # Format labels
    fig.suptitle(f'CVP - Exp: {filename}, Sheet: {sheetname}')
    cvp_ax.set_xlabel('log2(Colony Size)')
    cvp_ax.set_ylabel('log2(Growth Rate variance)')
    # Get results values
    results = {'filename': filename, 'sheet': sheetname, 'max binned colony size': max_binned_colony_size,
               'bins': bins, 'repeats': repeats}
    # Get CoDy values
    max_x_value = round(get_mean_line_arrays(df=bootstrapped_df)[0][-1], 2)
    cody_range = [i for i in range(1, 7) if i < max_x_value]
    for i in cody_range:
        results[f'CoDy {i}'] = round(calculate_cody(df=bootstrapped_df, cody_n=i, yvals=None), 4)
    results[f'CoDy {max_x_value}'] = round(calculate_cody(df=bootstrapped_df, cody_n=None, yvals=None), 4)
    if show_ci:
        for i in cody_range:
            results[f'CoDy {i} upper CI'] = round(calculate_cody(df=bootstrapped_df, cody_n=i, yvals=upp), 4)
        results[f'CoDy {max_x_value} upper CI'] = round(calculate_cody(df=bootstrapped_df, cody_n=None, yvals=upp), 4)
        for i in cody_range:
            results[f'CoDy {i} lower CI'] = round(calculate_cody(df=bootstrapped_df, cody_n=i, yvals=low), 4)
        results[f'CoDy {max_x_value} lower CI'] = round(calculate_cody(df=bootstrapped_df, cody_n=None, yvals=low), 4)
    xs, ys = get_mean_line_arrays(df=bootstrapped_df)
    xs = [str(round(x, 4)) for x in xs]
    ys = [str(round(y, 4)) for y in ys]
    return results, xs, ys


def calculate_growth_rate(df: pd.DataFrame, time_delta: float) -> pd.DataFrame:
    """Calculates GR values from CS1 and CS2"""
    growth_rate = (np.log2(df['GR']) - np.log2(df['CS'])) / (time_delta / 24)
    if growth_rate.isna().any():  # happens if the log of CS1 or CS2 is negative - which doesn't make sense anyway
        raise ValueError('Growth rate could not be calculated from the given colony size ranges. '
                         'Did you mean to select Colony Size and Growth Rate instead? '
                         'Please check your selected column ranges.')
    return df.assign(GR=growth_rate)


def filter_bad_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter low CS values (colonies with less than 1 cell should not exist)"""
    return df.loc[df['CS'] >= 1]


def perform_outlier_filtering(df: pd.DataFrame) -> pd.DataFrame:
    """Filters GR outliers"""
    q1, q3 = df['GR'].quantile([0.25, 0.75])
    iqr = abs(q3-q1)
    tf = 3
    upper_cutoff = q3 + (iqr * tf)
    lower_cutoff = q1 - (iqr * tf)
    return df.loc[df['GR'].between(lower_cutoff, upper_cutoff)]


def add_bins(df: pd.DataFrame, max_binned_colony_size: int, bins: int) -> pd.DataFrame:
    """Returns the data DataFrame with a "bins" column, which divides the population of values in
    bins with a close number of instances in them"""
    bin_condition = df['CS'] > max_binned_colony_size
    single_bins = df.loc[~bin_condition]['CS']
    try:
        multiple_bins = pd.qcut(df.loc[bin_condition]['CS'], bins, labels=False) + (max_binned_colony_size + 1)
    except ValueError:
        mes = (f'Could not divide the large CS population into {bins} unique groups. ' 
               'Please reduce the value of the "number_of_bins" parameter and try again.')
        raise TooManyBinsError(mes)
    return df.assign(bins=pd.concat([single_bins, multiple_bins]))


def bootstrap_data(df: pd.DataFrame, repeats: int) -> pd.DataFrame:
    """Performs bootstrapping"""
    warns = []
    columns = ['CS_mean', 'GR_var', 'bins']
    output_df = pd.DataFrame(columns=columns)
    for bin_number, bin_values in df.groupby('bins'):
        n = len(bin_values)
        if n < N_WARNING_LEVEL:
            warns.append((bin_number, n))
        for repeat in range(repeats):
            sample = bin_values.sample(n=n, replace=True)
            row = pd.Series([sample['CS'].mean(), sample['GR'].var(), bin_number], index=columns)
            output_df = output_df.append(row, ignore_index=True)
    for bin_number, n in warns:
        # TODO: change this to actual GUI message
        print(f'Warning: group {bin_number} has small N (only {n} instances')
    return output_df


def add_log_columns(df: pd.DataFrame) -> None:
    """Adds columns with log2 values in-place"""
    df['log2_CS_mean'] = np.log2(df['CS_mean'])
    df['log2_GR_var'] = np.log2(df['GR_var'])


def plot_supporting_lines(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plots the three supporting lines of the CVP."""
    mean_line = get_mean_line_arrays(df=df)
    start_x, end_x, start_y, _ = get_start_end_values(line=mean_line)
    # horizontal blue line (H0)
    ax.plot([start_x, end_x], [start_y, start_y], color='blue', lw=3)
    # diagonal red line (H1)
    linear_y = -1 * end_x + start_y  # y = ax + b, a=1, b=start_y
    ax.plot([start_x, end_x], [start_y, linear_y], color='red', lw=3)
    # vertical black line (Y axis)
    ax.axvline(start_x, color='darkgray', lw=3, zorder=0)


def plot_supporting_lines_ci(df: pd.DataFrame, ax: plt.Axes, upper: float, lower: float) -> None:
    """Plots the CI for the supporting lines of the CVP."""
    mean_line = get_mean_line_arrays(df=df)
    start_x, end_x, start_y, _ = get_start_end_values(line=mean_line)
    # horizontal blue line (H0)
    ax.fill_between([start_x, end_x], [upper, upper], [lower, lower], color='blue', alpha=0.3)
    # diagonal red line (H1)
    upper_linear_y = -1 * end_x + upper  # y = ax + b, a=1, b=start_y
    lower_linear_y = -1 * end_x + lower  # y = ax + b, a=1, b=start_y
    ax.fill_between([start_x, end_x], [upper, upper_linear_y], [lower, lower_linear_y], color='red', alpha=0.3)


def plot_bootstrap_violins(df: pd.DataFrame, ax: plt.Axes, max_binned_colony_size: int) -> None:
    """Plots bootstrap result as violins"""
    xs, _ = get_mean_line_arrays(df=df)
    ys = [df.loc[df['bins'] == b]['log2_GR_var'] for b in sorted(df['bins'].unique())]
    parts = ax.violinplot(positions=xs, dataset=ys, showmeans=False, showmedians=False, showextrema=False)
    for i, body in enumerate(parts['bodies'], 1):
        body.set_facecolor('red') if i <= max_binned_colony_size else body.set_facecolor('gray')
        body.set_edgecolor('black')
        body.set_alpha(0.3)


def plot_bootstrap_scatter(df: pd.DataFrame, ax: plt.Axes, max_binned_colony_size: int) -> None:
    """Plots bootstrap result as scatter"""
    xs = df['log2_CS_mean']
    ys = df['log2_GR_var']
    facecolors = df['bins'].apply(lambda curr_bin: 'gray' if curr_bin > max_binned_colony_size else 'red')
    ax.scatter(xs, ys, marker='.', edgecolor='k', facecolor=facecolors, alpha=0.3)


def plot_mean_line(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plots mean line for all data in ax."""
    mean_line = get_mean_line_arrays(df=df)
    ax.plot(*mean_line, color='green', alpha=0.9, lw=3)


def plot_confidence_intervals(df: pd.DataFrame, confidence: float, ax: plt.Axes) -> Tuple[List[float], List[float]]:
    """Plots CI"""
    xs, ys = get_mean_line_arrays(df=df)
    upper_ci = []
    lower_ci = []
    for bin_number, bin_values in df.groupby('bins'):
        variance = bin_values['log2_GR_var']
        n = len(variance)
        m = variance.mean()
        std_err = sem(variance)
        h = std_err * t.ppf((1 + confidence) / 2, n - 1)
        upper_ci.append(m + h)
        lower_ci.append(m - h)
    ax.fill_between(xs, upper_ci, lower_ci, color='gray', alpha=0.5)
    return upper_ci, lower_ci


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


def calculate_cody(df: pd.DataFrame, cody_n: Optional[int], yvals: Optional[List[float]]) -> float:
    """Returns the area above the curve (mean green line) up to a maximal x position of 2**cody_n. If rcodiff
     is none, use the whole range of XY values instead"""
    xs, ys = get_mean_line_arrays(df=df)
    if yvals is not None:
        ys = yvals
    start_x, end_x, start_y, _ = get_start_end_values(line=(xs, ys))
    if cody_n is not None:  # Cap off CoDy value
        end_x, end_y = cody_n, np.interp(cody_n, xs, ys)
        xs = [x for x in xs if x < end_x] + [end_x]
        ys = ys[:len(xs)] + [end_y]
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


class TooManyBinsError(Exception):
    """Exception raised when samples cannot be found in the input file."""
    def __init__(self, *args):
        super().__init__(*args)
