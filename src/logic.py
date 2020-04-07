from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import Workbook
from scipy.stats import sem, t

from src.validator import ExcelValidator

# Value from which to throw a warning of low N
N_WARNING_LEVEL = 20
# Vectorized round function
round_arr = np.vectorize(round)
# namedtuple encapsulating the return value
DynaFitReturn = namedtuple("DynaFitReturn", [''])

def dynafit(data: Workbook, filename: str, sheetname: str, need_to_calculate_gr: bool, time_delta: float,
            cs_start_cell: str, cs_end_cell: str, gr_start_cell: str, gr_end_cell: str, individual_colonies: int,
            large_colony_groups: int, bootstrap_repeats: int, add_confidence_interval: bool, confidence_value: float,
            remove_outliers: bool, add_violin: bool,  fig: plt.Figure, cvp_ax: plt.Axes,
            hist_ax: plt.Axes) -> Tuple[Dict[str, Any], np.array, np.array]:
    """Main DynaFit function"""
    # Store parameters used for DynaFit analysis
    results = {'file': filename, 'sheet': sheetname, 'max individual colony size': individual_colonies,
               'number of large colony groups': large_colony_groups, 'bootstrapping repeats': bootstrap_repeats}
    # Validate input data
    df = ExcelValidator(data=data, sheetname=sheetname, cs_start_cell=cs_start_cell, cs_end_cell=cs_end_cell,
                        gr_start_cell=gr_start_cell, gr_end_cell=gr_end_cell).data
    # Preprocess data
    if need_to_calculate_gr is True:
        df = calculate_growth_rate(df=df, time_delta=time_delta)
    df = filter_bad_data(df=df)
    if remove_outliers is True:
        df = filter_outliers(df=df)
    small_n_bins = check_for_small_sample_sizes(df=df)
    if small_n_bins:

    # Bin samples into groups and plot the resulting histogram
    binned_df = add_bins(df=df, individual_colonies=individual_colonies, bins=large_colony_groups)
    # TODO: implement warning (GUI blocking) when a bin contains a small number of colonies!
    plot_histogram(df=binned_df, ax=hist_ax)
    # Perform DynaFit bootstrap
    bootstrapped_df = bootstrap_data(df=binned_df, repeats=bootstrap_repeats)
    bootstrapped_df = add_log_columns(df=bootstrapped_df)
    # Extract and plot mean values
    xs, ys = get_mean_line_arrays(df=bootstrapped_df)
    plot_mean_line(xs=xs, ys=ys, ax=cvp_ax)
    plot_supporting_lines(xs=xs, ys=ys, ax=cvp_ax)
    # Extract and plot bootstrapping distributions
    plot_bootstrap_scatter(df=bootstrapped_df, ax=cvp_ax, individual_colonies=individual_colonies)
    if add_violin:
        plot_bootstrap_violins(df=bootstrapped_df, xs=xs, ax=cvp_ax, individual_colonies=individual_colonies)
    # Add CoDy values to results dictionary
    max_x_value = round(max(xs), 2)
    cody_range = [i for i in range(1, 7) if i < max_x_value]
    for i in cody_range:
        results[f'CoDy {i}'] = round(calculate_cody(xs=xs, ys=ys, cody_n=i), 4)
    results[f'CoDy {max_x_value}'] = round(calculate_cody(xs=xs, ys=ys, cody_n=None), 4)
    # Execute only if CI needs to be calculated
    if add_confidence_interval:
        # Extract and plot CI values
        upper_ys, lower_ys = get_confidence_interval_values(df=bootstrapped_df, confidence_value=confidence_value)
        plot_mean_line_ci(xs=xs, upper_ys=upper_ys, lower_ys=lower_ys, ax=cvp_ax)
        plot_supporting_lines_ci(xs=xs, upper_ys=upper_ys, lower_ys=lower_ys, ax=cvp_ax)
        # Add CoDy CI values to results dictionary
        for ys, name in zip([upper_ys, lower_ys], ['upper', 'lower']):
            for i in cody_range:
                results[f'CoDy {i} {name} CI'] = round(calculate_cody(xs=xs, ys=ys, cody_n=i), 4)
            results[f'CoDy {max_x_value} {name} CI'] = round(calculate_cody(xs=xs, ys=ys, cody_n=None), 4)
    # Format plot labels
    fig.suptitle(f'CVP - Exp: {filename}, Sheet: {sheetname}')
    cvp_ax.set_xlabel('log2(Colony Size)')
    cvp_ax.set_ylabel('log2(Growth Rate variance)')
    # Return results dictionary and mean XY arrays
    return results, round_arr(xs, 4), round_arr(ys, 4)


def calculate_growth_rate(df: pd.DataFrame, time_delta: float) -> pd.DataFrame:
    """Calculates GR values from CS1 and CS2 and a time delta. These values are place on the "GR" column (which
    initially contained the CS2 values)."""
    growth_rate = (np.log2(df['GR']) - np.log2(df['CS'])) / (time_delta / 24)
    if growth_rate.isna().any():  # happens if the log of CS1 or CS2 is negative - which doesn't make sense anyway
        raise ValueError('Growth rate could not be calculated from the given colony size ranges. '
                         'Did you mean to select Colony Size and Growth Rate instead? '
                         'Please check your selected column ranges.')
    return df.assign(GR=growth_rate)


def filter_bad_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter low CS values (colonies with less than 1 cell should not exist anyway)."""
    return df.loc[df['CS'] >= 1]


def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Filters GR outliers using Tukey's boxplot method (with a Tukey factor of 3.0)."""
    q1, q3 = df['GR'].quantile([0.25, 0.75])
    iqr = abs(q3-q1)
    tf = 3
    upper_cutoff = q3 + (iqr * tf)
    lower_cutoff = q1 - (iqr * tf)
    return df.loc[df['GR'].between(lower_cutoff, upper_cutoff)]


def add_bins(df: pd.DataFrame, individual_colonies: int, bins: int) -> pd.DataFrame:
    """Returns the data DataFrame with a "bins" column, which divides the population into groups (bins) of cells.
    Colonies with size <= the "individual_colonies" parameters are grouped with colonies with the same number of cells,
    while larger colonies are split into N groups (N=bins) with a close number of instances in each one of them."""
    bin_condition = df['CS'] > individual_colonies
    single_bins = df.loc[~bin_condition]['CS']
    try:
        multiple_bins = pd.qcut(df.loc[bin_condition]['CS'], bins, labels=False) + (individual_colonies + 1)
    except ValueError:
        mes = (f'Could not divide the large CS population into {bins} unique groups. ' 
               'Please reduce the value of the "number_of_bins" parameter and try again.')
        raise TooManyBinsError(mes)
    return df.assign(bins=pd.concat([single_bins, multiple_bins]))


def bootstrap_data(df: pd.DataFrame, repeats: int) -> pd.DataFrame:
    """Performs bootstrapping. Each bin is sampled N times (N="repeats" parameter)."""
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
    return output_df


def add_log_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Adds columns with the log2-transformed values of colony size means and growth rate variances."""
    return df.assign(log2_CS_mean=np.log2(df['CS_mean']), log2_GR_var=np.log2(df['GR_var']))


def get_mean_line_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Returns a tuple of numpy arrays representing the X and Y values of the mean line in the CVP, respectively."""
    xs = df.groupby('bins').mean()['log2_CS_mean'].values
    ys = df.groupby('bins').mean()['log2_GR_var'].values
    return xs, ys


def plot_supporting_lines(xs: np.ndarray, ys: np.ndarray, ax: plt.Axes) -> None:
    """Plots the three supporting lines of the CVP."""
    start_x, end_x = get_start_end_values(array=xs)
    start_y, end_y = get_start_end_values(array=ys)
    plot_h0(start=start_x, end=end_x, initial_height=start_y, ax=ax)
    plot_h1(start=start_x, end=end_x, initial_height=start_y, ax=ax)
    plot_vertical_axis(start=start_x, ax=ax)


def plot_h0(start: float, end: float, initial_height: float, ax: plt.Axes) -> None:
    """Plots H0 on the CVP (horizontal blue line)"""
    ax.plot([start, end], [initial_height, initial_height], color='blue', lw=3)


def plot_h1(start: float, end: float, initial_height: float, ax: plt.Axes) -> None:
    """Plots H1 on the CVP (diagonal red line)"""
    final_height = get_missing_coord(x1=start, y1=initial_height, x2=end)
    ax.plot([start, end], [initial_height, final_height], color='red', lw=3)


def get_missing_coord(x1: float, y1: float, x2: float, angular_coefficient: float = -1) -> float:
    """Returns the y2 coordinate at the point (x2, y2) of a line which has an angular coefficient of
    "angular_coefficient" and passes through the point (x1, y1)."""
    linear_coefficient = y1 - (angular_coefficient * x1)
    y2 = linear_coefficient - x2
    return y2


def plot_vertical_axis(start: float, ax: plt.Axes) -> None:
    """Plots a bold vertical Y axis on the left limit of the CVP plot."""
    ax.axvline(start, color='darkgray', lw=3, zorder=0)


def plot_supporting_lines_ci(xs: np.ndarray, upper_ys: np.ndarray, lower_ys: np.ndarray, ax: plt.Axes) -> None:
    """Plots the CI for the supporting lines of the CVP."""
    start_x, end_x = get_start_end_values(array=xs)
    upper_start_y, upper_end_y = get_start_end_values(array=upper_ys)
    lower_start_y, lower_end_y = get_start_end_values(array=lower_ys)
    plot_h0_ci(start=start_x, end=end_x, upper=upper_start_y, lower=lower_start_y, ax=ax)
    plot_h1_ci(start=start_x, end=end_x, upper=upper_start_y, lower=lower_start_y, ax=ax)


def plot_h0_ci(start: float, end: float, upper: float, lower: float, ax: plt.Axes) -> None:
    """Plots H0 confidence interval on the CVP (diagonal red line)"""
    ax.fill_between([start, end], [upper, upper], [lower, lower], color='blue', alpha=0.3)


def plot_h1_ci(start: float, end: float, upper: float, lower: float, ax: plt.Axes) -> None:
    """Plots H1 confidence interval on the CVP (diagonal red line)"""
    final_height_upper = get_missing_coord(x1=start, y1=upper, x2=end)
    final_height_lower = get_missing_coord(x1=start, y1=lower, x2=end)
    ax.fill_between([start, end], [upper, final_height_upper], [lower, final_height_lower], color='red', alpha=0.3)


def plot_bootstrap_violins(df: pd.DataFrame, xs: np.ndarray, ax: plt.Axes, individual_colonies: int) -> None:
    """Plots the bootstrap populations for each bin as violin plots."""
    ys = [df.loc[df['bins'] == b]['log2_GR_var'] for b in sorted(df['bins'].unique())]
    parts = ax.violinplot(positions=xs, dataset=ys, showmeans=False, showmedians=False, showextrema=False)
    for i, body in enumerate(parts['bodies'], 1):
        body.set_facecolor('red') if i <= individual_colonies else body.set_facecolor('gray')
        body.set_edgecolor('black')
        body.set_alpha(0.3)


def plot_bootstrap_scatter(df: pd.DataFrame, ax: plt.Axes, individual_colonies: int) -> None:
    """Plots bootstrap populations for each bin as scatter plots."""
    xs = df['log2_CS_mean']
    ys = df['log2_GR_var']
    facecolors = df['bins'].apply(lambda curr_bin: 'gray' if curr_bin > individual_colonies else 'red')
    ax.scatter(xs, ys, marker='.', edgecolor='k', facecolor=facecolors, alpha=0.3)


def plot_mean_line(xs: np.ndarray, ys: np.ndarray, ax: plt.Axes) -> None:
    """Plots the mean value for each bootstrapped population as a line plot."""
    ax.plot(xs, ys, color='green', alpha=0.9, lw=3)


def get_confidence_interval_values(df: pd.DataFrame, confidence_value: float) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates and returns the a tuple of arrays representing the Y values of the upper and lower confidence
    interval for the bootstrapped population."""
    upper_ys = []
    lower_ys = []
    for bin_number, bin_values in df.groupby('bins'):
        variance = bin_values['log2_GR_var']
        n = len(variance)
        m = variance.mean()
        std_err = sem(variance)
        h = std_err * t.ppf((1 + confidence_value) / 2, n - 1)
        upper_ys.append(m + h)
        lower_ys.append(m - h)
    return np.array(upper_ys), np.array(lower_ys)


def plot_mean_line_ci(xs: np.ndarray, upper_ys: np.ndarray, lower_ys: np.ndarray, ax: plt.Axes):
    """Plots the confidence interval around the mean line as a line plot."""
    ax.fill_between(xs, upper_ys, lower_ys, color='gray', alpha=0.5)


def plot_histogram(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plots a histogram of the colony size, indicating the "cuts" and group sizes made by the binning process."""
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


def get_start_end_values(array: Optional[List[float]] = None) -> Tuple[float, float]:
    """Returns the first and last value of the input array."""
    return array[0], array[-1]


def calculate_cody(xs: np.ndarray, ys: np.ndarray, cody_n: Optional[int]) -> float:
    """Returns the area above the curve (mean green line) up to the X coordinate equivalent to the "cody_n" parameter.
    If that parameter is a None value, uses the entire range of X values instead."""
    if cody_n is not None:  # Truncate arrays so that a specific CoDy value is calculated
        xs, ys = truncate_arrays(xs=xs, ys=ys, cutoff=cody_n)
    triangle_area = calculate_triangle_area(xs=xs, ys=ys)
    area_above_curve = trapezium_integration(xs=xs, ys=ys)
    return area_above_curve / triangle_area


def truncate_arrays(xs: np.ndarray, ys: np.ndarray, cutoff: int) -> Tuple[np.ndarray, np.ndarray]:
    """Truncates and returns the arrays "xs" and "ys" by removing values from the "xs" arrays lower than the cutoff,
    and the removing "ys"'s last elements until both arrays have the same size."""
    xs_trunc = [x for x in xs if x < cutoff] + [cutoff]
    end_y = np.interp(cutoff, xs, ys)
    ys_trunc = ys[:len(xs_trunc)] + [end_y]
    return xs_trunc, ys_trunc


def calculate_triangle_area(xs: np.ndarray, ys: np.ndarray) -> float:
    """Calculates the triangle area of the CVP, i.e. the are delimited by H0 and H1 up to the max X coordinate present
    in the "xs" array."""
    start_x, end_x = get_start_end_values(array=xs)
    triangle_length = abs(end_x - start_x)
    start_y, end_y = get_start_end_values(array=ys)
    final_height = get_missing_coord(x1=start_x, y1=start_y, x2=end_x)
    triangle_height = abs(start_y - final_height)
    triangle_area = (triangle_length * triangle_height) / 2
    return triangle_area


def trapezium_integration(xs: np.ndarray, ys: np.ndarray) -> float:
    """Performs trapezium integration over the XY series of coordinates (mean green line), calculating the area
    above the line and below H0. Any area above H0 is calculated as negative area."""
    integrated_area = 0
    for i, (x, y) in enumerate(zip(xs, ys)):
        try:
            next_x = xs[i+1]
            next_y = ys[i+1]
        except IndexError:  # Nothing more to add
            return integrated_area
        square = (next_x - x) * (ys[0] - y)  # SIGNED area
        triangle = (next_x - x) * (y - next_y) / 2  # SIGNED area
        trapezium = square + triangle
        integrated_area += trapezium


class TooManyBinsError(Exception):
    """Exception raised when samples cannot be found in the input file."""


class AbortedByUser(Exception):
    """Exception raised when user aborts the execution of DynaFit analysis."""
