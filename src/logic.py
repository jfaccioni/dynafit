import time
from itertools import count
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from PySide2.QtCore import QEvent, Signal
from openpyxl import Workbook
from scipy.stats import sem, t

from src.plotter import Plotter
from src.utils import get_missing_coord, get_start_end_values
from src.validator import ExcelValidator

# Value from which to throw a warning of low N
N_WARNING_LEVEL = 20
# Vectorized round function
round_arr = np.vectorize(round)


def dynafit(data: Workbook, filename: str, sheetname: str, need_to_calculate_gr: bool, time_delta: float,
            cs_start_cell: str, cs_end_cell: str, gr_start_cell: str, gr_end_cell: str, individual_colonies: int,
            large_colony_groups: int, bootstrap_repeats: int, add_confidence_interval: bool, confidence_value: float,
            remove_outliers: bool, add_violin: bool, progress_callback: Signal,
            ss_warning_callback: Signal) -> Tuple[Plotter, pd.DataFrame, Tuple[str, str]]:
    """Main DynaFit function"""
    # Store parameters used for DynaFit analysis
    results_dict = {'file': filename, 'sheet': sheetname, 'max individual colony size': individual_colonies,
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
    # Bin samples into groups and plot the resulting histogram
    binned_df = add_bins(df=df, individual_colonies=individual_colonies, bins=large_colony_groups)
    # Sample size warning
    warns = sample_size_warning(df=binned_df)
    if warns:
        answered_event = QEvent(QEvent.WindowActivate)
        user_wants_to_continue_event = QEvent(QEvent.WindowActivate)
        answered_event.setAccepted(False)
        ss_warning_callback.emit((answered_event, user_wants_to_continue_event, warns))
        while not answered_event.isAccepted():
            time.sleep(0.1)
            if not user_wants_to_continue_event.isAccepted():
                raise AbortedByUser("User decided to stop DynaFit analysis.")
    # Perform DynaFit bootstrap
    df = bootstrap_data(df=binned_df, repeats=bootstrap_repeats, progress_callback=progress_callback)
    df = add_log_columns(df=df)
    # Base results
    xs, ys = get_mean_line_arrays(df=df)
    scatter_xs = df['log2_CS_mean'].values
    scatter_ys = df['log2_GR_var'].values
    scatter_colors = df['bins'].apply(lambda curr_bin: 'gray' if curr_bin > individual_colonies else 'red').values
    # Add CoDy values to dataframe_results dictionary
    max_x_value = round(max(xs), 2)
    cody_range = [i for i in range(1, 7) if i < max_x_value]
    for i in cody_range:
        results_dict[f'CoDy {i}'] = round(calculate_cody(xs=xs, ys=ys, cody_n=i), 4)
    results_dict[f'CoDy {max_x_value}'] = round(calculate_cody(xs=xs, ys=ys, cody_n=None), 4)
    # Violins
    violin_ys, violin_colors = None, None
    if add_violin:
        violin_ys = [df.loc[df['bins'] == b]['log2_GR_var'].values for b in sorted(df['bins'].unique())]
        violin_colors = individual_colonies
    # CI
    upper_ys, lower_ys = None, None
    if add_confidence_interval:
        upper_ys, lower_ys = get_confidence_interval_values(df=df, confidence_value=confidence_value)
        # Add CoDy CI values to dataframe_results dictionary
        for ys, name in zip([upper_ys, lower_ys], ['upper', 'lower']):
            for i in cody_range:
                results_dict[f'CoDy {i} {name} CI'] = round(calculate_cody(xs=xs, ys=ys, cody_n=i), 4)
            results_dict[f'CoDy {max_x_value} {name} CI'] = round(calculate_cody(xs=xs, ys=ys, cody_n=None), 4)
    # Other histogram values
    hist_x = np.log2(binned_df['CS'])
    hist_pos, hist_bin_mins, hist_bin_maxs, hist_instances = get_histogram_values(df=binned_df)
    # Get and return results
    plot_results = Plotter(mean_xs=xs, mean_ys=ys, upper_ys=upper_ys, lower_ys=lower_ys, scatter_xs=scatter_xs,
                           scatter_ys=scatter_ys, scatter_colors=scatter_colors, violin_ys=violin_ys,
                           violin_colors=violin_colors, hist_x=hist_x, hist_pos=hist_pos, hist_bin_mins=hist_bin_mins,
                           hist_bin_maxs=hist_bin_maxs, hist_instances=hist_instances)
    dataframe_results = results_to_dataframe(results_dict=results_dict, xs=round_arr(xs), ys=round_arr(ys))
    plot_title_info = filename, sheetname
    return plot_results, dataframe_results, plot_title_info


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


def sample_size_warning(df: pd.DataFrame) -> Dict[float, float]:
    warns = {}
    for bin_number, bin_values in df.groupby('bins'):
        n = len(bin_values)
        if n < N_WARNING_LEVEL:
            warns[bin_number] = n
    return warns


def bootstrap_data(df: pd.DataFrame, repeats: int, progress_callback: Signal) -> pd.DataFrame:
    """Performs bootstrapping. Each bin is sampled N times (N="repeats" parameter)."""
    total_progress = repeats * df['bins'].max()
    current_progress = count()
    columns = ['CS_mean', 'GR_var', 'bins']
    output_df = pd.DataFrame(columns=columns)
    for bin_number, bin_values in df.groupby('bins'):
        n = len(bin_values)
        for repeat in range(repeats):
            progress = int(round(100 * next(current_progress) / total_progress))
            progress_callback.emit(progress)
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


def get_histogram_values(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns values for the histogram of the colony size, indicating the group sizes made by the binning process."""
    grouped_data = df.groupby('bins')
    positions = np.log2(grouped_data.max()['CS'])
    bin_min_labels = np.core.defchararray.add('> ', np.roll(grouped_data.max()['CS'], 1).astype(int).astype(str))
    bin_min_labels[0] = '> 0'
    bin_max_labels = np.core.defchararray.add('<= ', grouped_data.max()['CS'].astype(int).astype(str))
    bin_max_labels[-1] = '<= inf'
    number_of_instances = grouped_data.count()['CS'].values
    return positions, bin_min_labels, bin_max_labels, number_of_instances


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


def results_to_dataframe(results_dict: Dict[str, Any], xs: np.ndarray, ys: np.ndarray) -> pd.DataFrame:
    """Saves DynaFit dataframe_results as a pandas DataFrame (used for Excel/csv export)."""
    size = max(len(results_dict), len(xs), len(ys))
    params = list(results_dict.keys()) + [np.nan for _ in range(size - len(results_dict))]
    values = list(results_dict.values()) + [np.nan for _ in range(size - len(results_dict))]
    xs = list(xs) + [np.nan for _ in range(size - len(xs))]
    ys = list(ys) + [np.nan for _ in range(size - len(ys))]
    return pd.DataFrame({'Parameter': params, 'Value': values, 'Mean X Values': xs, 'Mean Y Values': ys})


class TooManyBinsError(Exception):
    """Exception raised when samples cannot be found in the input file."""


class AbortedByUser(Exception):
    """Exception raised when user aborts the execution of DynaFit analysis."""
