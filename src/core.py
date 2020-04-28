"""core.py - bundles all computations of DynaFit."""

from itertools import count
from queue import Queue
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from PySide2.QtCore import Signal  # noqa
from openpyxl import Workbook
from scipy.stats import sem, t

from src.exceptions import AbortedByUser, TooManyGroupsError
from src.plotter import Plotter
from src.utils import get_missing_coord, get_start_end_values
from src.validator import ExcelValidator

# Value from which to throw a warning of low N
WARNING_LEVEL = 20


def dynafit(workbook: Workbook, filename: str, sheetname: str, must_calculate_growth_rate: bool, time_delta: float,
            cs_start_cell: str, cs_end_cell: str, gr_start_cell: str, gr_end_cell: str, individual_colonies: int,
            large_colony_groups: int, bootstrap_repeats: int, show_ci: bool, confidence_value: float,
            must_remove_outliers: bool, show_violin: bool, **kwargs) -> Tuple[Dict[str, Any], Plotter, pd.DataFrame]:
    """Main DynaFit function"""
    # Extract values to be emitted by thread from kwargs
    progress_callback = kwargs.get('progress_callback')
    warning_callback = kwargs.get('warning_callback')

    # Validate input data and return it
    df = ExcelValidator(workbook=workbook, sheetname=sheetname, cs_start_cell=cs_start_cell, cs_end_cell=cs_end_cell,
                        gr_start_cell=gr_start_cell, gr_end_cell=gr_end_cell).get_data()

    # Preprocess data
    df = preprocess_data(df=df, must_calculate_growth_rate=must_calculate_growth_rate, time_delta=time_delta,
                         must_remove_outliers=must_remove_outliers)

    # Bin samples into groups
    df = add_bins(df=df, individual_colonies=individual_colonies, bins=large_colony_groups)

    # Check for sample size warning
    warning_info = sample_size_warning_info(df=df, warning_level=WARNING_LEVEL)
    if sample_size_warning_answer(warning_info=warning_info, callback=warning_callback) is True:
        raise AbortedByUser("User decided to stop DynaFit analysis.")

    # Get histogram values
    hist_x = df['CS']
    hist_breakpoints, hist_instances = get_histogram_values(df=df)

    # Perform DynaFit bootstrap
    df = bootstrap_data(df=df, repeats=bootstrap_repeats, progress_callback=progress_callback)
    df = add_log_columns(df=df)

    # Get mean line values
    xs, ys = get_mean_line_arrays(df=df)

    # Get scatter values
    scatter_xs = df['log2_CS_mean'].values
    scatter_ys = df['log2_GR_var'].values
    scatter_colors = df['bins'].apply(lambda curr_bin: 'red' if curr_bin <= individual_colonies else 'gray').values

    # Get violin values (if user wants to do so)
    violin_ys, violin_colors = None, None
    if show_violin:
        violin_ys = [df.loc[df['bins'] == b]['log2_GR_var'].values for b in sorted(df['bins'].unique())]
        violin_colors = ['red' if i < individual_colonies else 'gray' for i, _ in enumerate(xs)]

    # Get CoDy values for CoDy plot
    cumcody_ys = get_cumulative_cody_values(xs=xs, ys=ys)
    endcody_ys = get_endpoint_cody_values(xs=xs, ys=ys)

    # Get CI values (if user wants to do so)
    upper_ys, lower_ys = None, None
    cumcody_upper_ys, cumcody_lower_ys = None, None
    endcody_upper_ys, endcody_lower_ys = None, None
    if show_ci:
        upper_ys, lower_ys = get_mean_line_confidence_interval(df=df, confidence_value=confidence_value)
        cumcody_upper_ys, cumcody_lower_ys = get_cumcody_confidence_interval(xs=xs, upper_ys=upper_ys,
                                                                             lower_ys=lower_ys)
        endcody_upper_ys, endcody_lower_ys = get_endcody_confidence_interval(xs=xs, upper_ys=upper_ys,
                                                                             lower_ys=lower_ys)

    # Store parameters used for DynaFit analysis
    original_parameters = {
        'filename': filename,
        'sheetname': sheetname,
        'max individual colony size': individual_colonies,
        'number of large colony groups': large_colony_groups,
        'bootstrapping repeats': bootstrap_repeats
    }
    dataframe_results = results_to_dataframe(original_parameters=original_parameters, xs=xs, ys=ys,
                                             cumcody_ys=cumcody_ys, endcody_ys=endcody_ys)
    # Encapsulates data for plots
    plot_results = Plotter(xs=xs, ys=ys, scatter_xs=scatter_xs, scatter_ys=scatter_ys, scatter_colors=scatter_colors,
                           show_violin=show_violin, violin_ys=violin_ys, violin_colors=violin_colors,
                           cumcody_ys=cumcody_ys, endcody_ys=endcody_ys, show_ci=show_ci, upper_ys=upper_ys,
                           lower_ys=lower_ys, cumcody_upper_ys=cumcody_upper_ys, cumcody_lower_ys=cumcody_lower_ys,
                           endcody_lower_ys=endcody_lower_ys, endcody_upper_ys=endcody_upper_ys,  hist_x=hist_x,
                           hist_breakpoints=hist_breakpoints, hist_instances=hist_instances)

    # Return results
    return original_parameters, plot_results, dataframe_results


def preprocess_data(df: pd.DataFrame, must_calculate_growth_rate: bool, time_delta: float,
                    must_remove_outliers: bool) -> pd.DataFrame:
    """Calls downstream methods related to data preprocessing, based on boolean flags."""
    if must_calculate_growth_rate:
        df = calculate_growth_rate(df=df, time_delta=time_delta)
    df = filter_colony_sizes_less_than_one(df=df)
    if must_remove_outliers:
        df = filter_outliers(df=df)
    return df


def calculate_growth_rate(df: pd.DataFrame, time_delta: float) -> pd.DataFrame:
    """Calculates GR values from CS1 and CS2 and a time delta. These values are place on the "GR" column (which
    initially contained the CS2 values)."""
    growth_rate = (np.log2(df['GR']) - np.log2(df['CS'])) / (time_delta / 24)
    if growth_rate.isna().any():  # happens if the log of CS1 or CS2 is negative - which doesn't make sense anyway
        raise ValueError('Growth rate could not be calculated from the given colony size ranges. '
                         'Did you mean to select Colony Size and Growth Rate instead? '
                         'Please check your selected column ranges.')
    return df.assign(GR=growth_rate)


def filter_colony_sizes_less_than_one(df: pd.DataFrame) -> pd.DataFrame:
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
        mes = (f'Could not divide the population of large colonies into {bins} unique groups. ' 
               'Please reduce the number of large colony groups and try again.')

        raise TooManyGroupsError(mes)
    return df.assign(bins=pd.concat([single_bins, multiple_bins]))


def sample_size_warning_info(df: pd.DataFrame, warning_level: int) -> Optional[Dict[int, int]]:
    """Checks whether any group resulting from the binning process has a low number of instances (based on the
    global value os WARNING_LEVEL."""
    warning_info = {}
    for bin_number, bin_values in df.groupby('bins'):
        n = len(bin_values)
        if n < warning_level:
            warning_info[int(bin_number)] = n
    return warning_info if warning_info else None


def sample_size_warning_answer(warning_info: Optional[Dict[int, int]], callback: Signal) -> bool:
    """Emits a GUI-blocking warning back to the main thread as a Queue. The GUI is meant to wrap the warning in
    a QMessageBox and put a boolean value in the Queue, in order to pass in the user's answer."""
    if warning_info is None:
        return False
    answer = Queue()
    warning = answer, warning_info
    callback.emit(warning)
    return answer.get(block=True)


def bootstrap_data(df: pd.DataFrame, repeats: int, progress_callback: Signal) -> pd.DataFrame:
    """Performs bootstrapping. Each bin is sampled N times (N="repeats" parameter)."""
    total_progress = repeats * len(df['bins'].unique())
    progress_counter = count(start=1)
    columns = ['CS_mean', 'GR_var', 'bins']
    output_df = pd.DataFrame(columns=columns)
    for bin_number, bin_values in df.groupby('bins'):
        sample_size = len(bin_values)
        for repeat in range(repeats):
            emit_bootstrap_progress(current=next(progress_counter), total=total_progress, callback=progress_callback)
            sample = bin_values.sample(n=sample_size, replace=True)
            row = pd.Series([sample['CS'].mean(), sample['GR'].var(), bin_number], index=columns)
            output_df = output_df.append(row, ignore_index=True)
    return output_df


def emit_bootstrap_progress(current: int, total: int, callback: Signal):
    """Emits a integer back to the GUI thread, in order to update the progress bar."""
    progress = int(round(100 * current / total))
    callback.emit(progress)


def add_log_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Adds columns with the log2-transformed values of colony size means and growth rate variances."""
    return df.assign(log2_CS_mean=np.log2(df['CS_mean']), log2_GR_var=np.log2(df['GR_var']))


def get_mean_line_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Returns a tuple of numpy arrays representing the X and Y values of the mean line in the CVP, respectively."""
    xs = df.groupby('bins').mean()['log2_CS_mean'].values
    ys = df.groupby('bins').mean()['log2_GR_var'].values
    return xs, ys


def get_mean_line_confidence_interval(df: pd.DataFrame, confidence_value: float) -> Tuple[np.ndarray, np.ndarray]:
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


def get_histogram_values(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Returns values for the histogram of the colony size, indicating the group sizes made by the binning process."""
    grouped_data = df.groupby('bins')
    bin_breakpoints = grouped_data.max()['CS']
    bin_instances = grouped_data.count()['CS'].values
    return bin_breakpoints, bin_instances


def calculate_cumcody(xs: np.ndarray, ys: np.ndarray, cody_n: Optional[int]) -> float:
    """Returns the area above the curve (mean green line) up to the X coordinate equivalent to the "cody_n" parameter.
    If that parameter is a None value, uses the entire range of X values instead."""
    if cody_n is not None:  # Truncate arrays so that a specific CoDy value is calculated
        xs, ys = truncate_arrays(xs=xs, ys=ys, cutoff=cody_n)
    triangle_area = calculate_triangle_area(xs=xs, ys=ys)
    area_above_curve = trapezium_integration(xs=xs, ys=ys)
    return area_above_curve / triangle_area if triangle_area != 0 else 0.0


def truncate_arrays(xs: np.ndarray, ys: np.ndarray, cutoff: int) -> Tuple[np.ndarray, np.ndarray]:
    """Truncates and returns the arrays "xs" and "ys" by removing values from the "xs" arrays lower than the cutoff,
    and the removing "ys"'s last elements until both arrays have the same size."""
    xs_trunc = np.array([x for x in xs if x < cutoff] + [cutoff])
    end_y = np.interp(cutoff, xs, ys)
    ys_trunc = np.array(ys[:len(xs_trunc)] + [end_y])
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
    integrated_area = 0.0
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


def calculate_endcody(xs: np.ndarray, ys: np.ndarray, cody_n: Optional[int]) -> float:
    """Returns the normed triangular area defined by (xs[0], ys[0]), (xs[-1], ys[0]) and (xs[-1], ys[-1])"""
    if cody_n is not None:  # Truncate arrays so that a specific CoDy value is calculated
        xs, ys = truncate_arrays(xs=xs, ys=ys, cutoff=cody_n)
    triangle_area = calculate_triangle_area(xs=xs, ys=ys)
    cody_triangle_area = calculate_cody_triangle_area(xs=xs, ys=ys)
    return cody_triangle_area / triangle_area if triangle_area != 0 else 0.0


def calculate_cody_triangle_area(xs: np.ndarray, ys: np.ndarray) -> float:
    """Calculates the CoDy triangle area."""
    width = xs[-1] - xs[0]
    height = ys[0] - ys[-1]  # SIGNED area
    return (width * height) / 2


def get_cumulative_cody_values(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Calculates cumulative CoDy values for all points in the mean line arrays."""
    cody_ys = np.array([calculate_cumcody(xs=xs, ys=ys, cody_n=x) for x in xs])
    return cody_ys


def get_cumcody_confidence_interval(xs: np.ndarray, upper_ys: np.ndarray,
                                    lower_ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the confidence interval values for the cumulative hypothesis plot."""
    cody_upper_ys = np.array([calculate_cumcody(xs=xs, ys=upper_ys, cody_n=x) for x in xs])
    cody_lower_ys = np.array([calculate_cumcody(xs=xs, ys=lower_ys, cody_n=x) for x in xs])
    return cody_upper_ys, cody_lower_ys


def get_endpoint_cody_values(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Calculates endpoint CoDy values for all points in the mean line arrays."""
    cody_ys = np.array([calculate_endcody(xs=xs, ys=ys, cody_n=x) for x in xs])
    return cody_ys


def get_endcody_confidence_interval(xs: np.ndarray, upper_ys: np.ndarray,
                                    lower_ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the confidence interval values for the endpoint hypothesis plot."""
    cody_upper_ys = np.array([calculate_endcody(xs=xs, ys=upper_ys, cody_n=x) for x in xs])
    cody_lower_ys = np.array([calculate_endcody(xs=xs, ys=lower_ys, cody_n=x) for x in xs])
    return cody_upper_ys, cody_lower_ys


def results_to_dataframe(original_parameters: Dict[str, Any], xs: np.ndarray, ys: np.ndarray, cumcody_ys: np.ndarray,
                         endcody_ys: np.ndarray) -> pd.DataFrame:
    """Saves DynaFit dataframe_results as a pandas DataFrame (used for Excel/csv export)."""
    largest_seq_size = max(len(original_parameters), len(xs))
    params_padding = largest_seq_size - len(original_parameters)
    array_padding = largest_seq_size - len(xs)
    data = {
        'Parameter': np.concatenate([list(original_parameters.keys()), np.full(params_padding, np.nan)]),
        'Value': np.concatenate([list(original_parameters.values()), np.full(params_padding, np.nan)]),
        'Log2(Colony Size)': np.concatenate([xs, np.full(array_padding, np.nan)]).round(2),
        'Log2(Variance)': np.concatenate([ys, np.full(array_padding, np.nan)]).round(2),
        'Closeness to H0 (cumulative)': np.concatenate([np.abs(cumcody_ys), np.full(array_padding, np.nan)]).round(2),
        'Closeness to H1 (cumulative)': np.concatenate([np.abs(cumcody_ys-1), np.full(array_padding, np.nan)]).round(2),
        'Closeness to H0 (endpoint)': np.concatenate([np.abs(endcody_ys), np.full(array_padding, np.nan)]).round(2),
        'Closeness to H1 (endpoint)': np.concatenate([np.abs(endcody_ys-1), np.full(array_padding, np.nan)]).round(2),
    }
    return pd.DataFrame(data)
