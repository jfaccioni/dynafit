"""core.py - bundles all computations of DynaFit."""

from itertools import count
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PySide2.QtCore import Signal
from openpyxl import Workbook

from src.exceptions import AbortedByUser, TooManyGroupsError
from src.plotter import Plotter
from src.utils import get_missing_coordinate
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
    hist_x, hist_breakpoints, hist_instances = get_histogram_values(df=df)

    # Get original sample parameters
    xs = np.log2(df.groupby('bins').mean()['CS']).values
    ys = np.log2(df.groupby('bins').var()['GR']).values
    original_var = df.groupby('bins').var()['GR'].values
    original_var_se = np.array([group.var()['GR'] * np.sqrt(2 / (len(group) - 1)) for _, group in df.groupby('bins')])

    # Perform DynaFit bootstrap
    df = bootstrap_data(df=df, repeats=bootstrap_repeats, progress_callback=progress_callback, show_ci=show_ci)
    df = add_log_columns(df=df)

    # Get mean line values
    boot_xs, boot_ys = get_bootstrap_xy_values(df=df)

    # Get scatter values
    scatter_xs, scatter_ys, scatter_colors = get_scatter_values(df=df, individual_colonies=individual_colonies)

    # Get violin values (if user wants to do so)
    violin_ys, violin_colors, violin_data = None, None, None
    if show_violin:
        violin_ys, violin_colors = get_violin_values(df=df, individual_colonies=individual_colonies)

    # Get hypothesis values for hypothesis plot
    cumulative_hyp_ys = get_cumulative_hypothesis_values(xs=xs, ys=ys)
    endpoint_hyp_ys = get_endpoint_hypothesis_values(xs=xs, ys=ys)

    # Get CI values (if user wants to do so)
    upper_ys, lower_ys = None, None
    cumulative_hyp_upper_ys, cumulative_hyp_lower_ys = None, None
    endpoint_hyp_upper_ys, endpoint_hyp_lower_ys = None, None
    if show_ci:
        upper_ys, lower_ys = get_mean_line_ci(df=df, confidence_value=confidence_value,
                                              original_variance=original_var, original_variance_se=original_var_se)
        cumulative_hyp_upper_ys = get_cumulative_hypothesis_values(xs=xs, ys=upper_ys)
        cumulative_hyp_lower_ys = get_cumulative_hypothesis_values(xs=xs, ys=lower_ys)
        endpoint_hyp_upper_ys = get_endpoint_hypothesis_values(xs=xs, ys=upper_ys)
        endpoint_hyp_lower_ys = get_endpoint_hypothesis_values(xs=xs, ys=lower_ys)

    # Store parameters used for DynaFit analysis
    original_parameters = {
        'filename': filename,
        'sheetname': sheetname,
        'max individual colony size': individual_colonies,
        'number of large colony groups': large_colony_groups,
        'bootstrapping repeats': bootstrap_repeats
    }
    dataframe_results = results_to_dataframe(original_parameters=original_parameters, xs=xs, ys=ys,
                                             cumulative_hyp_ys=cumulative_hyp_ys, endpoint_hyp_ys=endpoint_hyp_ys,
                                             show_ci=show_ci, upper_ys=upper_ys, lower_ys=lower_ys,
                                             cumulative_hyp_upper_ys=cumulative_hyp_upper_ys,
                                             cumulative_hyp_lower_ys=cumulative_hyp_lower_ys,
                                             endpoint_hyp_upper_ys=endpoint_hyp_upper_ys,
                                             endpoint_hyp_lower_ys=endpoint_hyp_lower_ys)
    plot_results = Plotter(xs=xs, ys=ys, boot_xs=boot_xs, boot_ys=boot_ys, scatter_xs=scatter_xs, scatter_ys=scatter_ys,
                           show_violin=show_violin, violin_ys=violin_ys, cumulative_hyp_ys=cumulative_hyp_ys,
                           endpoint_hyp_ys=endpoint_hyp_ys, show_ci=show_ci, upper_ys=upper_ys, lower_ys=lower_ys,
                           cumulative_hyp_upper_ys=cumulative_hyp_upper_ys,
                           cumulative_hyp_lower_ys=cumulative_hyp_lower_ys, endpoint_hyp_upper_ys=endpoint_hyp_upper_ys,
                           endpoint_hyp_lower_ys=endpoint_hyp_lower_ys, hist_x=hist_x,
                           hist_breakpoints=hist_breakpoints, hist_instances=hist_instances)

    # Return results
    return original_parameters, plot_results, dataframe_results


def preprocess_data(df: pd.DataFrame, must_calculate_growth_rate: bool, time_delta: float,
                    must_remove_outliers: bool) -> pd.DataFrame:
    """Calls downstream methods related to data preprocessing, based on boolean flags."""
    df = filter_colony_sizes_less_than_one(df=df)
    if must_calculate_growth_rate:
        df = calculate_growth_rate(df=df, time_delta=time_delta)
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


def sample_size_warning_info(df: pd.DataFrame, warning_level: int) -> Optional[Dict[int, Tuple[int, float]]]:
    """Checks whether any group resulting from the binning process has a low number of instances (based on the
    global value os WARNING_LEVEL."""
    warning_info = {}
    for bin_number, bin_values in df.groupby('bins'):
        n = len(bin_values)
        if n < warning_level:
            warning_info[int(bin_number)] = (n, bin_values['CS'].mean())
    return warning_info if warning_info else None


def sample_size_warning_answer(warning_info: Optional[Dict[int, int]], callback: Signal) -> bool:
    """Emits a GUI-blocking warning back to the main thread as a Queue. The GUI is meant to wrap the warning in
    a QMessageBox and put a boolean value in the Queue, in order to pass in the user's answer."""
    if warning_info is None:
        return False
    answer = Queue()
    warning = answer, warning_info
    callback.emit(warning)  # noqa
    return answer.get(block=True)


def get_histogram_values(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns values for the histogram of the colony size, indicating the group sizes made by the binning process."""
    hist_xs = df['CS']
    grouped_data = df.groupby('bins')
    hist_breakpoints = grouped_data.max()['CS'].values
    hist_instances = grouped_data.count()['CS'].values
    return hist_xs, hist_breakpoints, hist_instances


def bootstrap_data(df: pd.DataFrame, repeats: int, progress_callback: Signal, show_ci: bool) -> pd.DataFrame:
    """Performs bootstrapping. Each bin is sampled N times (N="repeats" parameter)."""
    counter = count(start=0)
    total_progress = repeats * len(df['bins'].unique())
    output_df = pd.DataFrame(columns=['CS_mean', 'GR_var', 'bins', 't_star'], index=range(total_progress), dtype=float)
    for group, group_values in df.groupby('bins'):
        sample_size = len(group_values)
        for repeat in range(repeats):
            i = next(counter)
            output_df.loc[i] = get_bootstrap_params(data=group_values, n=sample_size, group=group, show_ci=show_ci)
            emit_bootstrap_progress(current=i+1, total=total_progress, callback=progress_callback)
    return output_df.replace([np.inf, -np.inf], np.nan).dropna()


def get_bootstrap_params(data: pd.DataFrame, n: int, group: float, show_ci: bool) -> Tuple:
    """Gets bootstrap parameters: mean colony size of bootstrap sample, variance in growth rate of bootstrap sample,
    group number of bootstrap sample, corresponding t statistic of bootstrap sample (if needed for CI)."""
    bootstrap_sample = data.sample(n=n, replace=True)
    mean_colony_size = bootstrap_sample['CS'].mean()
    growth_rate_variance = bootstrap_sample['GR'].var()
    data_var = data['GR'].var()
    t = np.nan if not show_ci else get_t_star(bootstrap_sample=bootstrap_sample, data_var=data_var)
    return mean_colony_size, growth_rate_variance, group, t


def get_t_star(bootstrap_sample: pd.DataFrame, data_var: float) -> float:
    """Calculates the t statistic (t-star) for the bootstrap distribution, relative to the total distribution.
    Source: https://arxiv.org/pdf/1411.5279.pdf"""
    bootstrap_var = bootstrap_sample['GR'].var()
    # The line below calculates variance SE, source: https://web.eecs.umich.edu/~fessler/papers/files/tr/stderr.pdf
    bootstrap_var_se = bootstrap_var * np.sqrt(2 / (len(bootstrap_sample) - 1))
    return (bootstrap_var - data_var) / bootstrap_var_se


def emit_bootstrap_progress(current: int, total: int, callback: Signal) -> None:
    """Emits a integer back to the GUI thread, in order to update the progress bar."""
    progress = int(round(100 * current / total))
    callback.emit(progress)  # noqa


def add_log_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Adds columns with the log2-transformed values of colony size means and growth rate variances."""
    return df.assign(log2_CS_mean=np.log2(df['CS_mean']), log2_GR_var=np.log2(df['GR_var']))


def get_bootstrap_xy_values(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Returns a tuple of numpy arrays representing the X and Y values of the mean line in the CVP, respectively."""
    xs = df.groupby('bins').mean()['log2_CS_mean'].values
    ys = df.groupby('bins').mean()['log2_GR_var'].values
    return xs, ys


def get_scatter_values(df: pd.DataFrame, individual_colonies: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns a tuple of numpy arrays representing the X, Y and color values of the bootstrap scatter."""
    scatter_xs = df['log2_CS_mean'].values
    scatter_ys = df['log2_GR_var'].values
    scatter_colors = df['bins'].apply(get_element_color, cutoff=individual_colonies).values
    return scatter_xs, scatter_ys, scatter_colors


def get_violin_values(df: pd.DataFrame, individual_colonies: int) -> Tuple[List[np.ndarray], np.ndarray]:
    """Returns a List of numpy arrays representing the values that serve as the base for a violin, and a secondary
    numpy array of violin colors."""
    unique_bins = df['bins'].unique()
    violin_ys = [df.loc[df['bins'] == b]['log2_GR_var'].values for b in unique_bins]
    violin_colors = np.array([get_element_color(b, cutoff=individual_colonies) for b in unique_bins])
    return violin_ys, violin_colors


def get_element_color(element: float, cutoff: float) -> str:
    """Returns whether a given plot element should be red or gray, based on it being above or below a cutoff."""
    return 'red' if element <= cutoff else 'gray'


def get_cumulative_hypothesis_values(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Calculates cumulative hypothesis values for all points in the mean line arrays."""
    ys_h1 = np.array([get_missing_coordinate(x1=xs[0], y1=ys[0], x2=x, angular_coefficient=-1.0) for x in xs])
    area_array = np.array([np.trapz(y=ys[:i] - ys[0], x=xs[:i]) for i, _ in enumerate(xs, 1)])
    triangle_array = (xs - xs[0]) * (ys_h1 - ys[0]) * 0.5
    return np.nan_to_num((area_array / triangle_array), posinf=0.0, neginf=0.0)


def get_endpoint_hypothesis_values(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Calculates endpoint hypothesis values for all points in the mean line arrays."""
    ys_h0 = np.array([ys[0] for _ in ys])
    ys_h1 = np.array([get_missing_coordinate(x1=xs[0], y1=ys[0], x2=x, angular_coefficient=-1.0) for x in xs])
    return np.nan_to_num((ys_h0 - ys) / (ys_h0 - ys_h1), posinf=0.0, neginf=0.0)


def get_mean_line_ci(df: pd.DataFrame, confidence_value: float, original_variance: np.ndarray,
                     original_variance_se: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates and returns the a tuple of arrays representing the Y values of the upper and lower confidence
    interval for the bootstrapped population."""
    upper_ys = []
    lower_ys = []
    alpha = 1 - confidence_value
    for (_, bin_values), var, var_se in zip(df.groupby('bins'), original_variance, original_variance_se):
        t_distribution = bin_values['t_star']
        upper, lower = calculate_bootstrap_ci_from_t_distribution(t_distribution=t_distribution, sample_stat=var,
                                                                  sample_stat_se=var_se, alpha=alpha)
        upper_ys.append(upper)
        lower_ys.append(lower)
    return np.array(upper_ys), np.array(lower_ys)


def calculate_bootstrap_ci_from_t_distribution(t_distribution: pd.Series, sample_stat: float, sample_stat_se: float,
                                               alpha: float) -> Tuple[float, float]:
    """Returns the CI upper and lower bounds from a series of data, using a t distribution."""
    upper = sample_stat - (t_distribution.quantile(alpha/2) * sample_stat_se)
    lower = sample_stat - (t_distribution.quantile(1 - alpha/2) * sample_stat_se)
    return np.log2(upper), np.log2(lower)


def results_to_dataframe(original_parameters: Dict[str, Any], xs: np.ndarray, ys: np.ndarray,
                         cumulative_hyp_ys: np.ndarray, endpoint_hyp_ys: np.ndarray, show_ci: bool,
                         upper_ys: Optional[np.ndarray], lower_ys: Optional[np.ndarray],
                         cumulative_hyp_upper_ys: Optional[np.ndarray], cumulative_hyp_lower_ys: Optional[np.ndarray],
                         endpoint_hyp_upper_ys: Optional[np.ndarray],
                         endpoint_hyp_lower_ys: Optional[np.ndarray]) -> pd.DataFrame:
    """Saves DynaFit dataframe_results as a pandas DataFrame (used for Excel/csv export)."""
    largest_seq_size = max(len(original_parameters), len(xs))
    params_padding = largest_seq_size - len(original_parameters)
    array_padding = largest_seq_size - len(xs)
    data = {
        'Parameter': np.concatenate([list(original_parameters.keys()), np.full(params_padding, np.nan)]),
        'Value': np.concatenate([list(original_parameters.values()), np.full(params_padding, np.nan)]),
        'Log2(Colony Size)': np.concatenate([xs, np.full(array_padding, np.nan)]).round(2),
        'Log2(Variance)': np.concatenate([ys, np.full(array_padding, np.nan)]).round(2),
        'Log2(Variance) upper CI': np.concatenate([upper_ys, np.full(array_padding, np.nan)]).round(2)
        if show_ci else None,
        'Log2(Variance) lower CI': np.concatenate([lower_ys, np.full(array_padding, np.nan)]).round(2)
        if show_ci else None,
        'Distance to H0 (cumulative)': np.concatenate([np.abs(cumulative_hyp_ys),
                                                       np.full(array_padding, np.nan)]).round(2),
        'Distance to H0 (cumulative) upper CI': np.concatenate([np.abs(cumulative_hyp_upper_ys),
                                                                np.full(array_padding, np.nan)]).round(2)
        if show_ci else None,
        'Distance to H0 (cumulative) lower CI': np.concatenate([np.abs(cumulative_hyp_lower_ys),
                                                                np.full(array_padding, np.nan)]).round(2)
        if show_ci else None,
        'Distance to H1 (cumulative)': np.concatenate([np.abs(cumulative_hyp_ys - 1),
                                                       np.full(array_padding, np.nan)]).round(2),
        'Distance to H1 (cumulative) upper CI': np.concatenate([np.abs(cumulative_hyp_upper_ys - 1),
                                                                np.full(array_padding, np.nan)]).round(2)
        if show_ci else None,
        'Distance to H1 (cumulative) lower CI': np.concatenate([np.abs(cumulative_hyp_lower_ys - 1),
                                                                np.full(array_padding, np.nan)]).round(2)
        if show_ci else None,
        'Distance to H0 (endpoint)': np.concatenate([np.abs(endpoint_hyp_ys),
                                                     np.full(array_padding, np.nan)]).round(2),
        'Distance to H0 (endpoint) upper CI': np.concatenate([np.abs(endpoint_hyp_upper_ys),
                                                              np.full(array_padding, np.nan)]).round(2)
        if show_ci else None,
        'Distance to H0 (endpoint) lower CI': np.concatenate([np.abs(endpoint_hyp_lower_ys),
                                                              np.full(array_padding, np.nan)]).round(2)
        if show_ci else None,
        'Distance to H1 (endpoint)': np.concatenate([np.abs(endpoint_hyp_ys - 1),
                                                     np.full(array_padding, np.nan)]).round(2),
        'Distance to H1 (endpoint) upper CI': np.concatenate([np.abs(endpoint_hyp_upper_ys - 1),
                                                              np.full(array_padding, np.nan)]).round(2)
        if show_ci else None,
        'Distance to H1 (endpoint) lower CI': np.concatenate([np.abs(endpoint_hyp_lower_ys - 1),
                                                              np.full(array_padding, np.nan)]).round(2)
        if show_ci else None,
    }
    filtered_data = {k: v for k, v in data.items() if v is not None}
    return pd.DataFrame(filtered_data)
