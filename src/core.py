"""core.py - bundles all computations of DynaFit.
IMPORTANT: all variance calculations are performed with pd.Series.var(), which uses ddof = 1 by default.
Numpy uses ddof = 0 by default, so take care when translating results from one package to the other."""

from itertools import count
from queue import Queue
from typing import Any, Dict, KeysView, List, Optional, Tuple, Union, ValuesView

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


def dynafit(workbook: Workbook, filename: str, sheetname: str, calculate_growth_rate: bool, time_delta: float,
            cs_start_cell: str, cs_end_cell: str, gr_start_cell: str, gr_end_cell: str, individual_colonies: int,
            large_groups: int, bootstrap_repeats: int, show_ci: bool, confidence_value: float, remove_outliers: bool,
            show_violin: bool, **kwargs) -> Tuple[Dict[str, Any], Plotter, pd.DataFrame]:
    """Main DynaFit function"""
    # Extract values to be emitted by thread from kwargs
    progress_callback = kwargs.get('progress_callback')
    warning_callback = kwargs.get('warning_callback')

    # Validate input data and return it
    df = ExcelValidator(workbook=workbook, sheetname=sheetname, cs_start_cell=cs_start_cell, cs_end_cell=cs_end_cell,
                        gr_start_cell=gr_start_cell, gr_end_cell=gr_end_cell).get_data()

    # Preprocess input data
    df = preprocess_data(df=df, calculate_growth_rate=calculate_growth_rate, time_delta=time_delta,
                         remove_outliers=remove_outliers)

    # Bin samples into groups
    df = divide_sample_into_groups(df=df, individual_colonies=individual_colonies, large_groups=large_groups)

    # Check for sample size warning
    warning_info = sample_size_warning_info(df=df, warning_level=WARNING_LEVEL)
    if sample_size_warning_answer(warning_info=warning_info, callback=warning_callback) is True:
        raise AbortedByUser("User decided to stop DynaFit analysis.")

    # Get original sample parameters
    xs, ys = get_original_sample_parameters(df=df)
    original_var, original_var_se = get_original_sample_statistics(df=df)

    # Get histogram values
    hist_xs, hist_intervals = get_histogram_values(df=df)

    # Get GR values
    groups, growth_rates = get_growth_rates(df=df)
    growth_rate_means, growth_rate_vars = get_growth_rate_statistics(df=df)
    global_growth_rate_mean, global_growth_rate_var = get_global_growth_rate_statistics(df=df)

    # Perform DynaFit bootstrap
    df = bootstrap_data(df=df, repeats=bootstrap_repeats, progress_callback=progress_callback, show_ci=show_ci)
    
    # Postprocess bootstrap data
    df = postprocess_data(df=df)
    
    # Get bootstrap scatter values
    scatter_xs, scatter_ys = get_bootstrap_scatter_values(df=df)

    # Get bootstrap violin values (if user wants to do so)
    violin_ys, violin_xs = None, None
    violin_q1, violin_medians, violin_q3 = None, None, None
    if show_violin:
        violin_xs, violin_ys = get_bootstrap_violin_values(df=df)
        violin_q1, violin_medians, violin_q3 = get_bootstrap_violin_statistics(violins=violin_ys)

    # Get hypothesis values
    cumulative_ys = get_cumulative_hypothesis_values(xs=xs, ys=ys)
    endpoint_ys = get_endpoint_hypothesis_values(xs=xs, ys=ys)

    # Get CI values for bootstrap scatter (if user wants to do so)
    upper_ys, lower_ys = None, None
    if show_ci:
        upper_ys, lower_ys = get_bootstrap_ci(df=df, confidence_value=confidence_value, original_var=original_var,
                                              original_var_se=original_var_se)
    # Get CI values for hypothesis
    cumulative_upper_ys, cumulative_lower_ys = None, None
    endpoint_upper_ys, endpoint_lower_ys = None, None
    if show_ci:
        cumulative_upper_ys = get_cumulative_hypothesis_values(xs=xs, ys=upper_ys)
        cumulative_lower_ys = get_cumulative_hypothesis_values(xs=xs, ys=lower_ys)
        endpoint_upper_ys = get_endpoint_hypothesis_values(xs=xs, ys=upper_ys)
        endpoint_lower_ys = get_endpoint_hypothesis_values(xs=xs, ys=lower_ys)

    # Store parameters used
    original_parameters = {
        'filename': filename,
        'sheetname': sheetname,
        'max individual colony size': individual_colonies,
        'number of large colony groups': large_groups,
        'bootstrapping repeats': bootstrap_repeats
    }
    
    # Create Plotter instance
    plot_results = Plotter(xs=xs, ys=ys, scatter_xs=scatter_xs, scatter_ys=scatter_ys, show_violin=show_violin,
                           violin_xs=violin_xs, violin_ys=violin_ys, violin_q1=violin_q1, violin_medians=violin_medians,
                           violin_q3=violin_q3, cumulative_ys=cumulative_ys, endpoint_ys=endpoint_ys,
                           show_ci=show_ci, upper_ys=upper_ys, lower_ys=lower_ys,
                           cumulative_upper_ys=cumulative_upper_ys, cumulative_lower_ys=cumulative_lower_ys,
                           endpoint_upper_ys=endpoint_upper_ys, endpoint_lower_ys=endpoint_lower_ys,
                           hist_xs=hist_xs, hist_intervals=hist_intervals, groups=groups, growth_rates=growth_rates,
                           growth_rate_means=growth_rate_means, growth_rate_vars=growth_rate_vars,
                           global_growth_rate_mean=global_growth_rate_mean,
                           global_growth_rate_var=global_growth_rate_var)
    
    # Store results as a pandas DataFrame
    dataframe_results = results_to_dataframe(original_parameters=original_parameters, xs=xs, ys=ys,
                                             cumulative_ys=cumulative_ys, endpoint_ys=endpoint_ys, show_ci=show_ci,
                                             upper_ys=upper_ys, lower_ys=lower_ys,
                                             cumulative_upper_ys=cumulative_upper_ys,
                                             cumulative_lower_ys=cumulative_lower_ys,
                                             endpoint_upper_ys=endpoint_upper_ys,
                                             endpoint_lower_ys=endpoint_lower_ys)
    
    return original_parameters, plot_results, dataframe_results


def preprocess_data(df: pd.DataFrame, calculate_growth_rate: bool, time_delta: float,
                    remove_outliers: bool) -> pd.DataFrame:
    """Calls downstream methods related to data preprocessing, based on boolean flags."""
    df = filter_colony_sizes_less_than_one(df=df)
    if calculate_growth_rate:
        df = add_growth_rate_column(df=df, time_delta=time_delta)
    if remove_outliers:
        df = filter_outliers(df=df)
    return df


def add_growth_rate_column(df: pd.DataFrame, time_delta: float) -> pd.DataFrame:
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


def divide_sample_into_groups(df: pd.DataFrame, individual_colonies: int, large_groups: int) -> pd.DataFrame:
    """Returns the data DataFrame with a "group" column, which divides the population into groups of cells.
    Colonies with size <= the "individual_colonies" parameters are grouped with colonies with the same number of cells,
    while larger colonies are split into N groups with a close number of instances in each one of them."""
    group_condition = df['CS'] > individual_colonies
    single_groups = df.loc[~group_condition]['CS']
    try:
        multiple_groups = pd.qcut(df.loc[group_condition]['CS'], large_groups, labels=False) + (individual_colonies + 1)
    except ValueError:
        mes = (f'Could not divide the population of large colonies into {large_groups} unique groups. ' 
               'Please reduce the number of large colony groups and try again.')
        raise TooManyGroupsError(mes)
    return df.assign(groups=pd.concat([single_groups, multiple_groups]))


def sample_size_warning_info(df: pd.DataFrame, warning_level: int) -> Optional[Dict[int, Tuple[int, float]]]:
    """Checks whether any group resulting from the binning process has a low number of instances (based on the
    global value os WARNING_LEVEL."""
    warning_info = {}
    for bin_number, bin_values in df.groupby('groups'):
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


def get_original_sample_parameters(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Returns a tuple of arrays representing the X and Y coordinates of the sample's original CS and GR."""
    xs = np.log2(df.groupby('groups').mean()['CS']).values
    ys = np.log2(df.groupby('groups').var()['GR']).values
    return xs, ys


def get_original_sample_statistics(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Returns a tuple of arrays representing the variance and its standard error of the sample's original GR."""
    original_var = df.groupby('groups').var()['GR'].values
    degrees_of_freedom = df.groupby('groups').count()['GR'].values - 1
    original_var_se = original_var * np.sqrt(2 / degrees_of_freedom)
    return original_var, original_var_se


def get_histogram_values(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Returns values for the histogram of the colony size, indicating the group sizes made by the binning process."""
    hist_xs = df['CS']
    hist_intervals = df.groupby('groups').max()['CS'].values
    return hist_xs, hist_intervals


def get_growth_rates(df: pd.DataFrame) -> Tuple[np.array, List[np.array]]:
    """Returns a numpy array representing the X coordinates of each CS group, and a list of numpy arrays representing
    the growth rate values inside each group."""
    groups = [group.mean() for _, group in df.groupby('groups')['CS']]
    growth_rates = [group.values for _, group in df.groupby('groups')['GR']]
    return groups, growth_rates


def get_growth_rate_statistics(df: pd.DataFrame) -> Tuple[np.array, np.array]:
    """Returns mean/var statistics of the growth rate inside each group."""
    growth_rate_means = np.array([group.mean() for _, group in df.groupby('groups')['GR']])
    growth_rate_vars = np.array([group.var() for _, group in df.groupby('groups')['GR']])
    return growth_rate_means, growth_rate_vars


def get_global_growth_rate_statistics(df: pd.DataFrame) -> Tuple[np.array, np.array]:
    """Returns mean/var statistics of the growth rate for all groups put together."""
    growth_rate_mean = df['GR'].mean()
    growth_rate_var = df['GR'].var()
    return growth_rate_mean, growth_rate_var


def bootstrap_data(df: pd.DataFrame, repeats: int, progress_callback: Signal, show_ci: bool) -> pd.DataFrame:
    """Performs bootstrapping. Each bin is sampled N times (N="repeats" parameter)."""
    counter = count(start=0)
    total_progress = repeats * len(df['groups'].unique())
    output_df = pd.DataFrame(columns=['bootstrap_CS_mean', 'bootstrap_GR_var', 'groups', 't_stat'],
                             index=range(total_progress), dtype=float)  # noqa
    for group, group_values in df.groupby('groups'):
        sample_size = len(group_values)
        for repeat in range(repeats):
            i = next(counter)
            output_df.loc[i] = get_bootstrap_params(data=group_values, n=sample_size, group=group, show_ci=show_ci)
            emit_bootstrap_progress(current=i+1, total=total_progress, callback=progress_callback)
    return output_df


def get_bootstrap_params(data: pd.DataFrame, n: int, group: float, show_ci: bool) -> Tuple:
    """Gets bootstrap parameters: mean colony size of bootstrap sample, variance in growth rate of bootstrap sample,
    group number of bootstrap sample, corresponding t statistic of bootstrap sample (if needed for CI)."""
    bootstrap_sample = data.sample(n=n, replace=True)
    mean_colony_size = bootstrap_sample['CS'].mean()
    growth_rate_variance = bootstrap_sample['GR'].var()
    data_var = data['GR'].var()
    t = 'no ci' if not show_ci else get_t_stat(bootstrap_sample=bootstrap_sample, data_var=data_var)
    return mean_colony_size, growth_rate_variance, group, t


def get_t_stat(bootstrap_sample: pd.DataFrame, data_var: float) -> float:
    """Calculates the t statistic (t-star) for the bootstrap distribution, relative to the total distribution.
    Source: https://arxiv.org/pdf/1411.5279.pdf"""
    bootstrap_var = bootstrap_sample['GR'].var()
    # The line below calculates the standard error of the bootstrap sample's variance (not its mean!).
    # Source: https://web.eecs.umich.edu/~fessler/papers/files/tr/stderr.pdf
    bootstrap_var_se = bootstrap_var * np.sqrt(2 / (len(bootstrap_sample) - 1))
    return (bootstrap_var - data_var) / bootstrap_var_se


def emit_bootstrap_progress(current: int, total: int, callback: Signal) -> None:
    """Emits a integer back to the GUI thread, in order to update the progress bar."""
    progress = int(round(100 * current / total))
    callback.emit(progress)  # noqa


def postprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Calls routines related to postprocessing the data obtained after bootstrap."""
    df = drop_inf_nan_values(df=df)
    df = add_log2_columns(df=df, column_names=['bootstrap_CS_mean', 'bootstrap_GR_var'])
    return df


def drop_inf_nan_values(df: pd.DataFrame) -> pd.DataFrame:
    """Removes rows containing np.inf, -np.inf or np.nan from the input dataframe."""
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def add_log2_columns(df: pd.DataFrame, column_names: List[str]) -> pd.DataFrame:
    """Adds columns with the log2-transformed values the input column name list."""
    log_columns = {f'log2_{column_name}': np.log2(df[column_name]) for column_name in column_names}
    return df.assign(**log_columns)


def get_bootstrap_scatter_values(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Returns a tuple of numpy arrays representing the X, Y and color values of the bootstrap scatter."""
    scatter_xs = df['log2_bootstrap_CS_mean'].values
    scatter_ys = df['log2_bootstrap_GR_var'].values
    return scatter_xs, scatter_ys


def get_bootstrap_violin_values(df: pd.DataFrame) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Returns a List of numpy arrays representing the values that serve as the base for a violin."""
    violin_xs = df.groupby('groups').mean()['log2_bootstrap_CS_mean'].values
    violin_ys = [group.values for _, group in df.groupby('groups')['log2_bootstrap_GR_var']]
    return violin_xs, violin_ys


def get_bootstrap_violin_statistics(violins: List[np.array]) -> Tuple[np.array, np.array, np.array]:
    """Returns a three numpy arrays for the q1, median and q3 values of each violin, respectively."""
    violin_q1 = [np.percentile(violin, 25) for violin in violins]
    violin_medians = [np.percentile(violin, 50) for violin in violins]
    violin_q3 = [np.percentile(violin, 75) for violin in violins]
    return violin_q1, violin_medians, violin_q3


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


def get_bootstrap_ci(df: pd.DataFrame, confidence_value: float, original_var: np.ndarray,
                     original_var_se: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates and returns the a tuple of arrays representing the Y values of the upper and lower confidence
    interval for the bootstrapped population."""
    upper_ys = []
    lower_ys = []
    alpha = 1 - confidence_value
    for (_, bin_values), var, var_se in zip(df.groupby('groups'), original_var, original_var_se):
        t_distribution = bin_values['t_stat']
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
                         cumulative_ys: np.ndarray, endpoint_ys: np.ndarray, show_ci: bool,
                         upper_ys: Optional[np.ndarray], lower_ys: Optional[np.ndarray],
                         cumulative_upper_ys: Optional[np.ndarray], cumulative_lower_ys: Optional[np.ndarray],
                         endpoint_upper_ys: Optional[np.ndarray],
                         endpoint_lower_ys: Optional[np.ndarray]) -> pd.DataFrame:
    """Saves DynaFit dataframe_results as a pandas DataFrame (used for Excel/csv export)."""
    data = {
        'Parameter': to_column_text_format(original_parameters.keys()),
        'Value': to_column_text_format(original_parameters.values()),
        'Log2(Colony Size)': to_column_float_format(xs),
        'Log2(Variance)': to_column_float_format(ys),
        'Log2(Variance) upper CI': to_column_ci_format(upper_ys) if show_ci else None,
        'Log2(Variance) lower CI': to_column_ci_format(lower_ys) if show_ci else None,
        'Distance to H0 (cumulative)': to_column_ci_format(cumulative_ys),
        'Distance to H0 (cumulative) upper CI': to_column_ci_format(cumulative_upper_ys) if show_ci else None,
        'Distance to H0 (cumulative) lower CI': to_column_ci_format(cumulative_lower_ys) if show_ci else None,
        'Distance to H0 (endpoint)': to_column_ci_format(endpoint_ys),
        'Distance to H0 (endpoint) upper CI': to_column_ci_format(endpoint_upper_ys) if show_ci else None,
        'Distance to H0 (endpoint) lower CI': to_column_ci_format(endpoint_lower_ys) if show_ci else None,
    }
    return pd.DataFrame({k: v for k, v in data.items() if v is not None})


def to_column_text_format(a: Union[KeysView, ValuesView]) -> pd.Series:
    """Formats text arrays to the expected format of the results dataframe."""
    return pd.Series(list(a))


def to_column_float_format(a: np.array) -> pd.Series:
    """Formats numerical arrays to the expected format of the results dataframe."""
    return pd.Series(a.round(2))


def to_column_ci_format(a: np.array) -> pd.Series:
    """Formats numerical arrays confidence intervals to the expected format of the results dataframe."""
    return pd.Series(a.round(4))
