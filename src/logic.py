from operator import attrgetter
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
from openpyxl import Workbook

from src.validator import ExcelValidator


def dynafit(data: Workbook, sheetname: str, cs_start_cell: str, cs_end_cell: str, gr_start_cell: str, gr_end_cell: str,
            max_binned_colony_size: int, bins: int, runs: int, repeats: int, sample_size: int, fig: plt.Figure,
            cvp_ax: plt.Axes, hist_ax: plt.Axes) -> None:
    """Main function of this script"""
    df = ExcelValidator(data=data, sheetname=sheetname, start_cell_01=cs_start_cell, end_cell_01=cs_end_cell,
                        start_cell_02=gr_start_cell, end_cell_02=gr_end_cell).validated_data
    df = add_bins(df=df, max_binned_colony_size=max_binned_colony_size, bins=bins)
    for _ in range(runs):
        cs, gr = sample_data(df=df, repeats=repeats, sample_size=sample_size)
        cvp_ax.plot(cs, gr, color=(0, 0, 0, 0), marker='.', markeredgecolor='k', markerfacecolor='gray')
    title = get_plot_title(runs=runs, repeats=repeats, sample_size=sample_size)
    format_plot(fig=fig, ax=cvp_ax, title=title)
    plot_histogram(df=df, ax=hist_ax)
    # TODO: line below causes mismatched size error
    # return area_above_curve(mean_line=mean_line)


def load_data(data: Workbook, sheetname: str, cs_start_cell: str, cs_end_cell: str, gr_start_cell: str,
              gr_end_cell: str) -> pd.DataFrame:
    """Loads relevant data from input file as a Pandas DataFrame.
    ASSUMES: data to be in cell range A4 to B1071; modify this as needed"""
    ws = data[sheetname]
    cs_range = f'{cs_start_cell}:{cs_end_cell}'
    gr_range = f'{gr_start_cell}:{gr_end_cell}'
    values = [(cs[0].value, gr[0].value) for cs, gr in zip(ws[cs_range], ws[gr_range])]
    return pd.DataFrame(values, columns=('CS', 'GR'))


def add_bins(df: pd.DataFrame, max_binned_colony_size: int, bins: int) -> pd.DataFrame:
    """Returns the data DataFrame with a "bins" column, which divides the population of values in
    bins with a close number of instances in them"""
    df['bins'] = df['CS']
    bin_condition = df['bins'] > max_binned_colony_size
    binned_data = pd.qcut(df.loc[bin_condition]['CS'], bins, labels=False)
    binned_data += max_binned_colony_size + 1
    binned_data = pd.concat([binned_data, df.loc[~bin_condition]['bins']])
    return df.assign(bins=binned_data)


def sample_data(df: pd.DataFrame, repeats: int, sample_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Samples data and returns a tuple of arrays ready for plotting. This accounts for any
    necessary transformations before plotting, like log2."""
    sampling_result = random_sampling(df, repeats, sample_size)
    cs = np.log2(sampling_result.index)
    gr = sampling_result.mean(axis=1)  # means of each repeat across columns
    return cs, gr


def random_sampling(df: pd.DataFrame, repeats: int, sample_size: int) -> pd.DataFrame:
    """Samples each bin in the population. This is performed for a specific number of times (repeats),
    and each sampling contains a specific number of instances (sample_size)."""
    index = df.groupby('bins').mean()['CS']
    output = pd.DataFrame(index=index)
    for i in range(repeats):
        variances = []
        for groupname, group in df.groupby('bins'):
            sample = group.sample(n=sample_size)
            variances.append(sample['GR'].var())
        output[f'repeat_{i+1}'] = variances
    return output.applymap(np.log2)


def get_plot_title(runs, repeats, sample_size) -> str:
    """Returns a string for the plot's title."""
    return f'CVP: independent runs={runs}, sampling repeats={repeats}, sample size={sample_size}'


def format_plot(fig: plt.Figure, ax: plt.Axes, title: str) -> plt.Line2D:
    """Adds formatting to CVP."""
    fig.suptitle(title)
    ax.set_xlabel('log2(Colony Size)')
    ax.set_ylabel('log2(Growth Rate variance)')
    set_limits(ax)
    plot_mean_line(ax)
    plot_supporting_lines(ax)


def set_limits(ax: plt.Axes) -> None:
    """Sets ax limits based on max and min values of XY lines."""
    _, max_x, min_x = get_axes_params(ax, 'x')
    _, max_y, min_y = get_axes_params(ax, 'y')
    ax.set_xlim(min_x - (abs(min_x) * 0.05), max_x + (abs(max_x) * 0.05))
    ax.set_ylim(min_y - (abs(min_y) * 0.05), max_y + (abs(max_y) * 0.05))


def plot_supporting_lines(ax: plt.Axes) -> None:
    """Plots the three supporting lines of the CVP."""
    start_x, *_ = get_axes_params(ax, 'x')
    start_y, *_ = get_axes_params(ax, 'y')
    ax.axhline(start_y, color='blue', lw=3)
    ax.plot([start_x, 10], [start_y, start_y - 10], color='red', lw=3)
    ax.axvline(start_x, color='black', lw=3)


def plot_mean_line(ax: plt.Axes) -> plt.Line2D:
    """Plots mean line for all data in ax."""
    xs = [x for x in ax.lines[0].get_xdata()]
    ys = np.array([line.get_ydata() for line in ax.lines]).mean(axis=0)
    return ax.plot(xs, ys, color='green', alpha=0.9, lw=3)


def area_above_curve(mean_line: plt.Line2D) -> None:
    """Returns the area above the curve (mean green line)."""
    # triangle_area = max_x * () / 2
    pass


def perform_smoothing(xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Performs smoothing with scipy.interpolate"""
    # noinspection PyArgumentList
    new_xs = np.linspace(xs.min(), xs.max(), 100)
    spline = interpolate.BSpline(*interpolate.splrep(xs, ys, s=5, k=4), extrapolate=False)
    return new_xs, spline(new_xs)


def get_axes_params(ax: plt.Axes, coord: str) -> Tuple[float, float, float]:
    """Gets start, max and min position of all lines in ax for X or Y coordinate."""
    coord_data = attrgetter(f'get_{coord}data')
    start_coord = max([coord_data(line)()[0] for line in ax.lines])
    max_coord = max([v for line in ax.lines for v in coord_data(line)()])
    min_coord = min([v for line in ax.lines for v in coord_data(line)()])
    return start_coord, max_coord, min_coord


def plot_histogram(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plots a histogram of the colony size, indicating the "cuts" made by the binning process"""
    grouped_data = df.groupby('bins')
    ax.hist(np.log2(df['CS']))
    for xmax, label in zip(grouped_data.max()['CS'], grouped_data.count()['CS']):
        ax.axvline(np.log2(xmax), c='k')
        ax.text(np.log2(xmax), ax.get_ylim()[1] * 0.5, f'<={int(xmax)}\nn={label}')
