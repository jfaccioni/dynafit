"""plotter.py - defines a Plotter object."""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from seaborn import distplot

from src.utils import get_missing_coordinate, get_start_end_values


class Plotter:
    """Class that contains all information necessary to plot the DynaFit results"""
    hypothesis_plot_lower_ylim = -0.5
    hypothesis_plot_upper_ylim = 1.5
    data_color = '#4b006e'  # royal purple
    h0_color = '#be0119'  # scarlet
    h1_color = '#1e488f'  # cobalt
    cumul_color = '#fb7d07'  # pumpkin orange
    endp_color = data_color
    v_axis_color = 'black'
    scatter_edgecolor = 'black'
    scatter_facecolor = cumul_color
    violin_facecolor = data_color
    violin_edgecolor = 'black'
    violin_median_color = 'white'
    violin_whisker_color = 'black'
    hist_interval_color = 'black'

    def __init__(self, xs: np.ndarray, ys: np.ndarray, scatter_xs: np.ndarray, scatter_ys: np.ndarray,
                 show_violin: bool, violin_xs: Optional[np.ndarray], violin_ys: Optional[List[np.ndarray]],
                 violin_q1: Optional[np.ndarray], violin_medians: Optional[np.ndarray], violin_q3: Optional[np.ndarray],
                 cumulative_ys: np.ndarray, endpoint_ys: np.ndarray, show_ci: bool, upper_ys: Optional[np.ndarray],
                 lower_ys: Optional[np.ndarray], cumulative_upper_ys: Optional[np.ndarray],
                 cumulative_lower_ys: Optional[np.ndarray], endpoint_upper_ys: Optional[np.ndarray],
                 endpoint_lower_ys: Optional[np.ndarray], hist_xs: np.ndarray, hist_intervals: np.ndarray) -> None:
        """Init method of Plotter class."""
        self.xs = xs
        self.ys = ys
        self.scatter_xs = scatter_xs
        self.scatter_ys = scatter_ys
        self.show_violin = show_violin
        self.violin_xs = violin_xs
        self.violin_ys = violin_ys
        self.violin_q1 = violin_q1
        self.violin_medians = violin_medians
        self.violin_q3 = violin_q3
        self.cumulative_ys = cumulative_ys
        self.endpoint_ys = endpoint_ys
        self.show_ci = show_ci
        self.upper_ys = upper_ys
        self.lower_ys = lower_ys
        self.cumulative_upper_ys = cumulative_upper_ys
        self.cumulative_lower_ys = cumulative_lower_ys
        self.endpoint_upper_ys = endpoint_upper_ys
        self.endpoint_lower_ys = endpoint_lower_ys
        self.hist_xs = np.log2(hist_xs)
        self.hist_intervals = np.log2(hist_intervals)

    def plot_cvp_ax(self, ax: plt.Axes) -> None:
        """Calls all the functions related to plotting the CVP."""
        self.plot_supporting_lines(ax=ax)
        if self.show_ci:
            self.plot_supporting_lines_ci(ax=ax)
            self.plot_mean_line_ci(ax=ax)
        self.plot_mean_line(ax=ax)
        if self.show_violin:
            violins = self.plot_bootstrap_violins(ax=ax)
            self.format_violins(violins=violins)
            self.plot_bootstrap_violin_statistics(ax=ax)
        self.plot_bootstrap_scatter(ax=ax)
        self.format_cvp(ax=ax)

    def plot_supporting_lines(self, ax: plt.Axes) -> None:
        """Plots the three supporting lines of the CVP."""
        start_x, end_x = get_start_end_values(array=self.xs)
        start_y, end_y = get_start_end_values(array=self.ys)
        self.plot_h0(start=start_x, end=end_x, initial_height=start_y, ax=ax)
        self.plot_h1(start=start_x, end=end_x, initial_height=start_y, ax=ax)
        self.plot_vertical_axis(start=start_x, ax=ax)

    def plot_h0(self, start: float, end: float, initial_height: float, ax: plt.Axes) -> None:
        """Plots H0 on the CVP (horizontal red line)."""
        ax.plot([start, end], [initial_height, initial_height], color=self.h0_color, lw=3)

    def plot_h1(self, start: float, end: float, initial_height: float, ax: plt.Axes) -> None:
        """Plots H1 on the CVP (diagonal blue line)."""
        final_height = get_missing_coordinate(x1=start, y1=initial_height, x2=end)
        ax.plot([start, end], [initial_height, final_height], color=self.h1_color, lw=3)

    def plot_vertical_axis(self, start: float, ax: plt.Axes) -> None:
        """Plots a bold vertical Y axis on the left limit of the CVP plot."""
        ax.axvline(start, color=self.v_axis_color, lw=3, zorder=0)

    def plot_mean_line(self, ax: plt.Axes) -> None:
        """Plots the mean value for each bootstrapped population as a line plot."""
        ax.plot(self.xs, self.ys, color=self.data_color, lw=3)

    def plot_bootstrap_scatter(self, ax: plt.Axes) -> None:
        """Plots bootstrap populations for each bin as scatter plots."""
        ax.scatter(self.scatter_xs, self.scatter_ys, marker='.', alpha=0.6, edgecolor=self.scatter_edgecolor,
                   facecolor=self.scatter_facecolor)

    def plot_bootstrap_violins(self, ax: plt.Axes) -> List[PolyCollection]:
        """Plots the bootstrap populations for each bin as violin plots."""
        violins = ax.violinplot(positions=self.violin_xs, dataset=self.violin_ys, showextrema=False)
        return violins['bodies']

    def format_violins(self, violins: List[PolyCollection]):
        """Adds formatting to the violins plotted."""
        for violin in violins:
            violin.set_alpha(0.3)
            violin.set_facecolor(self.violin_facecolor)
            violin.set_edgecolor(self.violin_edgecolor)

    def plot_bootstrap_violin_statistics(self, ax: plt.Axes) -> None:
        """Plots the median and quantile statistics for the violins."""
        ax.scatter(self.violin_xs, self.violin_medians, color=self.violin_median_color, s=10, zorder=100, alpha=0.8)
        ax.vlines(self.violin_xs, self.violin_q1, self.violin_q3, color=self.violin_whisker_color, lw=5, alpha=0.8)

    def plot_supporting_lines_ci(self, ax: plt.Axes) -> None:
        """Plots the CI for the supporting lines of the CVP."""
        start_x, end_x = get_start_end_values(array=self.xs)
        upper_start_y, upper_end_y = get_start_end_values(array=self.upper_ys)
        lower_start_y, lower_end_y = get_start_end_values(array=self.lower_ys)
        self.plot_h0_ci(start=start_x, end=end_x, upper=upper_start_y, lower=lower_start_y, ax=ax)
        self.plot_h1_ci(start=start_x, end=end_x, upper=upper_start_y, lower=lower_start_y, ax=ax)

    def plot_h0_ci(self, start: float, end: float, upper: float, lower: float, ax: plt.Axes) -> None:
        """Plots H0 confidence interval on the CVP (horizontal red line)"""
        ax.fill_between([start, end], [upper, upper], [lower, lower], color=self.h0_color, alpha=0.1)

    def plot_h1_ci(self, start: float, end: float, upper: float, lower: float, ax: plt.Axes) -> None:
        """Plots H1 confidence interval on the CVP (diagonal blue line)"""
        upper_end = get_missing_coordinate(x1=start, y1=upper, x2=end)
        lower_end = get_missing_coordinate(x1=start, y1=lower, x2=end)
        ax.fill_between([start, end], [upper, upper_end], [lower, lower_end], color=self.h1_color, alpha=0.1)

    def plot_mean_line_ci(self, ax: plt.Axes) -> None:
        """Plots the confidence interval around the mean line as a line plot."""
        ax.fill_between(self.xs, self.upper_ys, self.lower_ys, color=self.data_color, alpha=0.1)

    @staticmethod
    def format_cvp(ax: plt.Axes) -> None:
        """Adds formatting to the CVP."""
        ax.set_xlabel('log2(Colony Size)')
        ax.set_ylabel('log2(Growth Rate Variance)')

    def plot_hypothesis_ax(self, ax: plt.Axes, xlims: Tuple[float, float]) -> None:
        """Calls all the functions related to plotting the hypothesis distance plot."""
        self.plot_hypothesis_lines(ax=ax)
        self.plot_cumulative_hypothesis_distance(ax=ax)
        self.plot_endpoint_hypothesis_distance(ax=ax)
        self.set_hypothesis_plot_limits(ax=ax, xlims=xlims)
        if self.show_ci:
            self.plot_cumulative_hypothesis_ci(ax=ax)
            self.plot_endpoint_hypothesis_ci(ax=ax)
        self.format_hypothesis_plot(ax=ax)

    def plot_hypothesis_lines(self, ax: plt.Axes) -> None:
        """Plots the hypothesis in the hypothesis plot as horizontal lines."""
        ax.axhline(0, color=self.h0_color, linestyle='dotted', alpha=0.8)
        ax.axhline(1, color=self.h1_color, linestyle='dotted', alpha=0.8)

    def plot_cumulative_hypothesis_distance(self, ax: plt.Axes) -> None:
        """Plots the cumulative hypothesis values as a line plot."""
        ax.plot(self.xs, self.cumulative_ys, color=self.cumul_color, label='Cumulative')

    def plot_endpoint_hypothesis_distance(self, ax: plt.Axes) -> None:
        """Plots the endpoint hypothesis values as a line plot."""
        ax.plot(self.xs, self.endpoint_ys, color=self.endp_color, label='Endpoint')

    def set_hypothesis_plot_limits(self, ax: plt.Axes, xlims: Tuple[float, float]) -> None:
        """Calculates appropriate limits for the XY axes in the hypothesis plot."""
        ax.set_xlim(*xlims)
        current_limits = ax.get_ylim()
        if current_limits[0] >= self.hypothesis_plot_lower_ylim:
            ax.set_ylim(bottom=self.hypothesis_plot_lower_ylim)
        if current_limits[1] <= self.hypothesis_plot_upper_ylim:
            ax.set_ylim(top=self.hypothesis_plot_upper_ylim)

    def plot_cumulative_hypothesis_ci(self, ax: plt.Axes) -> None:
        """Plots the CI around the cumulative hypothesis values as a line plot."""
        ax.fill_between(self.xs, self.cumulative_upper_ys, self.cumulative_lower_ys, alpha=0.2, color=self.cumul_color)

    def plot_endpoint_hypothesis_ci(self, ax: plt.Axes) -> None:
        """Plots the CI around the endpoint hypothesis values as a line plot."""
        ax.fill_between(self.xs, self.endpoint_upper_ys, self.endpoint_lower_ys, alpha=0.2, color=self.endp_color)

    @staticmethod
    def format_hypothesis_plot(ax: plt.Axes) -> None:
        """Adds formatting to the hypothesis plot."""
        ax.set_title('Hypothesis plot')
        ax.set_xlabel('log2(Colony Size)')
        ax.set_ylabel('Hypothesis')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['H0', 'H1'])
        ax.legend()

    def plot_histogram_ax(self, ax: plt.Axes) -> None:
        """Calls all the functions related to plotting the histogram."""
        self.plot_distributions(ax=ax)
        self.plot_group_divisions(ax=ax)
        self.format_histogram(ax=ax)

    def plot_distributions(self, ax: plt.Axes) -> None:
        """Plots the histogram."""
        distplot(self.hist_xs, bins=self.hist_intervals, ax=ax, color=self.data_color)

    def plot_group_divisions(self, ax: plt.Axes) -> None:
        """Plots the group divisions in the histogram as vertical lines."""
        ax.vlines(self.hist_intervals, *ax.get_ylim(), color=self.hist_interval_color, linestyle='dotted', alpha=0.8)

    @staticmethod
    def format_histogram(ax: plt.Axes) -> None:
        """Adds formatting to the histogram."""
        ax.set_title('Histogram of colony groups')
        ax.set_xlabel('log2(Colony Size)')
        ax.set_ylabel('% of colonies')
