"""plotter.py - defines a Plotter object."""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.utils import get_missing_coord, get_start_end_values


class Plotter:
    """Class that contains all information necessary to plot the DynaFit results"""
    def __init__(self, xs: np.ndarray, ys: np.ndarray, scatter_xs: np.ndarray, scatter_ys: np.ndarray,
                 scatter_colors: np.ndarray, show_violin: bool, violin_ys: Optional[List[np.ndarray]],
                 violin_colors: Optional[List[str]], cumcody_ys: np.ndarray, endcody_ys: np.ndarray, show_ci: bool,
                 upper_ys: Optional[np.ndarray], lower_ys: Optional[np.ndarray], cumcody_upper_ys: Optional[np.ndarray],
                 cumcody_lower_ys: Optional[np.ndarray], endcody_upper_ys: Optional[np.ndarray],
                 endcody_lower_ys: Optional[np.ndarray], hist_x: np.ndarray, hist_breakpoints: np.ndarray,
                 hist_instances: np.ndarray, ) -> None:
        """Init method of Plotter class."""
        self.xs = xs
        self.ys = ys
        self.scatter_xs = scatter_xs
        self.scatter_ys = scatter_ys
        self.scatter_colors = scatter_colors
        self.show_violin = show_violin
        self.violin_ys = violin_ys
        self.violin_colors = violin_colors
        self.cumcody_ys = cumcody_ys
        self.endcody_ys = endcody_ys
        self.show_ci = show_ci
        self.upper_ys = upper_ys
        self.lower_ys = lower_ys
        self.cumcody_upper_ys = cumcody_upper_ys
        self.cumcody_lower_ys = cumcody_lower_ys
        self.endcody_upper_ys = endcody_upper_ys
        self.endcody_lower_ys = endcody_lower_ys
        self.hist_x = hist_x
        self.hist_breakpoints = hist_breakpoints
        self.hist_instances = hist_instances

    def plot_cvp_ax(self, ax: plt.Axes):
        """Calls all the functions related to plotting the CVP."""
        self.plot_supporting_lines(ax=ax)
        self.plot_mean_line(ax=ax)
        self.plot_bootstrap_scatter(ax=ax)
        if self.show_violin:
            self.plot_bootstrap_violins(ax=ax)
        if self.show_ci:
            self.plot_supporting_lines_ci(ax=ax)
            self.plot_mean_line_ci(ax=ax)
        ax.set_xlabel('log2(Colony Size)')
        ax.set_ylabel('log2(Growth Rate Variance)')

    def plot_mean_line(self, ax: plt.Axes) -> None:
        """Plots the mean value for each bootstrapped population as a line plot."""
        ax.plot(self.xs, self.ys, color='green', alpha=0.9, lw=3)

    def plot_supporting_lines(self, ax: plt.Axes) -> None:
        """Plots the three supporting lines of the CVP."""
        start_x, end_x = get_start_end_values(array=self.xs)
        start_y, end_y = get_start_end_values(array=self.ys)
        self.plot_h0(start=start_x, end=end_x, initial_height=start_y, ax=ax)
        self.plot_h1(start=start_x, end=end_x, initial_height=start_y, ax=ax)
        self.plot_vertical_axis(start=start_x, ax=ax)

    @staticmethod
    def plot_h0(start: float, end: float, initial_height: float, ax: plt.Axes) -> None:
        """Plots H0 on the CVP (horizontal red line)"""
        ax.plot([start, end], [initial_height, initial_height], color='red', lw=3)

    @staticmethod
    def plot_h1(start: float, end: float, initial_height: float, ax: plt.Axes) -> None:
        """Plots H1 on the CVP (diagonal blue line)"""
        final_height = get_missing_coord(x1=start, y1=initial_height, x2=end)
        ax.plot([start, end], [initial_height, final_height], color='blue', lw=3)

    @staticmethod
    def plot_vertical_axis(start: float, ax: plt.Axes) -> None:
        """Plots a bold vertical Y axis on the left limit of the CVP plot."""
        ax.axvline(start, color='darkgray', lw=3, zorder=0)

    def plot_bootstrap_scatter(self, ax: plt.Axes) -> None:
        """Plots bootstrap populations for each bin as scatter plots."""
        ax.scatter(self.scatter_xs, self.scatter_ys, marker='.', edgecolor='k', facecolor=self.scatter_colors,
                   alpha=0.3)

    def plot_mean_line_ci(self, ax: plt.Axes) -> None:
        """Plots the confidence interval around the mean line as a line plot."""
        ax.fill_between(self.xs, self.upper_ys, self.lower_ys, color='gray', alpha=0.5)

    def plot_supporting_lines_ci(self, ax: plt.Axes) -> None:
        """Plots the CI for the supporting lines of the CVP."""
        start_x, end_x = get_start_end_values(array=self.xs)
        upper_start_y, upper_end_y = get_start_end_values(array=self.upper_ys)
        lower_start_y, lower_end_y = get_start_end_values(array=self.lower_ys)
        self.plot_h0_ci(start=start_x, end=end_x, upper=upper_start_y, lower=lower_start_y, ax=ax)
        self.plot_h1_ci(start=start_x, end=end_x, upper=upper_start_y, lower=lower_start_y, ax=ax)

    @staticmethod
    def plot_h0_ci(start: float, end: float, upper: float, lower: float, ax: plt.Axes) -> None:
        """Plots H0 confidence interval on the CVP (horizontal red line)"""
        ax.fill_between([start, end], [upper, upper], [lower, lower], color='red', alpha=0.3)

    @staticmethod
    def plot_h1_ci(start: float, end: float, upper: float, lower: float, ax: plt.Axes) -> None:
        """Plots H1 confidence interval on the CVP (diagonal blue line)"""
        final_height_upper = get_missing_coord(x1=start, y1=upper, x2=end)
        final_height_lower = get_missing_coord(x1=start, y1=lower, x2=end)
        ax.fill_between([start, end], [upper, final_height_upper], [lower, final_height_lower], color='blue', alpha=0.3)

    def plot_bootstrap_violins(self, ax: plt.Axes) -> None:
        """Plots the bootstrap populations for each bin as violin plots."""
        parts = ax.violinplot(positions=self.xs, dataset=self.violin_ys, showmeans=False, showmedians=False,
                              showextrema=False)
        for body, color in zip(parts['bodies'], self.violin_colors):
            body.set_facecolor(color)
            body.set_edgecolor('black')
            body.set_alpha(0.3)

    def plot_histogram_ax(self, ax: plt.Axes) -> None:
        """Calls all the functions related to plotting the histogram."""
        self.plot_distributions(ax=ax)
        self.plot_group_divisions(ax=ax)
        ax.set_title('Histogram of colony groups')
        ax.set_xlabel('log2(Colony Size)')
        ax.set_ylabel('% of colonies')

    def plot_distributions(self, ax: plt.Axes) -> None:
        """Plots the histogram."""
        sns.distplot(np.log2(self.hist_x), bins=np.log2(self.hist_breakpoints), ax=ax)

    def plot_group_divisions(self, ax: plt.Axes) -> None:
        """Plots the group divisions in the histogram as vertical lines."""
        for bp in self.hist_breakpoints:
            ax.axvline(np.log2(bp), color='black', linestyle='dotted', alpha=0.8)

    def plot_cody_ax(self, ax: plt.Axes, xlims: Tuple[float, float]) -> None:
        """Calls all the functions related to plotting the CoDy value."""
        self.plot_hypothesis_lines(ax=ax)
        self.plot_cumcody_values(ax=ax)
        self.plot_endcody_values(ax=ax)
        self.set_cody_limits(ax=ax, xlims=xlims)
        if self.show_ci:
            self.plot_cumcody_ci_values(ax=ax)
            self.plot_relcody_ci_values(ax=ax)
        ax.set_title('Hypothesis plot')
        ax.set_xlabel('log2(Colony Size)')
        ax.set_ylabel('Hypothesis')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['H0', 'H1'])
        ax.legend()

    @staticmethod
    def plot_hypothesis_lines(ax: plt.Axes) -> None:
        """Plots the hypothesis in the cody plot as horizontal lines."""
        ax.axhline(0, color='red', linestyle='dotted', alpha=0.8)
        ax.axhline(1, color='blue', linestyle='dotted', alpha=0.8)

    def plot_cumcody_values(self, ax: plt.Axes) -> None:
        """Plots the cumulative CoDy values as a line plot."""
        ax.plot(self.xs, self.cumcody_ys, color='lightgreen', label='Cumulative')

    def plot_endcody_values(self, ax: plt.Axes) -> None:
        """Plots the endpoint CoDy values as a line plot."""
        ax.plot(self.xs, self.endcody_ys, color='darkgreen', label='Endpoint')

    def plot_cumcody_ci_values(self, ax: plt.Axes) -> None:
        """Plots the CI around the cumulative CoDy values as a line plot."""
        ax.fill_between(self.xs, self.cumcody_upper_ys, self.cumcody_lower_ys, color='gray', alpha=0.5)

    def plot_relcody_ci_values(self, ax: plt.Axes) -> None:
        """Plots the CI around the endpoint CoDy values as a line plot."""
        ax.fill_between(self.xs, self.endcody_upper_ys, self.endcody_lower_ys, color='gray', alpha=0.5)

    @staticmethod
    def set_cody_limits(ax: plt.Axes, xlims: Tuple[float, float]) -> None:
        """Calculates appropriate limits for the XY axes in the CoDy plot."""
        ax.set_xlim(*xlims)
        current_limits = ax.get_ylim()
        if current_limits[0] >= -0.5:
            ax.set_ylim(bottom=-0.5)
        if current_limits[1] <= 1.5:
            ax.set_ylim(top=1.5)
