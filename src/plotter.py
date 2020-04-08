import matplotlib.pyplot as plt
import numpy as np

from src.utils import get_missing_coord, get_start_end_values


class Plotter:
    """docstring"""
    def __init__(self, mean_xs: np.ndarray, mean_ys: np.ndarray, upper_ys: np.ndarray, lower_ys: np.ndarray,
                 scatter_xs: np.ndarray, scatter_ys: np.ndarray, scatter_colors: np.ndarray, violin_ys: np.ndarray,
                 violin_colors: int, hist_x: np.ndarray, hist_pos: np.ndarray, hist_bin_mins: np.ndarray,
                 hist_bin_maxs: np.ndarray, hist_instances: np.ndarray) -> None:
        """docstring"""
        self.xs = mean_xs
        self.ys = mean_ys
        self.upper_ys = upper_ys
        self.lower_ys = lower_ys
        self.scatter_xs = scatter_xs
        self.scatter_ys = scatter_ys
        self.scatter_colors = scatter_colors
        self.violin_ys = violin_ys
        self.violin_colors = violin_colors
        self.hist_x = hist_x
        self.hist_pos = hist_pos
        self.hist_bin_mins = hist_bin_mins
        self.hist_bin_maxs = hist_bin_maxs
        self.hist_instances = hist_instances

    def plot_cvp(self, ax: plt.Axes):
        """Calls all the functions related to plotting the CVP."""
        self.plot_mean_line(ax=ax)
        self.plot_supporting_lines(ax=ax)
        self.plot_bootstrap_scatter(ax=ax)
        if self.upper_ys is not None:
            self.plot_mean_line_ci(ax=ax)
            self.plot_supporting_lines_ci(ax=ax)
        if self.violin_ys is not None:
            self.plot_bootstrap_violins(ax=ax)

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
        """Plots H0 on the CVP (horizontal blue line)"""
        ax.plot([start, end], [initial_height, initial_height], color='blue', lw=3)

    @staticmethod
    def plot_h1(start: float, end: float, initial_height: float, ax: plt.Axes) -> None:
        """Plots H1 on the CVP (diagonal red line)"""
        final_height = get_missing_coord(x1=start, y1=initial_height, x2=end)
        ax.plot([start, end], [initial_height, final_height], color='red', lw=3)

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
        """Plots H0 confidence interval on the CVP (diagonal red line)"""
        ax.fill_between([start, end], [upper, upper], [lower, lower], color='blue', alpha=0.3)

    @staticmethod
    def plot_h1_ci(start: float, end: float, upper: float, lower: float, ax: plt.Axes) -> None:
        """Plots H1 confidence interval on the CVP (diagonal red line)"""
        final_height_upper = get_missing_coord(x1=start, y1=upper, x2=end)
        final_height_lower = get_missing_coord(x1=start, y1=lower, x2=end)
        ax.fill_between([start, end], [upper, final_height_upper], [lower, final_height_lower], color='red', alpha=0.3)

    def plot_bootstrap_violins(self, ax: plt.Axes) -> None:
        """Plots the bootstrap populations for each bin as violin plots."""
        parts = ax.violinplot(positions=self.xs, dataset=self.violin_ys, showmeans=False, showmedians=False,
                              showextrema=False)
        for i, body in enumerate(parts['bodies'], 1):
            body.set_facecolor('red') if i <= self.violin_colors else body.set_facecolor('gray')
            body.set_edgecolor('black')
            body.set_alpha(0.3)

    def plot_histogram(self, ax: plt.Axes) -> None:
        """Plots the histogram, indicating group sizes made during the binning process.."""
        ax.hist(self.hist_x)
        for pos, bn, bx, num in zip(self.hist_pos, self.hist_bin_mins, self.hist_bin_maxs, self.hist_instances):
            ax.axvline(pos, c='k')
            text = f'{bn}\n{bx}\nn={num}'
            ax.text(pos, ax.get_ylim()[1] * 0.5, text)
