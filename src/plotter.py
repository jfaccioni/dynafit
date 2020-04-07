import matplotlib.pyplot as plt

from src.logic import PlotResults
from src.utils import get_missing_coord, get_start_end_values


class Plotter:
    """docstring"""
    def __init__(self, cvp_ax: plt.Axes, hist_ax: plt.Axes, results: PlotResults) -> None:
        """docstring"""
        self.cvp_ax = cvp_ax
        self.hist_ax = hist_ax
        self.xs = results.mean_xs
        self.ys = results.mean_ys
        self.upper_ys = results.upper_ys
        self.lower_ys = results.lower_ys
        self.scatter_xs = results.scatter_xs
        self.scatter_ys = results.scatter_ys
        self.scatter_colors = results.scatter_colors
        self.violin_ys = results.violin_ys
        self.violin_colors = results.violin_colors

    def plot_all(self):
        self.plot_mean_line()
        self.plot_supporting_lines()
        self.plot_bootstrap_scatter()
        if self.upper_ys is not None:
            self.plot_mean_line_ci()
            self.plot_supporting_lines_ci()
        if self.violin_ys is not None:
            self.plot_bootstrap_violins()

    def plot_mean_line(self) -> None:
        """Plots the mean value for each bootstrapped population as a line plot."""
        self.cvp_ax.plot(self.xs, self.ys, color='green', alpha=0.9, lw=3)

    def plot_supporting_lines(self) -> None:
        """Plots the three supporting lines of the CVP."""
        start_x, end_x = get_start_end_values(array=self.xs)
        start_y, end_y = get_start_end_values(array=self.ys)
        self.plot_h0(start=start_x, end=end_x, initial_height=start_y)
        self.plot_h1(start=start_x, end=end_x, initial_height=start_y)
        self.plot_vertical_axis(start=start_x)

    def plot_h0(self, start: float, end: float, initial_height: float) -> None:
        """Plots H0 on the CVP (horizontal blue line)"""
        self.cvp_ax.plot([start, end], [initial_height, initial_height], color='blue', lw=3)

    def plot_h1(self, start: float, end: float, initial_height: float) -> None:
        """Plots H1 on the CVP (diagonal red line)"""
        final_height = get_missing_coord(x1=start, y1=initial_height, x2=end)
        self.cvp_ax.plot([start, end], [initial_height, final_height], color='red', lw=3)

    def plot_vertical_axis(self, start: float) -> None:
        """Plots a bold vertical Y axis on the left limit of the CVP plot."""
        self.cvp_ax.axvline(start, color='darkgray', lw=3, zorder=0)

    def plot_bootstrap_scatter(self) -> None:
        """Plots bootstrap populations for each bin as scatter plots."""
        self.cvp_ax.scatter(self.scatter_xs, self.scatter_ys, marker='.', edgecolor='k',
                            facecolor=self.scatter_colors, alpha=0.3)

    def plot_mean_line_ci(self) -> None:
        """Plots the confidence interval around the mean line as a line plot."""
        self.cvp_ax.fill_between(self.xs, self.upper_ys, self.lower_ys, color='gray', alpha=0.5)

    def plot_supporting_lines_ci(self) -> None:
        """Plots the CI for the supporting lines of the CVP."""
        start_x, end_x = get_start_end_values(array=self.xs)
        upper_start_y, upper_end_y = get_start_end_values(array=self.upper_ys)
        lower_start_y, lower_end_y = get_start_end_values(array=self.lower_ys)
        self.plot_h0_ci(start=start_x, end=end_x, upper=upper_start_y, lower=lower_start_y)
        self.plot_h1_ci(start=start_x, end=end_x, upper=upper_start_y, lower=lower_start_y)

    def plot_h0_ci(self, start: float, end: float, upper: float, lower: float) -> None:
        """Plots H0 confidence interval on the CVP (diagonal red line)"""
        self.cvp_ax.fill_between([start, end], [upper, upper], [lower, lower], color='blue', alpha=0.3)

    def plot_h1_ci(self, start: float, end: float, upper: float, lower: float) -> None:
        """Plots H1 confidence interval on the CVP (diagonal red line)"""
        final_height_upper = get_missing_coord(x1=start, y1=upper, x2=end)
        final_height_lower = get_missing_coord(x1=start, y1=lower, x2=end)
        self.cvp_ax.fill_between([start, end], [upper, final_height_upper], [lower, final_height_lower],
                                 color='red', alpha=0.3)

    def plot_bootstrap_violins(self) -> None:
        """Plots the bootstrap populations for each bin as violin plots."""
        parts = self.cvp_ax.violinplot(positions=self.xs, dataset=self.violin_ys, showmeans=False,
                                       showmedians=False, showextrema=False)
        for i, body in enumerate(parts['bodies'], 1):
            body.set_facecolor('red') if i <= self.violin_colors else body.set_facecolor('gray')
            body.set_edgecolor('black')
            body.set_alpha(0.3)
