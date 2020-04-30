"""test_plotter.py - unit tests for plotter.py."""

import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from src.plotter import Plotter


class TestPlotterModule(unittest.TestCase):
    """Tests the plotter.py module."""
    plotter_kwargs = {
        'xs': np.array([1, 2, 3]),
        'ys': np.array([1, 2, 3]),
        'scatter_xs': np.array([1, 2, 3]),
        'scatter_ys': np.array([1, 2, 3]),
        'scatter_colors': np.array(['red', 'red', 'gray']),
        'show_violin': True,
        'violin_ys': [np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([1, 2, 3])],
        'violin_colors':  np.array(['red', 'red', 'gray']),
        'cumulative_hyp_ys': np.array([1, 2, 3]),
        'endpoint_hyp_ys': np.array([1, 2, 3]),
        'show_ci': True,
        'upper_ys': np.array([1, 2, 3]),
        'lower_ys': np.array([1, 2, 3]),
        'cumulative_hyp_upper_ys': np.array([1, 2, 3]),
        'cumulative_hyp_lower_ys': np.array([1, 2, 3]),
        'endpoint_hyp_upper_ys': np.array([1, 2, 3]),
        'endpoint_hyp_lower_ys': np.array([1, 2, 3]),
        'hist_x': np.array([1, 2, 3]),
        'hist_breakpoints': np.array([1, 2, 3]),
        'hist_instances': np.array([1, 2, 3]),
    }

    @classmethod
    def setUpClass(cls) -> None:
        """Sets up the entire test suite test by creating a Figure and three Axes instances."""
        cls.fig = plt.Figure(facecolor='white', figsize=(12, 12))
        cls.cvp_ax = cls.fig.add_axes([0.1, 0.45, 0.85, 0.5], label='cvp')
        cls.hypothesis_ax = cls.fig.add_axes([0.1, 0.25, 0.85, 0.1], label='hypothesis')
        cls.histogram_ax = cls.fig.add_axes([0.1, 0.05, 0.85, 0.1], label='histogram')

    def setUp(self) -> None:
        """Sets up each unit test by refreshing the Plotter instance."""
        self.plotter = Plotter(**self.plotter_kwargs)

    def tearDown(self) -> None:
        """Tears down each unit test by clearing the Axes instances."""
        self.cvp_ax.clear()
        self.hypothesis_ax.clear()
        self.histogram_ax.clear()

    def disable_violins(self) -> None:
        """Changes Plotter instance attributes to reflect a DynaFit analysis without violin plots enabled."""
        self.plotter.show_violin = False
        self.plotter.violin_ys = None
        self.plotter.violin_colors = None

    def disable_ci(self) -> None:
        """Changes Plotter instance attributes to reflect a DynaFit analysis without confidence interval enabled."""
        self.plotter.show_ci = False
        self.plotter.upper_ys = None
        self.plotter.lower_ys = None
        self.plotter.cumulative_hyp_upper_ys = None
        self.plotter.cumulative_hyp_lower_ys = None
        self.plotter.endpoint_hyp_upper_ys = None
        self.plotter.endpoint_hyp_lower_ys = None

    @patch('test_plotter.Plotter.plot_bootstrap_violins')
    @patch('test_plotter.Plotter.plot_supporting_lines_ci')
    @patch('test_plotter.Plotter.plot_mean_line_ci')
    def test_plot_cvp_ax_plots_everything_if_boolean_flags_are_set_to_true(self, mock_plot_mean_line_ci,
                                                                           mock_plot_supporting_lines_ci,
                                                                           mock_plot_bootstrap_violins) -> None:
        self.plotter.plot_cvp_ax(ax=self.cvp_ax)
        for mock_plot_function in (mock_plot_mean_line_ci, mock_plot_supporting_lines_ci, mock_plot_bootstrap_violins):
            with self.subTest(mock_plot_function=mock_plot_function):
                mock_plot_function.assert_called_with(ax=self.cvp_ax)

    @patch('test_plotter.Plotter.plot_bootstrap_violins')
    @patch('test_plotter.Plotter.plot_supporting_lines_ci')
    @patch('test_plotter.Plotter.plot_mean_line_ci')
    def test_plot_cvp_ax_does_not_plot_violins_if_flag_is_set_to_false(self, mock_plot_mean_line_ci,
                                                                       mock_plot_supporting_lines_ci,
                                                                       mock_plot_bootstrap_violins) -> None:
        self.disable_violins()
        self.plotter.plot_cvp_ax(ax=self.cvp_ax)
        for mock_plot_function in (mock_plot_mean_line_ci, mock_plot_supporting_lines_ci):
            with self.subTest(mock_plot_function=mock_plot_function):
                mock_plot_function.assert_called_with(ax=self.cvp_ax)
        mock_plot_bootstrap_violins.assert_not_called()

    @patch('test_plotter.Plotter.plot_bootstrap_violins')
    @patch('test_plotter.Plotter.plot_supporting_lines_ci')
    @patch('test_plotter.Plotter.plot_mean_line_ci')
    def test_plot_cvp_ax_does_not_add_ci_if_flag_is_set_to_false(self, mock_plot_mean_line_ci,
                                                                 mock_plot_supporting_lines_ci,
                                                                 mock_plot_bootstrap_violins) -> None:
        self.disable_ci()
        self.plotter.plot_cvp_ax(ax=self.cvp_ax)
        mock_plot_bootstrap_violins.assert_called_with(ax=self.cvp_ax)
        for mock_plot_function in (mock_plot_mean_line_ci, mock_plot_supporting_lines_ci):
            with self.subTest(mock_plot_function=mock_plot_function):
                mock_plot_function.assert_not_called()


if __name__ == '__main__':
    unittest.main()
