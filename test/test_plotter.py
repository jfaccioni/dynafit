"""test_plotter.py - unit tests for plotter.py."""

import unittest
from typing import Sequence
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from src.plotter import Plotter
from src.utils import array_in_sequence


class TestPlotterModule(unittest.TestCase):
    """Tests the plotter.py module."""
    plotter_kwargs = {
        'xs': np.array([1, 2, 3]),
        'ys': np.array([4, 5, 6]),
        'scatter_xs': np.array([7, 8, 9]),
        'scatter_ys': np.array([10, 11, 12]),
        'show_violin': True,
        'violin_xs': np.array([13, 14, 15]),
        'violin_ys': [np.array([16, 17, 18]), np.array([19, 20, 21]), np.array([22, 23, 24])],
        'violin_q1': np.array([25, 26, 27]),
        'violin_medians': np.array([28, 29, 30]),
        'violin_q3': np.array([31, 32, 33]),
        'cumulative_ys': np.array([34, 35, 36]),
        'endpoint_ys': np.array([37, 38, 39]),
        'show_ci': True,
        'upper_ys': np.array([40, 41, 42]),
        'lower_ys': np.array([43, 44, 45]),
        'cumulative_upper_ys': np.array([46, 47, 48]),
        'cumulative_lower_ys': np.array([49, 50, 51]),
        'endpoint_upper_ys': np.array([52, 53, 54]),
        'endpoint_lower_ys': np.array([55, 56, 57]),
        'hist_xs': np.array([58, 59, 60]),
        'hist_intervals': np.array([61, 62, 63]),
    }

    def setUp(self) -> None:
        """Sets up each unit test by refreshing the Plotter instance, the MagicMock instance representing an Axes
        instance, the Figure instance and the Axes instance."""
        self.plotter = Plotter(**self.plotter_kwargs)
        self.mock_ax = MagicMock()
        self.fig, self.ax = plt.subplots()

    def tearDown(self) -> None:
        """Tears down each unit test by deleting the Figure and Axes instances."""
        self.ax.clear()
        plt.close(self.fig)
        del self.ax
        del self.fig

    def assertArrayIn(self, array: np.ndarray, sequence: Sequence) -> None:
        """Asserts whether a numpy array is inside a regular Python sequence."""
        self.assertTrue(array_in_sequence(array, sequence))

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
        self.plotter.cumulative_upper_ys = None
        self.plotter.cumulative_lower_ys = None
        self.plotter.endpoint_upper_ys = None
        self.plotter.endpoint_lower_ys = None

    @patch('test_plotter.Plotter.plot_supporting_lines')
    @patch('test_plotter.Plotter.plot_supporting_lines_ci')
    @patch('test_plotter.Plotter.plot_mean_line_ci')
    @patch('test_plotter.Plotter.plot_mean_line')
    @patch('test_plotter.Plotter.plot_bootstrap_violin_statistics')
    @patch('test_plotter.Plotter.plot_bootstrap_scatter')
    @patch('test_plotter.Plotter.format_cvp')
    @patch('test_plotter.Plotter.plot_bootstrap_violins')
    @patch('test_plotter.Plotter.format_violins')
    def test_plot_cvp_ax_calls_all_cvp_related_plot_functions(self, mock_format_violins, mock_plot_bootstrap_violins,
                                                              *cvp_functions) -> None:
        self.plotter.plot_cvp_ax(ax=self.mock_ax)
        mock_format_violins.assert_called_with(violins=mock_plot_bootstrap_violins.return_value)
        for cvp_function in cvp_functions:
            cvp_function.assert_called_with(ax=self.mock_ax)

    @patch('test_plotter.Plotter.plot_bootstrap_violins')
    @patch('test_plotter.Plotter.plot_supporting_lines_ci')
    @patch('test_plotter.Plotter.plot_mean_line_ci')
    def test_plot_cvp_ax_plots_everything_if_boolean_flags_are_set_to_true(self, mock_plot_mean_line_ci,
                                                                           mock_plot_supporting_lines_ci,
                                                                           mock_plot_bootstrap_violins) -> None:
        self.plotter.plot_cvp_ax(ax=self.mock_ax)
        for mock_plot_function in (mock_plot_mean_line_ci, mock_plot_supporting_lines_ci, mock_plot_bootstrap_violins):
            with self.subTest(mock_plot_function=mock_plot_function):
                mock_plot_function.assert_called_with(ax=self.mock_ax)

    @patch('test_plotter.Plotter.plot_bootstrap_violins')
    @patch('test_plotter.Plotter.plot_supporting_lines_ci')
    @patch('test_plotter.Plotter.plot_mean_line_ci')
    def test_plot_cvp_ax_does_not_plot_violins_if_flag_is_set_to_false(self, mock_plot_mean_line_ci,
                                                                       mock_plot_supporting_lines_ci,
                                                                       mock_plot_bootstrap_violins) -> None:
        self.disable_violins()
        self.plotter.plot_cvp_ax(ax=self.mock_ax)
        for mock_plot_function in (mock_plot_mean_line_ci, mock_plot_supporting_lines_ci):
            with self.subTest(mock_plot_function=mock_plot_function):
                mock_plot_function.assert_called_with(ax=self.mock_ax)
        mock_plot_bootstrap_violins.assert_not_called()

    @patch('test_plotter.Plotter.plot_bootstrap_violins')
    @patch('test_plotter.Plotter.plot_supporting_lines_ci')
    @patch('test_plotter.Plotter.plot_mean_line_ci')
    def test_plot_cvp_ax_does_not_add_ci_if_flag_is_set_to_false(self, mock_plot_mean_line_ci,
                                                                 mock_plot_supporting_lines_ci,
                                                                 mock_plot_bootstrap_violins) -> None:
        self.disable_ci()
        self.plotter.plot_cvp_ax(ax=self.mock_ax)
        mock_plot_bootstrap_violins.assert_called_with(ax=self.mock_ax)
        for mock_plot_function in (mock_plot_mean_line_ci, mock_plot_supporting_lines_ci):
            with self.subTest(mock_plot_function=mock_plot_function):
                mock_plot_function.assert_not_called()

    def test_plot_supporting_lines_plots_h0_and_h1_as_line_plots(self) -> None:
        self.plotter.plot_supporting_lines(ax=self.mock_ax)
        self.assertEqual(self.mock_ax.plot.call_count, 2)

    def test_plot_h0_plots_a_red_horizontal_line(self) -> None:
        with patch('test_plotter.Plotter.plot_h1'):  # do not call ax.plot inside Plotter.plot_h1 for this test
            self.plotter.plot_supporting_lines(ax=self.mock_ax)
        actual_args, actual_kwargs = self.mock_ax.plot.call_args
        self.assertEqual(*actual_args[-1])  # horizontal line: start and end Y coordinates are equal for h0
        self.assertIn(self.plotter.h0_color, actual_kwargs.values())

    def test_plot_h1_plots_a_blue_diagonal_line(self) -> None:
        with patch('test_plotter.Plotter.plot_h0'):  # do not call ax.plot inside Plotter.plot_h0 for this test
            self.plotter.plot_supporting_lines(ax=self.mock_ax)
        actual_args, actual_kwargs = self.mock_ax.plot.call_args
        self.assertGreater(*actual_args[-1])  # diagonal line: end Y coordinate is below start Y coordinate
        self.assertIn(self.plotter.h1_color, actual_kwargs.values())

    def test_plot_supporting_lines_plots_vertical_y_axis_as_a_vertical_line(self) -> None:
        self.plotter.plot_supporting_lines(ax=self.mock_ax)
        self.mock_ax.axvline.assert_called_once()

    def test_plot_mean_line_plots_a_green_line_of_sample_xy_values(self) -> None:
        self.plotter.plot_mean_line(ax=self.mock_ax)
        self.mock_ax.plot.assert_called_once()
        actual_args, actual_kwargs = self.mock_ax.plot.call_args
        self.assertIn(self.plotter.data_color, actual_kwargs.values())
        self.assertArrayIn(self.plotter.xs, actual_args)
        self.assertArrayIn(self.plotter.ys, actual_args)

    def test_plot_bootstrap_scatter_plots_scatter_xs_and_ys(self) -> None:
        self.plotter.plot_bootstrap_scatter(ax=self.mock_ax)
        self.mock_ax.scatter.assert_called_once()
        actual_args, _ = self.mock_ax.scatter.call_args
        self.assertArrayIn(self.plotter.scatter_xs, actual_args)
        self.assertArrayIn(self.plotter.scatter_ys, actual_args)

    def test_plot_bootstrap_scatter_uses_scatter_edgecolor_and_facecolor_attributes(self) -> None:
        self.plotter.plot_bootstrap_scatter(ax=self.mock_ax)
        self.mock_ax.scatter.assert_called_once()
        _, actual_kwargs = self.mock_ax.scatter.call_args
        self.assertIn(self.plotter.scatter_edgecolor, actual_kwargs.values())
        self.assertIn(self.plotter.scatter_facecolor, actual_kwargs.values())

    def test_plot_bootstrap_violins_plots_violins(self) -> None:
        self.plotter.plot_bootstrap_violins(ax=self.mock_ax)
        self.mock_ax.violinplot.assert_called_once()
        actual_args, actual_kwargs = self.mock_ax.violinplot.call_args
        self.assertArrayIn(self.plotter.violin_xs, actual_kwargs.values())
        for expected_violin_array, actual_violin_array in zip(self.plotter.violin_ys, actual_kwargs.get('dataset')):
            with self.subTest(expected_violin_array=expected_violin_array, actual_violin_array=actual_violin_array):
                np.testing.assert_allclose(expected_violin_array, actual_violin_array)

    def test_plot_bootstrap_violins_returns_violins_as_a_list_of_polycollection_objects(self) -> None:
        return_value = self.plotter.plot_bootstrap_violins(ax=self.mock_ax)
        for expected_violin in return_value:
            self.assertIsInstance(expected_violin, PolyCollection)

    def test_format_violins_sets_violin_attributes_with_proper_values(self) -> None:
        mock_violin = MagicMock()
        self.plotter.format_violins(violins=[mock_violin])
        mock_violin.set_facecolor.assert_called_with(self.plotter.violin_facecolor)
        mock_violin.set_edgecolor.assert_called_with(self.plotter.violin_edgecolor)

    def test_plot_supporting_lines_ci_plots_h0_ci_and_h1_ci_as_filled_areas(self) -> None:
        self.plotter.plot_supporting_lines_ci(ax=self.mock_ax)
        self.assertEqual(self.mock_ax.fill_between.call_count, 2)

    def test_plot_h0_ci_fills_a_red_horizontal_area(self) -> None:
        with patch('test_plotter.Plotter.plot_h1_ci'):  # avoids ax.fill_between call inside Plotter.plot_h1_ci
            self.plotter.plot_supporting_lines_ci(ax=self.mock_ax)
        actual_args, actual_kwargs = self.mock_ax.fill_between.call_args
        self.assertEqual(*actual_args[-2])  # horizontal upper CI: start and end Y coordinates are equal for h0
        self.assertEqual(*actual_args[-1])  # horizontal lower CI: start and end Y coordinates are equal for h0
        self.assertIn(self.plotter.h0_color, actual_kwargs.values())

    def test_plot_h0_ci_fills_a_blue_diagonal_area(self) -> None:
        with patch('test_plotter.Plotter.plot_h0_ci'):  # avoids ax.fill_between call inside Plotter.plot_h0_ci
            self.plotter.plot_supporting_lines_ci(ax=self.mock_ax)
        actual_args, actual_kwargs = self.mock_ax.fill_between.call_args
        self.assertGreater(*actual_args[-2])  # diagonal upper CI: end Y coordinate is below start Y coordinate
        self.assertGreater(*actual_args[-1])  # diagonal lower CI: end Y coordinate is below start Y coordinate
        self.assertIn(self.plotter.h1_color, actual_kwargs.values())

    def test_plot_mean_line_ci_fills_an_area_of_xs_and_ys_values(self) -> None:
        self.plotter.plot_mean_line_ci(ax=self.mock_ax)
        self.mock_ax.fill_between.assert_called_once()
        actual_args, _ = self.mock_ax.fill_between.call_args
        self.assertArrayIn(self.plotter.xs, actual_args)
        self.assertArrayIn(self.plotter.upper_ys, actual_args)
        self.assertArrayIn(self.plotter.lower_ys, actual_args)

    def test_plot_mean_line_ci_fills_an_area_with_correct_color(self) -> None:
        self.plotter.plot_mean_line_ci(ax=self.mock_ax)
        _, actual_kwargs = self.mock_ax.fill_between.call_args
        self.assertIn(self.plotter.data_color, actual_kwargs.values())

    def test_format_cvp_adds_xy_labels(self) -> None:
        self.assertFalse(self.ax.get_xlabel())
        self.assertFalse(self.ax.get_ylabel())
        self.plotter.format_cvp(ax=self.ax)
        self.assertTrue(self.ax.get_xlabel())
        self.assertTrue(self.ax.get_ylabel())

    @patch('test_plotter.Plotter.plot_hypothesis_lines')
    @patch('test_plotter.Plotter.plot_cumulative_hypothesis_distance')
    @patch('test_plotter.Plotter.plot_endpoint_hypothesis_distance')
    @patch('test_plotter.Plotter.plot_cumulative_hypothesis_ci')
    @patch('test_plotter.Plotter.plot_endpoint_hypothesis_ci')
    @patch('test_plotter.Plotter.format_hypothesis_plot')
    @patch('test_plotter.Plotter.invert_hypothesis_plot_y_axis')
    @patch('test_plotter.Plotter.set_hypothesis_plot_limits')
    def test_plot_hypothesis_ax_calls_all_hypothesis_related_plot_functions(self, mock_set_hypothesis_plot_limits,
                                                                            *hypothesis_functions) -> None:
        xlims = (0, 5)
        self.plotter.plot_hypothesis_ax(ax=self.mock_ax, xlims=xlims)
        mock_set_hypothesis_plot_limits.assert_called_with(ax=self.mock_ax, xlims=xlims)
        for hypothesis_function in hypothesis_functions:
            hypothesis_function.assert_called_with(ax=self.mock_ax)

    @patch('test_plotter.Plotter.plot_endpoint_hypothesis_ci')
    @patch('test_plotter.Plotter.plot_endpoint_hypothesis_distance')
    @patch('test_plotter.Plotter.plot_cumulative_hypothesis_ci')
    @patch('test_plotter.Plotter.plot_cumulative_hypothesis_distance')
    def test_plot_hypothesis_ax_returns_values_from_appropriate_functions(self, mock_cumulative_distance,
                                                                          mock_cumulative_ci, mock_endpoint_distance,
                                                                          mock_endpoint_ci) -> None:
        return_value = self.plotter.plot_hypothesis_ax(ax=self.ax, xlims=(0, 5))
        (cumulative_line, cumulative_ci), (endpoint_line, endpoint_ci) = return_value
        self.assertEqual(cumulative_line, mock_cumulative_distance.return_value)
        self.assertEqual(cumulative_ci, mock_cumulative_ci.return_value)
        self.assertEqual(endpoint_line, mock_endpoint_distance.return_value)
        self.assertEqual(endpoint_ci, mock_endpoint_ci.return_value)

    @patch('test_plotter.Plotter.plot_endpoint_hypothesis_distance')
    @patch('test_plotter.Plotter.plot_cumulative_hypothesis_distance')
    def test_plot_hypothesis_ax_returns_none_values_if_boolean_flags_are_set_to_false(self, mock_cumulative_distance,
                                                                                      mock_endpoint_distance) -> None:
        self.disable_ci()
        return_value = self.plotter.plot_hypothesis_ax(ax=self.ax, xlims=(0, 5))
        (cumulative_line, cumulative_ci), (endpoint_line, endpoint_ci) = return_value
        self.assertEqual(cumulative_line, mock_cumulative_distance.return_value)
        self.assertIsNone(cumulative_ci)
        self.assertEqual(endpoint_line, mock_endpoint_distance.return_value)
        self.assertIsNone(endpoint_ci)

    @patch('test_plotter.Plotter.plot_endpoint_hypothesis_ci')
    @patch('test_plotter.Plotter.plot_cumulative_hypothesis_ci')
    def test_plot_hypothesis_ax_plots_everything_if_boolean_flags_are_set_to_true(self, plot_cumulative_hypothesis_ci,
                                                                                  plot_endpoint_hypothesis_ci) -> None:
        self.mock_ax.get_ylim.return_value = (0, 1)  # Mocking standard Axes limits
        self.plotter.plot_hypothesis_ax(ax=self.mock_ax, xlims=(0, 5))
        for mock_plot_function in (plot_cumulative_hypothesis_ci, plot_endpoint_hypothesis_ci):
            with self.subTest(mock_plot_function=mock_plot_function):
                mock_plot_function.assert_called_with(ax=self.mock_ax)

    @patch('test_plotter.Plotter.plot_endpoint_hypothesis_ci')
    @patch('test_plotter.Plotter.plot_cumulative_hypothesis_ci')
    def test_plot_hypothesis_ax_does_not_plot_ci_if_boolean_flags_are_set_to_false(self, plot_cumulative_hypothesis_ci,
                                                                                   plot_endpoint_hypothesis_ci) -> None:
        self.disable_ci()
        self.mock_ax.get_ylim.return_value = (0, 1)  # Mocking standard Axes limits
        self.plotter.plot_hypothesis_ax(ax=self.mock_ax, xlims=(0, 5))
        for mock_plot_function in (plot_cumulative_hypothesis_ci, plot_endpoint_hypothesis_ci):
            with self.subTest(mock_plot_function=mock_plot_function):
                mock_plot_function.assert_not_called()

    def test_plot_hypothesis_lines_plots_red_h0_at_y0(self) -> None:
        self.plotter.plot_hypothesis_lines(ax=self.mock_ax)
        (h0_args, h0_kwargs), _ = self.mock_ax.axhline.call_args_list
        self.assertIn(0, h0_args)
        self.assertIn(self.plotter.h0_color, h0_kwargs.values())

    def test_plot_hypothesis_lines_plots_red_h1_at_y1(self) -> None:
        self.plotter.plot_hypothesis_lines(ax=self.mock_ax)
        _, (h1_args, h1_kwargs) = self.mock_ax.axhline.call_args_list
        self.assertIn(1, h1_args)
        self.assertIn(self.plotter.h1_color, h1_kwargs.values())

    def test_plot_cumulative_hypothesis_distance_plots_line_of_cumulative_distance_values(self) -> None:
        self.plotter.plot_cumulative_hypothesis_distance(ax=self.mock_ax)
        self.mock_ax.plot.assert_called_once()
        actual_args, _ = self.mock_ax.plot.call_args
        self.assertArrayIn(self.plotter.xs, actual_args)
        self.assertArrayIn(self.plotter.cumulative_ys, actual_args)

    def test_plot_cumulative_hypothesis_distance_plots_line_of_correct_color(self) -> None:
        self.plotter.plot_cumulative_hypothesis_distance(ax=self.mock_ax)
        _, actual_kwargs = self.mock_ax.plot.call_args
        self.assertIn(self.plotter.cumul_color, actual_kwargs.values())

    def test_plot_cumulative_hypothesis_returns_a_line2d_instance(self) -> None:
        expected_line2d = self.plotter.plot_cumulative_hypothesis_distance(ax=self.ax)
        self.assertIsInstance(expected_line2d, plt.Line2D)

    def test_plot_endpoint_hypothesis_distance_plots_line_of_endpoint_distance_values(self) -> None:
        self.plotter.plot_endpoint_hypothesis_distance(ax=self.mock_ax)
        self.mock_ax.plot.assert_called_once()
        actual_args, _ = self.mock_ax.plot.call_args
        self.assertArrayIn(self.plotter.xs, actual_args)
        self.assertArrayIn(self.plotter.endpoint_ys, actual_args)

    def test_plot_endpoint_hypothesis_distance_plots_line_of_correct_color(self) -> None:
        self.plotter.plot_endpoint_hypothesis_distance(ax=self.mock_ax)
        _, actual_kwargs = self.mock_ax.plot.call_args
        self.assertIn(self.plotter.endp_color, actual_kwargs.values())

    def test_plot_endpoint_hypothesis_returns_a_line2d_instance(self) -> None:
        expected_line2d = self.plotter.plot_endpoint_hypothesis_distance(ax=self.ax)
        self.assertIsInstance(expected_line2d, plt.Line2D)

    def test_set_hypothesis_plot_limits_sets_x_limits_to_argument_passed_in(self) -> None:
        xlims = (-50, 50)
        self.plotter.set_hypothesis_plot_limits(ax=self.ax, xlims=xlims)
        self.assertEqual(self.ax.get_xlim(), xlims)

    def test_set_hypothesis_plot_limits_does_not_adjust_with_y_limits_if_they_are_large_enough(self) -> None:
        ylims = (-50, 50)
        self.ax.set_ylim(ylims)
        self.plotter.set_hypothesis_plot_limits(ax=self.ax, xlims=(0, 5))
        self.assertEqual(self.ax.get_ylim(), ylims)

    def test_set_hypothesis_plot_limits_adjusts_with_y_limits_if_they_are_not_large_enough(self) -> None:
        ylims = (-0.1, 0.1)
        self.plotter.set_hypothesis_plot_limits(ax=self.ax, xlims=(0, 5))
        self.assertNotEqual(self.ax.get_ylim(), ylims)
        self.assertEqual(self.ax.get_ylim(), (self.plotter.hypothesis_plot_lower_ylim,
                                              self.plotter.hypothesis_plot_upper_ylim))

    def test_plot_cumulative_hypothesis_ci_fills_an_area(self) -> None:
        self.plotter.plot_cumulative_hypothesis_ci(ax=self.mock_ax)
        self.mock_ax.fill_between.assert_called_once()
        actual_args, _ = self.mock_ax.fill_between.call_args
        self.assertArrayIn(self.plotter.xs, actual_args)
        self.assertArrayIn(self.plotter.cumulative_upper_ys, actual_args)
        self.assertArrayIn(self.plotter.cumulative_lower_ys, actual_args)

    def test_plot_cumulative_hypothesis_ci_uses_correct_color(self) -> None:
        self.plotter.plot_cumulative_hypothesis_ci(ax=self.mock_ax)
        _, actual_kwargs = self.mock_ax.fill_between.call_args
        self.assertIn(self.plotter.cumul_color, actual_kwargs.values())

    def test_plot_cumulative_hypothesis_returns_a_polycollection_instance(self) -> None:
        expected_polycollection = self.plotter.plot_cumulative_hypothesis_ci(ax=self.ax)
        self.assertIsInstance(expected_polycollection, PolyCollection)

    def test_plot_endpoint_hypothesis_ci_fills_an_area(self) -> None:
        self.plotter.plot_endpoint_hypothesis_ci(ax=self.mock_ax)
        self.mock_ax.fill_between.assert_called_once()
        actual_args, _ = self.mock_ax.fill_between.call_args
        self.assertArrayIn(self.plotter.xs, actual_args)
        self.assertArrayIn(self.plotter.endpoint_upper_ys, actual_args)
        self.assertArrayIn(self.plotter.endpoint_lower_ys, actual_args)

    def test_plot_endpoint_hypothesis_ci_uses_correct_color(self) -> None:
        self.plotter.plot_endpoint_hypothesis_ci(ax=self.mock_ax)
        _, actual_kwargs = self.mock_ax.fill_between.call_args
        self.assertIn(self.plotter.endp_color, actual_kwargs.values())

    def test_plot_endpoint_hypothesis_returns_a_polycollection_instance(self) -> None:
        expected_polycollection = self.plotter.plot_endpoint_hypothesis_ci(ax=self.ax)
        self.assertIsInstance(expected_polycollection, PolyCollection)

    def test_format_hypothesis_plot_adds_title_labels_ticks_and_set_plot_legends(self) -> None:
        self.assertFalse(self.ax.get_title())
        self.assertFalse(self.ax.get_xlabel())
        self.assertFalse(self.ax.get_ylabel())
        self.ax.legend = MagicMock()
        self.plotter.format_hypothesis_plot(ax=self.ax)
        self.ax.legend.assert_called_once()
        self.assertTrue(self.ax.get_title())
        self.assertTrue(self.ax.get_xlabel())
        self.assertTrue(self.ax.get_ylabel())
        for expected_label, text_object in zip(['H0', 'H1'], self.ax.get_yticklabels()):
            actual_label = text_object.get_text()
            with self.subTest(expected_label=expected_label, actual_label=actual_label):
                self.assertEqual(expected_label, actual_label)

    def test_invert_hypothesis_plot_y_axis_calls_ax_invert_yaxis(self) -> None:
        self.mock_ax.invert_yaxis.assert_not_called()
        self.plotter.invert_hypothesis_plot_y_axis(ax=self.mock_ax)
        self.mock_ax.invert_yaxis.assert_called()

    @patch('test_plotter.Plotter.plot_distributions')
    @patch('test_plotter.Plotter.plot_group_divisions')
    @patch('test_plotter.Plotter.format_histogram')
    def test_plot_histogram_ax_calls_all_histogram_related_plot_functions(self, *histogram_functions) -> None:
        self.plotter.plot_histogram_ax(ax=self.mock_ax)
        for histogram_function in histogram_functions:
            histogram_function.assert_called_with(ax=self.mock_ax)

    @patch('src.plotter.distplot')
    def test_plot_distributions_calls_seaborn_distplot(self, mock_seaborn_distplot) -> None:
        self.plotter.plot_distributions(ax=self.mock_ax)
        actual_args, actual_kwargs = mock_seaborn_distplot.call_args
        self.assertArrayIn(self.plotter.hist_xs, actual_args)
        self.assertArrayIn(self.plotter.hist_intervals, actual_kwargs.values())
        self.assertIn(self.mock_ax, actual_kwargs.values())

    def test_plot_group_divisions_adds_vertical_lines_based_on_breakpoints(self) -> None:
        self.plotter.plot_group_divisions(ax=self.mock_ax)
        actual_args, _ = self.mock_ax.vlines.call_args
        np.testing.assert_allclose(self.plotter.hist_intervals, actual_args[0])

    def test_plot_group_divisions_adds_vertical_lines_of_correct_colors(self) -> None:
        self.plotter.plot_group_divisions(ax=self.mock_ax)
        for interval, (_, actual_kwargs) in zip(self.plotter.hist_intervals, self.mock_ax.vlines.call_args_list):
            with self.subTest(interval=interval, actual_kwargs=actual_kwargs):
                self.assertIn(self.plotter.hist_interval_color, actual_kwargs.values())

    def test_format_histogram_modifies_title_and_xy_labels(self) -> None:
        self.assertFalse(self.ax.get_title())
        self.assertFalse(self.ax.get_xlabel())
        self.assertFalse(self.ax.get_ylabel())
        self.plotter.format_histogram(ax=self.ax)
        self.assertTrue(self.ax.get_title())
        self.assertTrue(self.ax.get_xlabel())
        self.assertTrue(self.ax.get_ylabel())


if __name__ == '__main__':
    unittest.main()
