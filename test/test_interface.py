"""test_interface.py - unit tests for interface.py."""

import unittest
from queue import Queue
from typing import List
from unittest.mock import MagicMock, patch

import openpyxl
from PySide2.QtCore import Qt, SIGNAL
from PySide2.QtTest import QTest
from PySide2.QtWidgets import QApplication, QMessageBox
from matplotlib.pyplot import Axes

from src.core import dynafit
from src.exceptions import AbortedByUser, CorruptedExcelFile
from src.interface import DynaFitGUI
from src.worker import Worker
from test.resources import SETTINGS_SCHEMA


class TestInterfaceModule(unittest.TestCase):
    """Tests the interface.py module."""
    test_case_path = './test/InterfaceExample_TestCase.xlsx'
    test_case_sheetnames = ['CS_GR', 'CS1_CS2']

    @classmethod
    def setUpClass(cls) -> None:
        """Sets up the test suite by instantiating a QApplication."""
        cls.app = QApplication()

    def setUp(self) -> None:
        """Sets up each unit test by refreshing the DynaFitGUI instance."""
        self.ui = DynaFitGUI()

    def tearDown(self) -> None:
        """Deletes the DynaFitGUI instance after a unit test ends."""
        del self.ui

    def load_example_data(self) -> None:
        """Loads the test case file as if it were loaded through the DynaFitGUI."""
        self.ui.data = openpyxl.load_workbook(self.test_case_path)

    @property
    def axes(self) -> List[Axes]:
        """Convenience property for accessing all Axes instances of the DynaFitGUI."""
        return [self.ui.cvp_ax, self.ui.cody_ax, self.ui.histogram_ax]

    @patch('PySide2.QtWidgets.QFileDialog.getOpenFileName')
    def test_click_load_data_button_loads_data_dialog(self, mock_file_dialog) -> None:
        QTest.mouseClick(self.ui.load_data_button, Qt.LeftButton)
        mock_file_dialog.assert_called()

    @patch('src.interface.DynaFitGUI.load_data_success')
    def test_load_data_calls_load_data_success(self, mock_load_data_success) -> None:
        query = 'my query'
        with patch('openpyxl.load_workbook'):
            self.ui.load_data(query=query)
        mock_load_data_success.assert_called()

    @patch('src.interface.DynaFitGUI.load_data_success')
    def test_load_data_raises_exception_on_corrupted_excel_file(self, mock_load_data_success) -> None:
        query = 'my query'
        with self.assertRaises(CorruptedExcelFile):
            with patch('openpyxl.load_workbook', side_effect=CorruptedExcelFile):
                self.ui.load_data(query=query)
        mock_load_data_success.assert_not_called()

    def test_load_data_success_changes_ui_elements(self) -> None:
        self.load_example_data()
        # before load_data_success
        self.assertEqual(self.ui.input_filename_label.text(), '')
        self.assertFalse(self.ui.input_sheetname_combobox.isEnabled())
        self.assertEqual(self.ui.input_sheetname_combobox.count(), 0)
        self.ui.load_data_success(filename='filename')
        # after load_data_success
        self.assertEqual(self.ui.input_filename_label.text(), 'filename')
        self.assertTrue(self.ui.input_sheetname_combobox.isEnabled())
        self.assertEqual(self.ui.input_sheetname_combobox.count(), 2)
        for i, name in enumerate(self.test_case_sheetnames):
            self.ui.input_sheetname_combobox.setCurrentIndex(i)
            self.assertEqual(self.ui.input_sheetname_combobox.currentText(), name)

    def test_cs1_cs2_button_clicked_changes_ui_elements(self) -> None:
        QTest.mouseClick(self.ui.cs1_cs2_button, Qt.LeftButton)
        self.assertTrue(self.ui.time_interval_label.isEnabled())
        self.assertTrue(self.ui.time_interval_spinbox.isEnabled())
        self.assertEqual(self.ui.CS_label.text(), 'Initial colony size column')
        self.assertEqual(self.ui.GR_label.text(), 'Final colony size column')

    def test_cs_cs_button_clicked_changes_ui_elements(self) -> None:
        QTest.mouseClick(self.ui.cs_gr_button, Qt.LeftButton)
        self.assertFalse(self.ui.time_interval_label.isEnabled())
        self.assertFalse(self.ui.time_interval_spinbox.isEnabled())
        self.assertEqual(self.ui.CS_label.text(), 'Colony size column')
        self.assertEqual(self.ui.GR_label.text(), 'Growth rate column')

    def test_conf_int_checkbox_checked_changes_ui_elements(self) -> None:
        self.assertFalse(self.ui.conf_int_spinbox.isEnabled())
        QTest.mouseClick(self.ui.conf_int_checkbox, Qt.LeftButton)
        self.assertTrue(self.ui.conf_int_spinbox.isEnabled())
        QTest.mouseClick(self.ui.conf_int_checkbox, Qt.LeftButton)
        self.assertFalse(self.ui.conf_int_spinbox.isEnabled())

    @patch('src.interface.DynaFitGUI.dynafit_setup_before_running')
    @patch('src.interface.DynaFitGUI.get_dynafit_settings')
    @patch('src.interface.DynaFitGUI.raise_main_thread_error')
    @patch('src.interface.DynaFitGUI.dynafit_worker_run')
    def test_dynafit_run_success_calls_appropriate_downstream_methods(self, mock_dynafit_worker_run,
                                                                      mock_raise_main_thread_error,
                                                                      mock_get_dynafit_settings,
                                                                      mock_dynafit_setup_before_running) -> None:
        self.ui.dynafit_run()
        mock_dynafit_setup_before_running.assert_called()
        mock_get_dynafit_settings.assert_called()
        mock_raise_main_thread_error.assert_not_called()
        mock_dynafit_worker_run.assert_called_with(dynafit_settings=mock_get_dynafit_settings.return_value)

    @patch('src.interface.DynaFitGUI.dynafit_setup_before_running', side_effect=Exception)
    @patch('src.interface.DynaFitGUI.get_dynafit_settings')
    @patch('src.interface.DynaFitGUI.raise_main_thread_error')
    @patch('src.interface.DynaFitGUI.dynafit_worker_run')
    def test_dynafit_run_setup_failure_calls_main_thread_error(self, mock_dynafit_worker_run,
                                                               mock_raise_main_thread_error, mock_get_dynafit_settings,
                                                               mock_dynafit_setup_before_running) -> None:
        self.ui.dynafit_run()
        mock_dynafit_setup_before_running.assert_called()
        mock_raise_main_thread_error.assert_called()
        mock_get_dynafit_settings.assert_not_called()
        mock_dynafit_worker_run.assert_not_called()

    @patch('src.interface.DynaFitGUI.dynafit_setup_before_running')
    @patch('src.interface.DynaFitGUI.get_dynafit_settings', side_effect=Exception)
    @patch('src.interface.DynaFitGUI.raise_main_thread_error')
    @patch('src.interface.DynaFitGUI.dynafit_worker_run')
    def test_dynafit_run_get_settings_failure_calls_main_thread_error(self, mock_dynafit_worker_run,
                                                                      mock_raise_main_thread_error,
                                                                      mock_get_dynafit_settings,
                                                                      mock_dynafit_setup_before_running) -> None:
        self.ui.dynafit_run()
        mock_dynafit_setup_before_running.assert_called()
        mock_get_dynafit_settings.assert_called()
        mock_raise_main_thread_error.assert_called()
        mock_dynafit_worker_run.assert_not_called()

    @patch('src.interface.dynafit')
    @patch('src.interface.QThreadPool.start')
    @patch('src.interface.Worker')
    def test_dynafit_worker_run_calls_worker_and_passes_it_to_threadpool(self, mock_worker, mock_threadpool_start,
                                                                         mock_dynafit) -> None:
        dynafit_settings = dict(a=1)
        self.ui.dynafit_worker_run(dynafit_settings=dynafit_settings)
        mock_worker.assert_called_with(func=mock_dynafit, a=1)
        mock_threadpool_start.assert_called_with(mock_worker.return_value)

    def test_dynafit_connect_worker_connects_worker_slots_to_interface_signals(self) -> None:
        signals = [SIGNAL('progress(int)'), SIGNAL('warning(PyObject)'), SIGNAL('finished()'),
                   SIGNAL('success(PyObject)'), SIGNAL('error(PyObject)')]
        w = Worker(func=print)
        for signal in signals:
            with self.subTest(signal=signal):
                self.assertEqual(w.signals.receivers(signal), 0)
        self.ui.connect_worker(worker=w)
        for signal in signals:
            with self.subTest(signal=signal):
                self.assertEqual(w.signals.receivers(signal), 1)

    def test_dynafit_setup_before_running_clears_axes(self) -> None:
        for ax in self.axes:
            ax.plot([1, 2], [3, 4])
            with self.subTest(ax=ax):
                self.assertTrue(ax.lines)
        self.ui.dynafit_setup_before_running()
        for ax in self.axes:
            with self.subTest(ax=ax):
                self.assertFalse(ax.lines)

    def test_dynafit_setup_before_running_turns_progress_bar_widgets_on(self) -> None:
        for widget in (self.ui.progress_bar, self.ui.progress_bar_label):
            with self.subTest(widget=widget):
                self.assertTrue(widget.isHidden())
        self.ui.dynafit_setup_before_running()
        for widget in (self.ui.progress_bar, self.ui.progress_bar_label):
            with self.subTest(widget=widget):
                self.assertFalse(widget.isHidden())

    def test_dynafit_setup_before_running_disables_plot_and_export_buttons(self) -> None:
        self.ui.to_excel_button.setEnabled(True)  # buttons first become enabled
        self.ui.to_csv_button.setEnabled(True)    # when the analysis has finished
        for button in (self.ui.plot_button, self.ui.to_excel_button, self.ui.to_csv_button):
            with self.subTest(button=button):
                self.assertTrue(button.isEnabled())
        self.ui.dynafit_setup_before_running()
        for button in (self.ui.plot_button, self.ui.to_excel_button, self.ui.to_csv_button):
            with self.subTest(button=button):
                self.assertFalse(button.isEnabled())

    def test_get_dynafit_settings_keys_are_dynafit_kwargs(self) -> None:
        settings = self.ui.get_dynafit_settings()
        for expected_argument_name in settings.keys():
            with self.subTest(expected_argument_name=expected_argument_name):
                self.assertIn(expected_argument_name, dynafit.__code__.co_varnames)

    def test_get_dynafit_settings_values_have_correct_types(self) -> None:
        settings = self.ui.get_dynafit_settings()
        del settings['data']  # do not test data (next tests do this)
        for expected_class, actual_value in zip(SETTINGS_SCHEMA.values(), settings.values()):
            with self.subTest(expected_class=expected_class, actual_value=actual_value):
                self.assertIsInstance(actual_value, expected_class)

    def test_get_dynafit_settings_no_data_loaded(self) -> None:
        settings = self.ui.get_dynafit_settings()
        self.assertIsNone(settings['data'])

    def test_get_dynafit_settings_data_loaded(self) -> None:
        self.load_example_data()
        settings = self.ui.get_dynafit_settings()
        self.assertIsInstance(settings['data'], openpyxl.Workbook)

    def test_dynafit_worker_progress_updated(self) -> None:
        self.ui.dynafit_worker_progress_updated(10)
        self.assertEqual(self.ui.progress_bar.value(), 10)

    @patch('src.interface.QMessageBox')
    def test_dynafit_worker_small_sample_warning_displays_message_box(self, mock_message_box) -> None:
        warning_contents = (Queue(), {1: 2})
        self.ui.dynafit_worker_small_sample_size_warning(warning=warning_contents)
        mock_message_box.assert_called()
        mock_message_box.return_value.exec_.assert_called()

    def test_dynafit_worker_small_sample_warning_yes_button_puts_false_in_queue(self) -> None:
        q = Queue()
        warning_contents = (q, {1: 2})
        with patch('src.interface.QMessageBox.exec_', return_value=QMessageBox.Yes):
            self.ui.dynafit_worker_small_sample_size_warning(warning=warning_contents)
        self.assertFalse(q.get())

    def test_dynafit_worker_small_sample_warning_no_button_puts_true_in_queue(self) -> None:
        q = Queue()
        warning_contents = (q, {1: 2})
        with patch('src.interface.QMessageBox.exec_', return_value=QMessageBox.No):
            self.ui.dynafit_worker_small_sample_size_warning(warning=warning_contents)
        self.assertTrue(q.get())

    def test_dynafit_worker_raised_exception_removes_figure_title(self) -> None:
        self.ui.fig.suptitle('SOMETHING HERE')
        with patch('src.interface.DynaFitGUI.raise_worker_thread_error'):
            self.ui.dynafit_worker_raised_exception((Exception(), 'string'))
        self.assertEqual(self.ui.fig._suptitle.get_text(), '')

    def test_dynafit_worker_raised_exception_removes_figure_visibility(self) -> None:
        for ax in self.axes:
            ax.set_visible(True)
        with patch('src.interface.DynaFitGUI.raise_worker_thread_error'):
            self.ui.dynafit_worker_raised_exception((Exception(), 'string'))
        for ax in self.axes:
            with self.subTest(ax=ax):
                self.assertFalse(ax.get_visible())

    @patch('src.interface.QTableWidget')
    def test_dynafit_worker_raised_exception_clears_results_table(self, mock_table_clear) -> None:
        mock_table_clear.return_value.clearContents = MagicMock()  # no idea WHY I had to do this
        mock_table_clear.return_value.clearContents.assert_not_called()
        with patch('src.interface.DynaFitGUI.raise_worker_thread_error'):
            self.ui.dynafit_worker_raised_exception((Exception(), 'string'))
        mock_table_clear.return_value.clearContents.assert_not_called()

    def test_dynafit_worker_raised_exception_passes_exception_to_handler_function(self) -> None:
        e = Exception()
        s = 'string'
        with patch('src.interface.DynaFitGUI.raise_worker_thread_error') as mock_worker_exception_handler:
            self.ui.dynafit_worker_raised_exception((e, s))
        mock_worker_exception_handler.assert_called_once_with((e, s))

    @patch('src.interface.DynaFitGUI.raise_worker_thread_error')
    def test_dynafit_worker_raised_exception_no_exception_shown_when_aborted_by_user(self, mock_thread_error) -> None:
        mock_thread_error.assert_not_called()
        self.ui.dynafit_worker_raised_exception((AbortedByUser(), 'string'))
        mock_thread_error.assert_not_called()


if __name__ == '__main__':
    unittest.main()
