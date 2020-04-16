import unittest
from unittest.mock import patch, MagicMock

import openpyxl
from PySide2.QtCore import Qt, SIGNAL
from PySide2.QtTest import QTest
from PySide2.QtWidgets import QApplication

from src.exceptions import CorruptedExcelFile
from src.interface import DynaFitGUI
from src.worker import Worker


class TestInterfaceModule(unittest.TestCase):
    """Tests the interface.py module."""
    test_case_path = './test/InterfaceExample_TestCase.xlsx'
    test_case_sheetnames = ['CS_GR', 'CS1_CS2']

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication()

    def setUp(self) -> None:
        self.ui = DynaFitGUI()

    def tearDown(self) -> None:
        del self.ui

    def load_example_data(self) -> None:
        self.ui.data = openpyxl.load_workbook(self.test_case_path)

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

    def test_dynafit_setup_before_running_clears_axes(self):
        self.ui.cvp_ax.plot([1, 2], [3, 4])
        self.ui.histogram_ax.plot([1, 2], [3, 4])
        self.assertTrue(self.ui.cvp_ax.lines)
        self.assertTrue(self.ui.histogram_ax.lines)
        self.ui.dynafit_setup_before_running()
        self.assertFalse(self.ui.cvp_ax.lines)
        self.assertFalse(self.ui.histogram_ax.lines)

    def test_dynafit_setup_before_running_sets_histogram_axes_off(self):
        self.ui.histogram_ax = MagicMock()
        self.ui.histogram_ax.set_axis_off.assert_not_called()
        self.ui.dynafit_setup_before_running()
        self.ui.histogram_ax.set_axis_off.assert_called()

    def test_dynafit_setup_before_running_turns_progress_bar_widgets_on(self):
        for widget in (self.ui.progress_bar, self.ui.progress_bar_label):
            with self.subTest(widget=widget):
                self.assertTrue(widget.isHidden())
        self.ui.dynafit_setup_before_running()
        for widget in (self.ui.progress_bar, self.ui.progress_bar_label):
            with self.subTest(widget=widget):
                self.assertFalse(widget.isHidden())

    def test_dynafit_setup_before_running_disables_plot_and_export_buttons(self):
        self.ui.to_excel_button.setEnabled(True)  # buttons first become enabled
        self.ui.to_csv_button.setEnabled(True)    # when the analysis has finished
        for button in (self.ui.plot_button, self.ui.to_excel_button, self.ui.to_csv_button):
            with self.subTest(button=button):
                self.assertTrue(button.isEnabled())
        self.ui.dynafit_setup_before_running()
        for button in (self.ui.plot_button, self.ui.to_excel_button, self.ui.to_csv_button):
            with self.subTest(button=button):
                self.assertFalse(button.isEnabled())

    def test_get_dynafit_settings(self):
        settings = self.ui.get_dynafit_settings()


if __name__ == '__main__':
    unittest.main()
