import unittest
from unittest.mock import patch

import openpyxl
from PySide2.QtCore import Qt
from PySide2.QtTest import QTest
from PySide2.QtWidgets import QApplication

from src.exceptions import CorruptedExcelFile
from src.interface import DynaFitGUI


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
    def test_click_load_data_button_loads_data_dialog(self, mock_file_dialog):
        QTest.mouseClick(self.ui.load_data_button, Qt.LeftButton)
        mock_file_dialog.assert_called()

    @patch('src.interface.DynaFitGUI.load_data_success')
    def test_load_data_calls_load_data_success(self, mock_load_data_success):
        query = 'my query'
        with patch('openpyxl.load_workbook'):
            self.ui.load_data(query=query)
        mock_load_data_success.assert_called()

    @patch('src.interface.DynaFitGUI.load_data_success')
    def test_load_data_raises_exception_on_corrupted_excel_file(self, mock_load_data_success):
        query = 'my query'
        with self.assertRaises(CorruptedExcelFile):
            with patch('openpyxl.load_workbook', side_effect=CorruptedExcelFile):
                self.ui.load_data(query=query)
        mock_load_data_success.assert_not_called()

    def test_load_data_success_changes_ui_elements(self):
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

    def test_cs1_cs2_button_clicked_changes_ui_elements(self):
        QTest.mouseClick(self.ui.cs1_cs2_button, Qt.LeftButton)
        self.assertTrue(self.ui.time_interval_label.isEnabled())
        self.assertTrue(self.ui.time_interval_spinbox.isEnabled())
        self.assertEqual(self.ui.CS_label.text(), 'Initial colony size column')
        self.assertEqual(self.ui.GR_label.text(), 'Final colony size column')

    def test_cs_cs_button_clicked_changes_ui_elements(self):
        QTest.mouseClick(self.ui.cs_gr_button, Qt.LeftButton)
        self.assertFalse(self.ui.time_interval_label.isEnabled())
        self.assertFalse(self.ui.time_interval_spinbox.isEnabled())
        self.assertEqual(self.ui.CS_label.text(), 'Colony size column')
        self.assertEqual(self.ui.GR_label.text(), 'Growth rate column')

    def test_conf_int_checkbox_checked_changes_ui_elements(self):
        self.assertFalse(self.ui.conf_int_spinbox.isEnabled())
        QTest.mouseClick(self.ui.conf_int_checkbox, Qt.LeftButton)
        self.assertTrue(self.ui.conf_int_spinbox.isEnabled())
        QTest.mouseClick(self.ui.conf_int_checkbox, Qt.LeftButton)
        self.assertFalse(self.ui.conf_int_spinbox.isEnabled())

    @patch('src.interface.DynaFitGUI.dynafit_setup_before_running')
    @patch('src.interface.DynaFitGUI.get_dynafit_settings')
    @patch('src.interface.DynaFitGUI.dynafit_worker_run')
    def test_dynafit_run_success_calls_appropriate_downstream_methods(self, mock_dynafit_worker_run,
                                                                      mock_get_dynafit_settings,
                                                                      mock_dynafit_setup_before_running):
        self.ui.dynafit_run()
        mock_dynafit_setup_before_running.assert_called()
        mock_get_dynafit_settings.assert_called()
        mock_dynafit_worker_run.assert_called_with(dynafit_settings=mock_get_dynafit_settings.return_value)


if __name__ == '__main__':
    unittest.main()
