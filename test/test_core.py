"""test_core.py - unit tests for core.py."""

import unittest
import openpyxl
from src.core import *
from src.validator import ExcelValidator


class TestCoreModule(unittest.TestCase):
    """Tests the core.py module."""

    def setUp(self) -> None:
        """Sets up each unit test by refreshing the ExcelValidator object."""
        workbook = self.load_test_case()
        self.ev = ExcelValidator(workbook=workbook, sheetname='sheetname', cs_start_cell='A1', cs_end_cell='',
                                 gr_start_cell='B1', gr_end_cell='')

    @staticmethod
    def load_test_case() -> Workbook:
        """Returns the test case Workbook."""
        test_case_path = 'test/test_cases/excel_validator_test_case.xlsx'
        return openpyxl.load_workbook(test_case_path)

    def test_something(self) -> None:
        print(self.ev.get_data())


if __name__ == '__main__':
    unittest.main()
