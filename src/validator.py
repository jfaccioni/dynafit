import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
from openpyxl import Workbook
from openpyxl.cell.cell import Cell


class ExcelValidator:
    """Class that bundles many different validation methods for the input data (Excel spreadsheet) used for DynaFit.
    The validation methods aim to raise a specific, user-informative exception, if they fail."""
    def __init__(self, data: Optional[Workbook], sheetname: str, cs_start_cell: str, cs_end_cell: str,
                 gr_start_cell: str, gr_end_cell: str) -> None:
        """Init method of ExcelValidator class"""
        # Checks if any data has been loaded at all
        self.validate_input(data=data)
        self.ws = data[sheetname]
        # Structures cell ranges in a dictionary
        self.ranges = {
            'CS': [cs_start_cell.strip().upper(), cs_end_cell.strip().upper()],
            'GR': [gr_start_cell.strip().upper(), gr_end_cell.strip().upper()],
        }
        # Validates data as a whole
        self.data = self.validation_routine()

    @property
    def max_row(self) -> int:
        """Convenience method that returns the max row of the input Excel spreadsheet"""
        return self.ws.max_row

    @staticmethod
    def validate_input(data: Optional[Workbook]) -> None:
        """Validates input by checking whether an input Excel file has been selected at all"""
        if data is None:
            raise NoExcelFile('Please select an Excel spreadsheet as the input file')

    def validation_routine(self):
        """Method responsible for calling downstream validation methods. At the end (if no Exceptions were raised),
        this method returns a pandas DataFrame of the original data, in the format:
          CS: <colony size values>,
          GR: <growth rate values>
        Validation includes:
          - """  # TODO: finish this (start here)
        data = {}  # To be converted to a pandas DataFrame at the end
        for name, (start, end) in self.ranges.items():
            if end:  # User wants range from start cell to end cell
                self.validate_cell_range(start=start, end=end)
            else:  # User wants range from start cell to end of the start cell's column
                self.validate_cell_string(cell_str=start)
                end = self.get_end_cell(start=start)  # Figures out where the column ends
            cell_range = f'{start}:{end}'
            cells = self.ws[cell_range]
            values = self.get_cell_values(cells)
            data[name] = values
        self.validate_values_size(data=data)
        return pd.DataFrame(data)

    def validate_cell_range(self, start: str, end: str) -> None:
        """Validates range of cells (from start cell to end cell) in an Excel spreadsheet"""
        for cell_str in (start, end):
            self.validate_cell_string(cell_str=cell_str)
        if not self.ranges_share_same_column(start=start, end=end):
            raise MismatchedColumnsError(f'Cells {start} and {end} do not share same column')
        if not self.end_cell_comes_after_start_cell(start=start, end=end):
            raise MismatchedRowsError(f'Start cell {start} comes after end cell {end}')

    def validate_cell_string(self, cell_str: str) -> None:
        """Validates a cell string in an Excel spreadsheet"""
        if not self.is_valid_excel(cell_str=cell_str):
            if cell_str == '':  # Guaranteed only to happen on start cells
                raise BadCharError(f'Start cell cannot be empty')
            raise BadCharError(f'The string "{cell_str}" is not valid Excel cell accessor')

    @staticmethod
    def is_valid_excel(cell_str: str) -> bool:
        """Returns whether the string cell_str is a valid Excel cell accessor"""
        #  cell_str must be non-empty, and exclusively composed of letters and numbers
        if not cell_str.isalnum():
            return False
        # All letters in cell_str come before all numbers. Source:
        # https://stackoverflow.com/questions/60758670/check-if-a-python-string-is-a-valid-excel-cell
        return bool(re.match("[A-Z]+\d+$", cell_str))

    def ranges_share_same_column(self, start: str, end: str) -> bool:
        """Returns whether the start and end cells share the same column letter"""
        start_letters = self.extract_letters(start)
        end_letters = self.extract_letters(end)
        return start_letters == end_letters

    def end_cell_comes_after_start_cell(self, start: str, end: str) -> bool:
        """Returns whether the row number of the end cell comes after the row number of the start cell"""
        start_numbers = self.extract_digits(start)
        end_numbers = self.extract_digits(end)
        return int(start_numbers) < int(end_numbers)

    @staticmethod
    def get_cell_values(cells: Tuple[Tuple[Cell]]) -> List[float]:
        """docstring"""
        return [float(c[0].value) for c in cells if c[0].value not in (None, '')]

    def get_end_cell(self, start: str):
        letter = self.extract_letters(start)
        number = self.max_row
        return f'{letter}{number}'

    @staticmethod
    def extract_letters(s):
        return ''.join(char for char in s if char.isalpha())

    @staticmethod
    def extract_digits(s):
        return ''.join(char for char in s if char.isdigit())

    @staticmethod
    def validate_values_size(data: Dict[str, List[float]]) -> None:
        """Checks whether the data has same size"""
        l1, l2 = [len(vs) for vs in data.values()]
        if l1 != l2:
            raise DifferentSizeError('CS and GR cell ranges have different lengths (after removing blank/empty cells)')


class NoExcelFile(Exception):
    """Exception raised when user runs DynaFit with no input file."""
    def __init__(self, *args):
        super().__init__(*args)


class BadCharError(Exception):
    """Exception raised when samples cannot be found in the input file."""
    def __init__(self, *args):
        super().__init__(*args)


class MismatchedColumnsError(Exception):
    """Exception raised when samples cannot be found in the input file."""
    def __init__(self, *args):
        super().__init__(*args)


class MismatchedRowsError(Exception):
    """Exception raised when samples cannot be found in the input file."""
    def __init__(self, *args):
        super().__init__(*args)


class DifferentSizeError(Exception):
    """Exception raised when samples cannot be found in the input file."""
    def __init__(self, *args):
        super().__init__(*args)
