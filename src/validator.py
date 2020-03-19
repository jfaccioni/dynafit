import re
from typing import Dict, List, Tuple

import pandas as pd
from openpyxl import Workbook
from openpyxl.cell.cell import Cell


class ExcelValidator:
    def __init__(self, data: Workbook, sheetname: str, cs_start_cell: str, cs_end_cell: str, gr_start_cell: str,
                 gr_end_cell: str) -> None:
        """init"""
        self.ws = data[sheetname]
        self.ranges = {
            'CS': [cs_start_cell.strip().upper(), cs_end_cell.strip().upper()],
            'GR': [gr_start_cell.strip().upper(), gr_end_cell.strip().upper()],
        }
        self.data = self.validate()

    @property
    def max_row(self):
        """docstring"""
        return self.ws.max_row

    def validate(self):
        data = {}
        for name, (start, end) in self.ranges.items():
            if end:  # user wants range from start cell to end cell
                self.validate_range(start=start, end=end)
                cell_range = f'{start}:{end}'
            else:  # user wants range from start cell to end of column
                self.validate_column(start=start)
                cell_range = f'{start}:{self.get_end_cell(start=start)}'
            data[name] = self.get_cell_values(self.ws[cell_range])
        self.validate_data(data=data)
        return pd.DataFrame(data)

    def validate_range(self, start: str, end: str) -> None:
        """Validates range values"""
        for cell_str in (start, end):
            if not self.is_valid_excel(cell_str=cell_str):
                if cell_str == '':
                    raise BadCharError(f'An empty string is not valid as the starting cell')
                raise BadCharError(f'The string "{cell_str}" is not valid Excel cell accessor')
        if not self.ranges_share_same_column(start=start, end=end):
            raise MismatchedColumnsError(f'Cells {start} and {end} do not share same column')
        if not self.end_cell_comes_after_start_cell(start=start, end=end):
            raise MismatchedRowsError(f'Starting cell {start} comes after end cell {end}')

    def validate_column(self, start: str) -> None:
        """Validates column values"""
        if not self.is_valid_excel(cell_str=start):
            if start == '':
                raise BadCharError(f'An empty string is not valid as the starting cell')
            raise BadCharError(f'The string "{start}" is not valid Excel cell accessor')

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
    def validate_data(data: Dict[str, List[float]]) -> None:
        """Checks whether the data has same size"""
        l1, l2 = [len(vs) for vs in data.values()]
        if l1 != l2:
            raise DifferentSizeError('CS and GR cell ranges have different lengths (after removing blank/empty cells)')


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
