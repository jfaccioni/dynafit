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
        # Add character '1' column strings (typing 'A' implies starting from cell '1A'
        cs_start_cell = self.convert_column_to_first_cell(cs_start_cell)
        gr_start_cell = self.convert_column_to_first_cell(gr_start_cell)
        # Structures cell ranges in a dictionary
        self.ranges = {
            'CS': [cs_start_cell.strip().upper(), cs_end_cell.strip().upper()],
            'GR': [gr_start_cell.strip().upper(), gr_end_cell.strip().upper()],
        }
        # Validates data as a whole
        self.data = self.validation_routine()

    def convert_column_to_first_cell(self, cell_str: str) -> str:
        """Converts and returns an Excel column accessor such as "A" to a first row cell accessor ("A1").
        Returns the cell_string itself if it is not a column accessor."""
        if cell_str != '':
            if len(self.extract_digits(cell_str)) == 0:
                return cell_str + '1'
        return cell_str

    @staticmethod
    def validate_input(data: Optional[Workbook]) -> None:
        """Validates input by checking whether an input Excel file has been selected at all"""
        if data is None:
            raise NoExcelFileError('Please select an Excel spreadsheet as the input file')

    def validation_routine(self):
        """Method responsible for calling downstream validation methods. Validation steps includes:
        - Checking if the start cell is empty
        - Checking if the start and end cells are valid Excel cells
        - Checking if the start and end cells belong to the same column
        - Checking if the start cell row comes before the end cell row
        - Checking if the values between start and end cells can be converted to numbers (float),
          ignoring None values and empty strings
        - Checking if the ranges of values for CS and GR have the same size,
          after removal of None values and empty strings

        If no Exceptions were raised during validation, this method returns a pandas DataFrame in the format:
          CS: <colony size values>,
          GR: <growth rate values>"""
        data = {}  # To be converted to a pandas DataFrame at the end
        for name, (start, end) in self.ranges.items():
            if end:  # User wants range from start cell to end cell
                self.validate_cell_range(start=start, end=end)
            else:  # User wants range from start cell to end of the start cell's column
                self.validate_cell_string(cell_str=start)
                end = self.get_end_cell(start=start)  # Figures out where the column ends
            cell_range = f'{start}:{end}'
            cell_rows = self.ws[cell_range]
            self.validate_cell_values(start, cell_rows)
            values = self.get_cell_values(cell_rows)
            data[name] = values
        self.validate_values_size(data=data)
        return pd.DataFrame(data)

    def validate_cell_string(self, cell_str: str) -> None:
        """Validates a cell string in an Excel spreadsheet"""
        if cell_str == '':  # Guaranteed only to happen on start cells
            raise EmptyCellError(f'Start cell cannot be empty')
        if not self.is_valid_excel(cell_str=cell_str):
            raise BadCellStringError(f'The string "{cell_str}" is not valid Excel cell accessor')

    @staticmethod
    def is_valid_excel(cell_str: str) -> bool:
        """Returns whether the string cell_str is a valid Excel cell accessor. This implies that"""
        #  cell_str must be exclusively composed of letters and numbers
        if not cell_str.isalnum():
            return False
        # All letters in cell_str come before all numbers. Source:
        # https://stackoverflow.com/questions/60758670/check-if-a-python-string-is-a-valid-excel-cell
        return bool(re.match("[A-Z]+\d+$", cell_str))

    def validate_cell_range(self, start: str, end: str) -> None:
        """Validates range of cells (from start cell to end cell) in an Excel spreadsheet"""
        for cell_str in (start, end):
            self.validate_cell_string(cell_str=cell_str)
        if not self.ranges_share_same_column(start=start, end=end):
            raise MismatchedColumnsError(f'Cells {start} and {end} do not share same column')
        if not self.end_cell_comes_after_start_cell(start=start, end=end):
            raise MismatchedRowsError(f'Start cell {start} comes after end cell {end}')

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

    def validate_cell_values(self, start: str, cell_rows: Tuple[Tuple[Cell]]) -> None:
        """Validates each cell in a cell range by checking whether the value inside each cell can be
        cast into a float (after ignoring None values and empty strings)"""
        column = self.extract_letters(start)
        num = int(self.extract_digits(start))
        for i, row in enumerate(cell_rows, start=num):
            cell = row[0]
            if cell.value not in (None, ''):
                try:
                    float(cell.value)
                except ValueError:
                    cell_str = f'"{column}{i}"'
                    raise BadCellValueError(f'Could not convert value "{cell.value}" on cell {cell_str} to numerical.')

    @staticmethod
    def get_cell_values(rows: Tuple[Tuple[Cell]]) -> List[float]:
        """Returns the values of cells in a column (tuple of tuples) casted to float, ignoring None values
        and empty strings"""
        return [float(row[0].value) for row in rows if row[0].value not in (None, '')]

    def get_end_cell(self, start: str) -> str:
        """Given a valid cell string, returns the cell string at the end of the column in an Excel spreadsheet.
        Max row if looked up on the currently loaded worksheet"""
        letter = self.extract_letters(start)
        number = self.ws.max_row
        return f'{letter}{number}'

    @staticmethod
    def extract_letters(s: str) -> str:
        """Returns the letter portion of an alphanumerical string s"""
        return ''.join(char for char in s if char.isalpha())

    @staticmethod
    def extract_digits(s: str) -> str:
        """Returns the digit portion of an alphanumerical string s"""
        return ''.join(char for char in s if char.isdigit())

    @staticmethod
    def validate_values_size(data: Dict[str, List[float]]) -> None:
        """Checks whether the data has same size on both of its columns (CS and GR)"""
        l1, l2 = [len(vs) for vs in data.values()]
        if l1 != l2:
            raise DifferentSizeError('CS and GR cell ranges have different lengths (after removing blank/empty cells)')


class NoExcelFileError(Exception):
    """Exception raised when user runs DynaFit with no input file."""
    def __init__(self, *args):
        super().__init__(*args)


class EmptyCellError(Exception):
    """Exception raised when mandatory cell is empty."""
    def __init__(self, *args):
        super().__init__(*args)


class BadCellStringError(Exception):
    """Exception raised when cell does not correspond to a valid Excel cell accessor."""
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


class BadCellValueError(Exception):
    """Exception raised when a cell value cannot be coerced into a float."""
    def __init__(self, *args):
        super().__init__(*args)


class DifferentSizeError(Exception):
    """Exception raised when samples cannot be found in the input file."""
    def __init__(self, *args):
        super().__init__(*args)
