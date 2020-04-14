import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.worksheet.worksheet import Worksheet
from openpyxl.cell.cell import Cell

from exceptions import (BadCellStringError, DifferentSizeError, EmptyCellError, MismatchedColumnsError,
                        MismatchedRowsError, NoExcelFileError)


class ExcelValidator:
    """Class that bundles many different validation methods for the input data (Excel spreadsheet) used for DynaFit.
    The validation methods aim to raise a specific, user-informative exception, if they fail."""
    def __init__(self, workbook: Optional[Workbook], sheetname: str, cs_start_cell: str, cs_end_cell: str,
                 gr_start_cell: str, gr_end_cell: str) -> None:
        """Init method of ExcelValidator class."""
        self.wb = workbook
        self.sheetname = sheetname
        # Add character '1' column strings (typing 'A' implies starting from cell '1A'
        cs_start_cell = self.convert_column_to_first_cell(cs_start_cell.strip().upper())
        gr_start_cell = self.convert_column_to_first_cell(gr_start_cell.strip().upper())
        # Structures cell ranges in a dictionary
        self.ranges = {
            'CS': [cs_start_cell, cs_end_cell],
            'GR': [gr_start_cell, gr_end_cell],
        }

    @property
    def ws(self) -> Worksheet:
        try:
            return self.wb[self.sheetname]
        except TypeError:  # self.wb is None
            raise NoExcelFileError('Please select an Excel spreadsheet as the input file')

    def convert_column_to_first_cell(self, cell_str: str) -> str:
        """Converts and returns an Excel column accessor such as "A" to a first row cell accessor ("A1").
        Returns the cell_string itself if it is not a column accessor."""
        if cell_str != '':
            if len(self.extract_digits(cell_str)) == 0:
                return cell_str + '1'
        return cell_str

    def validation_routine(self) -> None:
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
          GR: <growth rate values>
        As an implementation detail, the GR column may actually be the final colony size column, if the user
        chose to let the program calculate the growth rate instead. This will be later overwritten by DynaFit."""
        for name, (start, end) in self.ranges.items():
            if end:  # User wants range from start cell to end cell
                self.validate_cell_range(start=start, end=end)
            else:  # User wants range from start cell to end of the start cell's column
                self.validate_cell_string(cell_str=start)

    def validate_cell_string(self, cell_str: str) -> None:
        """Validates a cell string in an Excel spreadsheet. Raises an appropriate error if the validation fails."""
        if cell_str == '':  # Guaranteed only to happen on start cells
            raise EmptyCellError(f'Start cell cannot be empty')
        if not self.is_valid_excel(cell_str=cell_str):
            raise BadCellStringError(f'The string "{cell_str}" is not a valid Excel cell accessor')

    @staticmethod
    def is_valid_excel(cell_str: str) -> bool:
        """Returns whether the string cell_str is a valid Excel cell accessor. This implies that it is an
        alphanumerical string, with all numbers appearing after all letters."""
        #  cell_str must be exclusively composed of letters and numbers
        if not cell_str.isalnum():
            return False
        # All letters in cell_str come before all numbers. Source:
        # https://stackoverflow.com/questions/60758670/
        return bool(re.match("[A-Z]+[1-9]\d*$", cell_str))

    def validate_cell_range(self, start: str, end: str) -> None:
        """Validates range of cells (from start cell to end cell) in an Excel spreadsheet. Raises an appropriate
        error if the validation fails."""
        for cell_str in (start, end):
            self.validate_cell_string(cell_str=cell_str)
        if not self.validate_cell_ranges_share_same_column(start=start, end=end):
            raise MismatchedColumnsError(f'Cells {start} and {end} do not share same column')
        if not self.validate_end_cell_comes_after_start_cell(start=start, end=end):
            raise MismatchedRowsError(f'Start cell {start} comes after end cell {end}')

    def validate_cell_ranges_share_same_column(self, start: str, end: str) -> bool:
        """Returns whether the start and end cells share the same column letter."""
        start_letters = self.extract_letters(start)
        end_letters = self.extract_letters(end)
        return start_letters == end_letters

    def validate_end_cell_comes_after_start_cell(self, start: str, end: str) -> bool:
        """Returns whether the row number of the end cell comes after the row number of the start cell."""
        start_numbers = self.extract_digits(start)
        end_numbers = self.extract_digits(end)
        return int(start_numbers) < int(end_numbers)

    def get_data(self) -> pd.DataFrame:
        self.validation_routine()
        data = {}
        for name, (start, end) in self.ranges.items():
            if not end:  # Figure out where the column ends
                end = self.get_end_cell(start=start)
            cell_range = f'{start}:{end}'
            cell_rows = self.ws[cell_range]
            values = self.get_cell_values(cell_rows)
            data[name] = values
        try:
            df = pd.DataFrame(data)
        except ValueError:
            raise DifferentSizeError(('Columns have different number of numeric elements (after removing rows '
                                      'containing text or empty cells). Please check the selected data ranges.'))
        return df.dropna()

    @staticmethod
    def get_cell_values(rows: Tuple[Tuple[Cell]]) -> np.ndarray:
        """Returns the values of cells in a column (tuple of tuples)."""
        return pd.to_numeric([row[0].value for row in rows], errors='coerce')

    def get_end_cell(self, start: str) -> str:
        """Given a valid cell string, returns the cell string at the end of the column (same column + max row)
        of the Excel spreadsheet associated with the ExcelValidator instance."""
        letter = self.extract_letters(start)
        number = self.ws.max_row
        return f'{letter}{number}'

    @staticmethod
    def extract_letters(s: str) -> str:
        """Returns the letter portion of an alphanumerical string."""
        return ''.join(char for char in s if char.isalpha())

    @staticmethod
    def extract_digits(s: str) -> str:
        """Returns the digit portion of an alphanumerical string."""
        return ''.join(char for char in s if char.isdigit())
