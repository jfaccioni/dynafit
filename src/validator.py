from typing import Dict, List, Tuple

import pandas as pd
from openpyxl import Workbook


class ExcelValidator:
    def __init__(self, data: Workbook, sheetname: str, start_cell_01: str, end_cell_01: str, start_cell_02: str,
                 end_cell_02: str) -> None:
        """init"""
        self.ws = data[sheetname]
        self.max_row = self.ws.max_row
        self.range01 = start_cell_01, end_cell_01
        self.range02 = start_cell_02, end_cell_02
        self.validated_data = self.validate()

    @property
    def ranges(self):
        return self.range01, self.range02

    def validate(self):
        d = {}
        cell_range_names = ['CS', 'GR']
        for range_name, cell_range in zip(cell_range_names, self.ranges):
            self.validate_range(cell_range)
            values = self.validate_values(cell_range)
            d[range_name] = values
        self.validate_size(d)
        return pd.DataFrame(d)

    def validate_range(self, cell_range: Tuple[str, str]):
        for interval in cell_range:
            if interval and not interval.isalnum():
                # Range is not exclusively composed of alphanumerical characters
                raise BadCharError
        start, end = cell_range
        if not start:
            # No start range passed to input
            raise NoStartCellError
        if end:
            # Do not perform following checks if no end range passed to input
            # (user wants to get entire column)
            start_letters = self.extract_letters(start)
            end_letters = self.extract_letters(end)
            if start_letters != end_letters:
                # User passed two different columns for single range
                raise MismatchedColumnsError
            start_number = self.extract_digits(start)
            end_number = self.extract_digits(end)
            if int(start_number) <= int(end_number):
                # Start row comes after end row
                raise MismatchedRowsError

    def validate_values(self, cell_range):
        start, end = cell_range
        if not end:
            end = self.get_end_cell(start_cell=start)
        cells = [v[0] for v in self.ws[f'{start}:{end}']]
        try:
            return [c.value for c in cells if c.value not in [None, '']]
        except TypeError:
            raise NotNumericalError

    def get_end_cell(self, start_cell: str):
        letter = self.extract_letters(start_cell)
        number = self.max_row
        return f'{letter}{number}'

    @staticmethod
    def extract_letters(s):
        return ''.join(char for char in s if char.isalpha())

    @staticmethod
    def extract_digits(s):
        return ''.join(char for char in s if char.isdigit())

    @staticmethod
    def validate_size(d: Dict[str, List[float]]) -> None:
        l1, l2 = [len(vs) for vs in d.values()]
        if l1 != l2:
            raise DifferentSizeError


class BadCharError(Exception):
    """Exception raised when samples cannot be found in the input file."""
    def __init__(self):
        super().__init__()


class NoStartCellError(Exception):
    """Exception raised when samples cannot be found in the input file."""
    def __init__(self):
        super().__init__()


class MismatchedColumnsError(Exception):
    """Exception raised when samples cannot be found in the input file."""
    def __init__(self):
        super().__init__()


class MismatchedRowsError(Exception):
    """Exception raised when samples cannot be found in the input file."""
    def __init__(self):
        super().__init__()


class NotNumericalError(Exception):
    """Exception raised when samples cannot be found in the input file."""
    def __init__(self):
        super().__init__()


class DifferentSizeError(Exception):
    """Exception raised when samples cannot be found in the input file."""
    def __init__(self):
        super().__init__()
