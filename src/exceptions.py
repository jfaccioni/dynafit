# Exceptions raised by all modules are bundled here


# Called on interface.py
class CorruptedExcelFile(Exception):
    """Exception raised when openpyxl cannot parse the input Excel file."""


# Called on core.py
class TooManyBinsError(Exception):
    """Exception raised when too many bins are used."""


class NegativeInfiniteVarianceError(Exception):
    """Exception raised when DynaFit do not result in a continuous function, due to negative infinity values
    calculated for the mean line Y coordinates (zero variance due to small groups)."""


class AbortedByUser(Exception):
    """Exception raised when user aborts the execution of DynaFit analysis."""


# Called on validator.py
class BadCellStringError(Exception):
    """Exception raised when cell does not correspond to a valid Excel cell accessor."""


class DifferentSizeError(Exception):
    """Exception raised when samples cannot be found in the input file."""


class EmptyCellError(Exception):
    """Exception raised when mandatory cell is empty."""


class MismatchedColumnsError(Exception):
    """Exception raised when samples cannot be found in the input file."""


class MismatchedRowsError(Exception):
    """Exception raised when samples cannot be found in the input file."""


class NoExcelFileError(Exception):
    """Exception raised when user runs DynaFit with no input file."""
