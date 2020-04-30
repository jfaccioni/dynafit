"""utils.py - bundles some utility functions used throughout the code."""

from typing import Tuple

import numpy as np


def get_start_end_values(array: np.ndarray) -> Tuple[float, float]:
    """Returns the first and last value of the input array."""
    return array[0], array[-1]


def get_missing_coordinate(x1: float, y1: float, x2: float, angular_coefficient: float = -1.0) -> float:
    """Returns the y2 coordinate at the point (x2, y2) of a line which has an angular coefficient of
    "angular_coefficient" and passes through the point (x1, y1)."""
    linear_coefficient = y1 - (angular_coefficient * x1)
    y2 = linear_coefficient + (x2 * angular_coefficient)
    return y2


def array_in_sequence(array: np.ndarray, sequence) -> bool:
    """Returns whether the input numpy array is inside the input Python sequence."""
    return any(isinstance(element, type(array)) and np.array_equal(element, array) for element in sequence)
