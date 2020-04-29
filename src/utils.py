"""utils.py - bundles some utility functions used throughout the code."""

from typing import Tuple

import numpy as np


def truncate_arrays(xs: np.ndarray, ys: np.ndarray, x_cutoff: int) -> Tuple[np.ndarray, np.ndarray]:
    """Truncates and returns the arrays "xs" and "ys" by removing values from the "xs" arrays lower than the cutoff,
    and the removing "ys"'s last elements until both arrays have the same size."""
    xs_trunc = np.array([x for x in xs if x < x_cutoff] + [x_cutoff])
    end_y = np.interp(x_cutoff, xs, ys)
    ys_trunc = np.array(ys[:len(xs_trunc)] + [end_y])
    return xs_trunc, ys_trunc


def calculate_triangle_area(xs: np.ndarray, ys: np.ndarray) -> float:
    """Calculates the endpoint hypothesis triangle area."""
    width = xs[-1] - xs[0]
    height = ys[0] - ys[-1]  # SIGNED area
    return (width * height) / 2


def calculate_triangle_area_with_missing_coordinate(xs: np.ndarray, ys: np.ndarray,
                                                    angular_coefficient: float = -1.0) -> float:
    """Calculates the triangle area of the CVP, i.e. the are delimited by H0 and H1 up to the max X coordinate present
    in the "xs" array."""
    start_x, end_x = get_start_end_values(array=xs)
    triangle_length = abs(end_x - start_x)
    start_y, end_y = get_start_end_values(array=ys)
    final_height = get_missing_coordinate(x1=start_x, y1=start_y, x2=end_x, angular_coefficient=angular_coefficient)
    triangle_height = abs(start_y - final_height)
    triangle_area = (triangle_length * triangle_height) / 2
    return triangle_area


def get_start_end_values(array: np.ndarray) -> Tuple[float, float]:
    """Returns the first and last value of the input array."""
    return array[0], array[-1]


def get_missing_coordinate(x1: float, y1: float, x2: float, angular_coefficient: float = -1.0) -> float:
    """Returns the y2 coordinate at the point (x2, y2) of a line which has an angular coefficient of
    "angular_coefficient" and passes through the point (x1, y1)."""
    linear_coefficient = y1 - (angular_coefficient * x1)
    y2 = linear_coefficient + (x2 * angular_coefficient)
    return y2


def trapezium_integration(xs: np.ndarray, ys: np.ndarray) -> float:
    """Performs trapezium integration over the XY series of coordinates (mean green line), calculating the area
    above the line and below H0. Any area above H0 is calculated as negative area."""
    integrated_area = 0.0
    for i, (x, y) in enumerate(zip(xs, ys)):
        try:
            next_x = xs[i+1]
            next_y = ys[i+1]
        except IndexError:  # Nothing more to add
            return integrated_area
        square = (next_x - x) * (ys[0] - y)  # SIGNED area
        triangle = (next_x - x) * (y - next_y) / 2  # SIGNED area
        trapezium = square + triangle
        integrated_area += trapezium
