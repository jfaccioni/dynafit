"""test_utils.py - unit tests for utils.py."""

import unittest

import numpy as np

from src.utils import get_missing_coordinate, get_start_end_values


class TestUtilsModule(unittest.TestCase):
    """Tests the utils.py module."""
    def test_get_start_end_values_two_or_more_elements_in_array(self) -> None:
        arr = np.array([1, 2])
        return_value = get_start_end_values(array=arr)
        self.assertEqual(len(return_value), 2)
        self.assertEqual(arr[0], return_value[0])
        self.assertEqual(arr[-1], return_value[1])

    def test_get_start_end_values_only_one_elements_in_array(self) -> None:
        arr = np.array([5])
        return_value = get_start_end_values(array=arr)
        self.assertEqual(len(return_value), 2)
        self.assertEqual(arr[0], return_value[0])
        self.assertEqual(arr[0], return_value[1])

    def test_get_start_end_values_raises_index_error_with_empty_array(self) -> None:
        arr = np.array([])
        with self.assertRaises(IndexError):
            get_start_end_values(array=arr)

    def test_get_missing_coord_y_equals_x_line(self) -> None:
        # line -> y = x
        p1 = (0, 0)
        x1, y1 = p1
        p2 = (1, 1)
        x2, expected_y2 = p2
        actual_y2 = get_missing_coordinate(x1=x1, y1=y1, x2=x2, angular_coefficient=1.0)
        self.assertEqual(expected_y2, actual_y2)

    def test_get_missing_coord_y_equals_minus_two_x_plus_three_line(self) -> None:
        # line -> y = -2x + 3
        p1 = (0, 3)
        x1, y1 = p1
        p2 = (5, -7)
        x2, expected_y2 = p2
        actual_y2 = get_missing_coordinate(x1=x1, y1=y1, x2=x2, angular_coefficient=-2.0)
        self.assertEqual(expected_y2, actual_y2)


if __name__ == '__main__':
    unittest.main()
