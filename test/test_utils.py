"""test_utils.py - unit tests for utils.py."""

import unittest

from src.utils import *


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

    def test_array_in_sequence_for_array_in_sequence(self) -> None:
        array_01 = np.array([1, 2, 3])
        array_02 = np.array([4, 5, 6])
        sequence = (array_01, 'a', True, array_01, array_02, 2)
        self.assertTrue(array_in_sequence(array_01, sequence))
        self.assertTrue(array_in_sequence(array_02, sequence))

    def test_array_in_sequence_for_array_not_sequence(self) -> None:
        array_01 = np.array([1, 2, 3])
        array_02 = np.array([4, 5, 6])
        sequence = ('a', array_02, True, array_02, 2)
        self.assertFalse(array_in_sequence(array_01, sequence))
        self.assertFalse(array_in_sequence(array_01, (True, False, 'Sun', 'Moon', 10)))
        self.assertFalse(array_in_sequence(array_01, 'This string is also a Python sequence.'))


if __name__ == '__main__':
    unittest.main()
