"""test_worker.py - unit tests for worker.py."""
import unittest
from typing import Any
from unittest.mock import MagicMock

from PySide2.QtCore import Slot

from src.worker import Worker, WorkerSignals


class TestWorkerModule(unittest.TestCase):
    """Tests the worker.py module."""
    arg_a = 1
    arg_b = 2
    arbitrary_args = (3, 4)
    kwarg_a = 5
    kwarg_b = 6
    arbitrary_kwargs = {'c': 7, 'd': 8}

    def setUp(self) -> None:
        """Sets up the each unit test by creating a Worker instance."""
        self.worker = Worker(self.add, self.arg_a, self.arg_b, *self.arbitrary_args, kwarg_a=self.kwarg_a,
                             kwarg_b=self.kwarg_b, **self.arbitrary_kwargs)
        self.value = None

    @staticmethod
    def add(a: float, b: float, *args, **kwargs) -> float:  # noqa
        """Returns the sum a + b. Accepts any arbitrary number of additional arguments and keyword arguments."""
        return a + b

    @Slot()  # noqa
    def slot_with_value(self, value: Any) -> None:
        """Slot for setting any value passed in from a Signal to self.value."""
        self.value = value

    @Slot()  # noqa
    def slot_without_value(self) -> Any:
        """Slot for receiving any signal emitted from a Signal without a value associated to it."""
        self.value = None

    def mock_worker_signals(self) -> None:
        """Replaces all Signals connected to self.worker by MagicMock instances."""
        self.worker.signals.finished = MagicMock()
        self.worker.signals.success = MagicMock()
        self.worker.signals.error = MagicMock()

    def test_worker_instance_accepts_arbitrary_args(self) -> None:
        for arg in self.arbitrary_args:
            with self.subTest(arg=arg):
                self.assertIn(arg, self.worker.args)

    def test_worker_instance_bundles_arguments_in_attribute_args(self) -> None:
        expected_args = (self.arg_a, self.arg_b, *self.arbitrary_args)
        self.assertEqual(expected_args, self.worker.args)

    def test_worker_instance_accepts_arbitrary_kwargs(self) -> None:
        for k, v in self.arbitrary_kwargs.items():
            with self.subTest(k=k, v=v):
                self.assertIn(k, self.worker.kwargs.keys())
                self.assertIn(v, self.worker.kwargs.values())

    def test_worker_instance_bundles_keyword_arguments_in_attribute_kwargs(self) -> None:
        expected_kwargs = dict(kwarg_a=self.kwarg_a, kwarg_b=self.kwarg_b, **self.arbitrary_kwargs)
        self.assertEqual(expected_kwargs, self.worker.kwargs)

    def test_worker_signal_attribute_is_an_instance_of_worker_signals_class(self) -> None:
        self.assertIsInstance(self.worker.signals, WorkerSignals)

    def test_add_callbacks_adds_callback_signals_to_kwargs(self) -> None:
        self.worker.add_callbacks()
        expected_kwargs = dict(kwarg_a=self.kwarg_a, kwarg_b=self.kwarg_b,
                               progress_callback=self.worker.signals.progress,
                               warning_callback=self.worker.signals.warning, **self.arbitrary_kwargs)
        self.assertEqual(expected_kwargs, self.worker.kwargs)

    def test_run_calls_func_with_args_and_kwargs(self) -> None:
        mock_func = MagicMock()
        self.worker.func = mock_func
        self.worker.run()
        mock_func.assert_called_with(*self.worker.args, **self.worker.kwargs)  # noqa

    def test_run_emits_finished_and_success_signals_when_no_error_happens(self) -> None:
        self.mock_worker_signals()
        self.worker.run()
        self.worker.signals.finished.emit.assert_called()
        self.worker.signals.success.emit.assert_called()
        self.worker.signals.error.emit.assert_not_called()

    def test_run_emits_finished_and_error_signals_when_some_error_happens(self) -> None:
        self.mock_worker_signals()
        mock_func = MagicMock(side_effect=Exception)
        self.worker.func = mock_func
        self.worker.run()
        self.worker.signals.finished.emit.assert_called()
        self.worker.signals.success.emit.assert_not_called()
        self.worker.signals.error.emit.assert_called()

    def test_run_passes_progress_and_warning_signals_to_func(self) -> None:
        self.worker.add_callbacks()
        mock_func = MagicMock()
        self.worker.func = mock_func
        self.worker.run()
        for downstream_signals in (self.worker.signals.progress, self.worker.signals.warning):
            func_args, func_kwargs = mock_func.call_args
            self.assertIn(downstream_signals, func_kwargs.values())

    def test_worker_signals_progress_emits_an_integer(self) -> None:
        signals = WorkerSignals(parent=None)
        signals.progress.connect(self.slot_with_value)
        value = 1
        signals.progress.emit(value)
        self.assertEqual(self.value, value)

    def test_worker_signals_warning_emits_a_python_object(self) -> None:
        signals = WorkerSignals(parent=None)
        signals.warning.connect(self.slot_with_value)
        value = object
        signals.warning.emit(value)
        self.assertEqual(self.value, value)

    def test_worker_signals_finished_emits_nothing(self) -> None:
        signals = WorkerSignals(parent=None)
        signals.finished.connect(self.slot_without_value)
        signals.finished.emit()
        self.assertEqual(self.value, None)

    def test_worker_signals_success_emits_a_python_object(self) -> None:
        signals = WorkerSignals(parent=None)
        signals.success.connect(self.slot_with_value)
        value = object
        signals.success.emit(value)
        self.assertEqual(self.value, value)

    def test_worker_signals_error_emits_a_python_object(self) -> None:
        signals = WorkerSignals(parent=None)
        signals.error.connect(self.slot_with_value)
        value = object
        signals.error.emit(value)
        self.assertEqual(self.value, value)


if __name__ == '__main__':
    unittest.main()
