"""worker.py - defines a Worker thread object."""

import traceback
from typing import Callable

from PySide2.QtCore import QObject, QRunnable, Signal, Slot


class Worker(QRunnable):
    """Worker thread for DynaFit analysis. Avoids unresponsive GUI."""
    def __init__(self, func: Callable, *args, **kwargs) -> None:
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()  # noqa

    def add_callbacks(self) -> None:
        """Adds keyword arguments related signaling between main thread and worker thread."""
        self.kwargs['progress_callback'] = self.signals.progress
        self.kwargs['warning_callback'] = self.signals.warning

    @Slot()  # noqa
    def run(self) -> None:
        """Runs the Worker thread."""
        try:
            return_value = self.func(*self.args, **self.kwargs)
        except Exception as error:
            trace = traceback.format_exc()
            self.signals.error.emit((error, trace))  # noqa
        else:
            self.signals.success.emit(return_value)  # noqa
        finally:
            self.signals.finished.emit()  # noqa


class WorkerSignals(QObject):
    """Defines the signals available from a running worker thread. Supported signals are:
        Progress: Worker has finished a percentage of its job. Emits an int representing that percentage (0-100).
        SS Warning: Worker has encountered low samples in one or more groups. Emits a tuple containing a QEvent and
          a dictionary containing the low sample groups. Meant to wait for used response in the GUI through a QMutex
          and QWaitCondition before moving on with its execution.
        Finished: Worker has done executing (either naturally or by an Exception). Nothing is emitted.
        Success: Worker finished executing without errors. Emits a tuple of a Plotter object and a pandas DataFrame.
        Error: an Exception was raised. Emits a tuple containing an Exception object and the traceback as a string."""
    progress = Signal(int)
    warning = Signal(object)
    finished = Signal()
    success = Signal(object)
    error = Signal(object)
