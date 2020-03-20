# OLD CODE USED FOR MULTITHREADING (NOT SUPPORTED AS OF NOW)
#
# class DynaFitWorker(QRunnable):
#     """Worker thread for DynaFit analysis. Avoids unresponsive GUI."""
#     def __init__(self, func: Callable, *args, **kwargs) -> None:
#         """Init method of DynaFitWorker class"""
#         super().__init__()
#         self.func = func
#         self.args = args
#         self.kwargs = kwargs
#         self.signals = WorkerSignals()
#
#     @Slot()
#     def run(self) -> None:
#         """Runs the Worker thread"""
#         self.signals.started.emit()
#         try:
#             return_value = self.func(*self.args, **self.kwargs)
#         except Exception as e:
#             self.emit_error(e)
#         else:
#             self.signals.success.emit(return_value)
#         finally:
#             self.signals.finished.emit()
#
#     def emit_error(self, e: Exception) -> None:
#         """Function responsible for emitting an error back to the main thread.
#         Argument must be an Exception"""
#         name = f'{e.__class__.__name__}:\n{e}'
#         trace = traceback.format_exc()
#         self.signals.error.emit((name, trace))
#
#
# class WorkerSignals(QObject):
#     """Defines the signals available from a running worker thread. Supported signals are:
#        - Started: Worker has begun working. Nothing is emitted.
#        - Finished: Worker has done executing (either naturally or by an Exception). Nothing is emitted.
#        - Success: Worker has finished executing without errors. Nothing is emitted.
#        - Error: an Exception was raised. Emits a tuple containing an Exception object and the
#        traceback as a string."""
#     started = Signal()
#     finished = Signal()
#     success = Signal(tuple)
#     error = Signal(Exception)


# def dynafit_run(self) -> None:
#     """Runs DynaFit in a separate Worker thread"""
#     # This part works as a Try block
#     worker = DynaFitWorker(func=dynafit, **self.get_dynafit_settings())
#     worker.signals.started.connect(self.dynafit_setup)
#     # This part works as an Except block
#     worker.signals.error.connect(self.dynafit_raised_exception)
#     # This part works as an Else block
#     worker.signals.success.connect(self.dynafit_no_exceptions_raised)
#     # This part works as a Finally block
#     worker.signals.finished.connect(self.dynafit_cleanup)
#     # Run DynaFit function with parameters from GUI
#     self.threadpool.start(worker)
