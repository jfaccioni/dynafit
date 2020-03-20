import sys
import traceback
from typing import Any, Callable, Dict, Tuple
from zipfile import BadZipFile

import matplotlib
import openpyxl
from PySide2.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot
from PySide2.QtWidgets import (QApplication, QComboBox, QFileDialog, QFormLayout, QFrame, QGridLayout, QHBoxLayout,
                               QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas, NavigationToolbar2QT as Navbar
from matplotlib.pyplot import Figure

from src.logic import dynafit

# Set a small font for plots
matplotlib.rc('font', size=8)
# Set debugging flag
DEBUG = False


class DynaFitGUI(QMainWindow):
    """Class representing the DynaFit GUI as a whole"""
    def __init__(self) -> None:
        """Init method of DynaFitGUI class"""
        super().__init__()
        self.resize(1080, 680)
        self.threadpool = QThreadPool()
        self.data = None

        # ### MAIN SETUP

        # --Central widget--
        self.frame = QWidget(self)
        self.setCentralWidget(self.frame)
        # --Main layout--
        main_layout = QVBoxLayout()
        self.frame.setLayout(main_layout)
        # --Columns--
        # App body
        columns = QHBoxLayout()
        main_layout.addLayout(columns)
        # --Left column--
        # Region where user selects parameters
        left_column = QVBoxLayout()
        columns.addLayout(left_column)
        # --Right column--
        # Region where plots are displayed
        right_column = QVBoxLayout()
        columns.addLayout(right_column)

        # ### LEFT COLUMN

        # --DynaFit GUI title--
        self.title = QLabel(self, text='DynaFit GUI', styleSheet='font-size: 16pt; font-weight: 600')
        left_column.addWidget(self.title)

        # --Input frame--
        # Region where users select worksheet and spreadsheet to analyse
        left_column.addWidget(QLabel(self, text='Input selection', styleSheet='font-weight: 600'))
        self.input_frame = QFrame(self, frameShape=QFrame.Box)
        self.input_frame.setLayout(QGridLayout())
        # Input button
        self.input_button = QPushButton(self, text='Load input data')
        self.input_button.clicked.connect(self.load_data_dialog)
        self.input_frame.layout().addWidget(self.input_button, 0, 0, 1, 2)
        # Input filename and label
        self.input_filename_label = QLabel(self, text='Data loaded:', styleSheet='font-weight: 600')
        self.input_frame.layout().addWidget(self.input_filename_label, 1, 0)
        self.input_filename = QLabel(self, text='None')
        self.input_frame.layout().addWidget(self.input_filename, 1, 1)
        # Input sheetname and label
        self.input_sheetname_label = QLabel(self, text='Sheetname to analyse:', styleSheet='font-weight: 600')
        self.input_sheetname = QComboBox(self)
        self.input_sheetname.addItem('No data yet')
        self.input_frame.layout().addWidget(self.input_sheetname_label, 2, 0)
        self.input_frame.layout().addWidget(self.input_sheetname, 2, 1)
        # Add section above to left column
        self.input_frame.layout()
        left_column.addWidget(self.input_frame)

        # --Cell range frame--
        # Region where user selects the ranges of cells containing CS and GR data
        left_column.addWidget(QLabel(self, text='Cell range selection', styleSheet='font-weight: 600'))
        self.cell_range_frame = QFrame(self, frameShape=QFrame.Box)
        self.cell_range_frame.setLayout(QGridLayout())
        # CS label
        self.CS_label = QLabel(self, text='Select Colony Size column (blank for entire column)', wordWrap=True)
        self.cell_range_frame.layout().addWidget(self.CS_label, 0, 0, 1, 2)
        # CS start
        self.CS_start_label = QLabel(self, text='From:')
        self.cell_range_frame.layout().addWidget(self.CS_start_label, 1, 0)
        self.CS_start_textbox = QLineEdit(self, placeholderText="A1")
        self.cell_range_frame.layout().addWidget(self.CS_start_textbox, 1, 1)
        # CS end
        self.CS_end_label = QLabel(self, text='To:')
        self.cell_range_frame.layout().addWidget(self.CS_end_label, 2, 0)
        self.CS_end_textbox = QLineEdit(self, placeholderText="None")
        self.cell_range_frame.layout().addWidget(self.CS_end_textbox, 2, 1)
        # GR label
        self.GR_label = QLabel(self, text='Select Growth Rate column (blank for entire column)', wordWrap=True)
        self.cell_range_frame.layout().addWidget(self.GR_label, 0, 2, 1, 2)
        # GR start
        self.GR_start_label = QLabel(self, text='From:')
        self.cell_range_frame.layout().addWidget(self.GR_start_label, 1, 2)
        self.GR_start_textbox = QLineEdit(self, placeholderText="A1")
        self.cell_range_frame.layout().addWidget(self.GR_start_textbox, 1, 3)
        # GR end
        self.GR_end_label = QLabel(self, text='To:')
        self.cell_range_frame.layout().addWidget(self.GR_end_label, 2, 2)
        self.GR_end_textbox = QLineEdit(self, placeholderText="None")
        self.cell_range_frame.layout().addWidget(self.GR_end_textbox, 2, 3)
        # Add section above to left column
        left_column.addWidget(self.cell_range_frame)

        # --Options frame--
        # Region where the user selects parameters for the CVP
        left_column.addWidget(QLabel(self, text='Parameter selection', styleSheet='font-weight: 600'))
        self.options_frame = QFrame(self, frameShape=QFrame.Box)
        self.options_frame.setLayout(QFormLayout())
        # Max colony size parameter
        self.maxbin_colsize_label = QLabel(self, text='Max binned colony size')
        self.maxbin_colsize_num = QSpinBox(self, minimum=0, value=5, maximum=20, singleStep=1)
        self.options_frame.layout().addRow(self.maxbin_colsize_label, self.maxbin_colsize_num)
        # Number of bins parameter
        self.nbins_label = QLabel(self, text='Number of bins for remaining population')
        self.nbins_num = QSpinBox(self, minimum=0, value=5, maximum=20, singleStep=1)
        self.options_frame.layout().addRow(self.nbins_label, self.nbins_num)
        # Number of runs parameter
        self.nruns_label = QLabel(self, text='Number of independent runs to perform')
        self.nruns_num = QSpinBox(self, minimum=0, value=10, maximum=100, singleStep=1)
        self.options_frame.layout().addRow(self.nruns_label, self.nruns_num)
        # Number of repeats parameter
        self.nrepeats_label = QLabel(self, text='Number of repeated samples for each run/CS')
        self.nrepeats_num = QSpinBox(self, minimum=0, value=10, maximum=100, singleStep=1)
        self.options_frame.layout().addRow(self.nrepeats_label, self.nrepeats_num)
        # Sample size parameter
        self.samplesize_label = QLabel(self, text='Sample size')
        self.samplesize_num = QSpinBox(self, minimum=0, value=20, maximum=100, singleStep=1)
        self.options_frame.layout().addRow(self.samplesize_label, self.samplesize_num)
        # Add section above to left column
        left_column.addWidget(self.options_frame)

        # --Plot frame--
        # Region where the button to plot is located, as well as the calculated AAC
        left_column.addWidget(QLabel(self, text='Plot options', styleSheet='font-weight: 600'))
        self.plot_frame = QFrame(self, frameShape=QFrame.Box)
        self.plot_frame.setLayout(QGridLayout())
        # Plot button
        self.plot_button = QPushButton(self, text='Plot CVP with above parameters')
        self.plot_button.clicked.connect(self.dynafit_run)
        self.plot_frame.layout().addWidget(self.plot_button, 0, 0, 1, 2)
        # AAC label and number
        self.area_above_curve_label = QLabel(self, text='Measured area above curve:', styleSheet='font-weight: 600')
        self.plot_frame.layout().addWidget(self.area_above_curve_label, 1, 0)
        self.area_above_curve_number = QLabel(self, text='None')
        self.plot_frame.layout().addWidget(self.area_above_curve_number, 1, 1)
        left_column.addWidget(self.plot_frame)

        # ### RIGHT COLUMN

        # --Plot canvas--
        # Region where plots are displayed
        self.fig = Figure(facecolor="white")
        self.canvas = Canvas(self.fig)
        self.canvas.setParent(self)
        # Axes instance used to plot CVP
        self.CVP_ax = self.fig.add_axes([0.1, 0.2, 0.85, 0.75])
        # Axes instance used to plot population histogram
        self.histogram_ax = self.fig.add_axes([0.05, 0.0, 0.9, 0.1])
        self.histogram_ax.set_axis_off()
        # Add canvas above to right column
        right_column.addWidget(self.canvas)
        # Add a navigation bar to right column
        right_column.addWidget(Navbar(self.canvas, self.frame))

        # Set column stretch so that only plot gets rescaled with GUI
        columns.setStretch(1, 10)

    def load_data_dialog(self) -> None:
        """Prompts the user to select the data (Excel spreadsheet) to load"""
        query, _ = QFileDialog.getOpenFileName(self, 'Select input file', '', 'Excel Spreadsheet (*.xlsx)')
        if query:
            self.load_data(query=query)

    def load_data(self, query: str) -> None:
        """Loads input data into memory, storing it in the GUI's self.data attribute"""
        try:
            self.data = openpyxl.load_workbook(query)
        except BadZipFile:
            self.raise_error_as_exception(BadExcelFile('Cannot load input Excel file. Is it corrupted?'))
        else:
            self.input_filename.setText(f'{query.split("/")[-1]}')
            self.input_sheetname.clear()
            self.input_sheetname.addItems(self.data.sheetnames)

    def dynafit_run(self) -> None:
        """Runs DynaFit in a separate Worker thread"""
        # This part works as a Try block
        worker = DynaFitWorker(func=dynafit, **self.get_dynafit_settings())
        worker.signals.started.connect(self.dynafit_setup)
        # This part works as an Except block
        worker.signals.error.connect(self.dynafit_raised_exception)
        # This part works as an Else block
        worker.signals.success.connect(self.dynafit_no_exceptions_raised)
        # This part works as a Finally block
        worker.signals.finished.connect(self.dynafit_cleanup)
        # Run DynaFit function with parameters from GUI
        self.threadpool.start(worker)

    def get_dynafit_settings(self) -> Dict[str, Any]:
        """Bundles the information and parameters selected by the user into a single
        dictionary and then returns it. Does not perform any kind of validation
        (this is delegated to the validator.py module)"""
        return {
            'data': self.data,
            'sheetname': self.input_sheetname.currentText(),
            'cs_start_cell': self.CS_start_textbox.text(),
            'cs_end_cell': self.CS_end_textbox.text(),
            'gr_start_cell': self.GR_start_textbox.text(),
            'gr_end_cell': self.GR_end_textbox.text(),
            'max_binned_colony_size': self.maxbin_colsize_num.value(),
            'bins': self.nbins_num.value(),
            'runs': self.nruns_num.value(),
            'repeats': self.nrepeats_num.value(),
            'sample_size': self.samplesize_num.value(),
            'fig': self.fig,
            'cvp_ax': self.CVP_ax,
            'hist_ax': self.histogram_ax,
        }

    def dynafit_setup(self) -> None:
        """Called by DynaFitWorker when it starts running.
        Sets up DynaFit by modifying the label on the plot button and clearing both Axes"""
        self.plot_button.setText('Plotting...')
        self.plot_button.setEnabled(False)
        self.CVP_ax.clear()
        self.histogram_ax.clear()

    def dynafit_raised_exception(self, error_tuple: Tuple[str, str]) -> None:
        """Called by DynaFitWorker if any Exception is raised.
        Clears both Axes and raises Exception as a message box"""
        self.CVP_ax.clear()
        self.histogram_ax.clear()
        self.raise_error_as_tuple(error_tuple)

    def dynafit_no_exceptions_raised(self):
        """Called by DynaFitWorker if no Exceptions are raised.
        Currently unimplemented"""
        pass

    def dynafit_cleanup(self):
        """Called by DynaFitWorker when it finished running (regardless of Exceptions).
        Restores label on the plot button and removes the axis lines from the histogram"""
        self.plot_button.setText('Plot CVP with above parameters')
        self.plot_button.setEnabled(True)
        self.histogram_ax.set_axis_off()
        self.canvas.draw()

    def raise_error_as_exception(self, error: Exception) -> None:
        """Function responsible for re-raising an error, but in a message box.
        Argument must be an Exception"""
        name = f'{error.__class__.__name__}:\n{error}'
        trace = traceback.format_exc()
        self.raise_error_as_tuple((name, trace))

    def raise_error_as_tuple(self, error_tuple):
        """Function responsible for re-raising an error, but in a message box.
        Argument must be a a tuple containing the string of the error name and the stack trace"""
        error, trace = error_tuple
        box = QMessageBox(self, windowTitle='An error occurred!', text=error, detailedText=trace)
        box.show()

    def debug(self):
        """Implemented for easier debugging"""
        self.load_data(query='data/example.xlsx')
        self.CS_start_textbox.setText('A2')
        self.GR_start_textbox.setText('B2')


class DynaFitWorker(QRunnable):
    """Worker thread for DynaFit analysis. Avoids unresponsive GUI."""
    def __init__(self, func: Callable, *args, **kwargs) -> None:
        """Init method of DynaFitWorker class"""
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        """Runs the Worker thread"""
        self.signals.started.emit()
        try:
            self.func(*self.args, **self.kwargs)
        except Exception as e:
            self.emit_error(e)

        else:
            self.signals.success.emit()
        finally:
            self.signals.finished.emit()

    def emit_error(self, e: Exception) -> None:
        """Function responsible for emitting an error back to the main thread.
        Argument must be an Exception"""
        name = f'{e.__class__.__name__}:\n{e}'
        trace = traceback.format_exc()
        self.signals.error.emit((name, trace))


class WorkerSignals(QObject):
    """Defines the signals available from a running worker thread. Supported signals are:
       - Started: Worker has begun working. Nothing is emitted.
       - Finished: Worker has done executing (either naturally or by an Exception). Nothing is emitted.
       - Success: Worker has finished executing without errors. Nothing is emitted.
       - Error: an Exception was raised. Emits a tuple containing an Exception object and the traceback as a string."""
    started = Signal()
    finished = Signal()
    success = Signal()
    error = Signal(Exception)


class BadExcelFile(Exception):
    """Exception raised when openpyxl cannot parse the input Excel file."""
    def __init__(self, *args):
        super().__init__(*args)


if __name__ == '__main__':
    app = QApplication()
    dfgui = DynaFitGUI()
    if DEBUG:
        dfgui.debug()
    dfgui.show()
    sys.exit(app.exec_())
