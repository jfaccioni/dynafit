import os
import sys
import traceback
from csv import writer
from io import StringIO
from itertools import zip_longest
from typing import Any, Dict, List, Tuple
from zipfile import BadZipFile

import matplotlib
import openpyxl
from PySide2.QtCore import QEvent, QThreadPool
from PySide2.QtGui import QKeySequence
from PySide2.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame,
                               QGridLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox,
                               QPushButton, QRadioButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget, qApp)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas, NavigationToolbar2QT as Navbar
from matplotlib.pyplot import Figure

from src.logic import dynafit

# Set a small font for plots
matplotlib.rc('font', size=8)
# Set debugging flag
DEBUG = True


class DynaFitGUI(QMainWindow):
    """Class representing the DynaFit GUI as a whole"""
    def __init__(self) -> None:
        """Init method of DynaFitGUI class"""
        super().__init__()
        self.resize(1080, 680)
        self.threadpool = QThreadPool()
        self.data = None
        self.results = None

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
        self.input_sheetname_label = QLabel(self, text='Sheet to analyse:', styleSheet='font-weight: 600')
        self.input_sheetname = QComboBox(self)
        self.input_sheetname.addItem('No data yet')
        self.input_frame.layout().addWidget(self.input_sheetname_label, 2, 0)
        self.input_frame.layout().addWidget(self.input_sheetname, 2, 1)
        # Add section above to left column
        left_column.addWidget(self.input_frame)

        # --Data type/range frame--
        # Region where user selects whether data represents CS1+CS2 or CS+GR and the respective ranges
        left_column.addWidget(QLabel(self, text='Data type and range', styleSheet='font-weight: 600'))
        self.data_frame = QFrame(self, frameShape=QFrame.Box)
        self.data_frame.setLayout(QGridLayout())
        # CS1 CS2 button
        self.cs1_cs2_button = QRadioButton(self, text='Initial and final colony sizes', checked=True)
        self.cs1_cs2_button.clicked.connect(self.cs1_cs2_setup)
        self.data_frame.layout().addWidget(self.cs1_cs2_button, 0, 0, 1, 2)
        # CS GR button
        self.cs_gr_button = QRadioButton(self, text='Initial colony size and growth rate')
        self.cs_gr_button.clicked.connect(self.cs_gr_setup)
        self.data_frame.layout().addWidget(self.cs_gr_button, 1, 0, 1, 2)
        # Time interval label
        self.time_interval_label = QLabel(self, text='Time delta (hours):', wordWrap=True)
        self.data_frame.layout().addWidget(self.time_interval_label, 0, 2, 1, 1)
        # Time interval value
        self.time_interval_num = QDoubleSpinBox(self, minimum=0.0, value=24.0, maximum=1000.0, singleStep=1.0)
        self.data_frame.layout().addWidget(self.time_interval_num, 0, 3, 1, 1)
        # CS label
        self.CS_label = QLabel(self, text='Initial colony size column', wordWrap=True)
        self.data_frame.layout().addWidget(self.CS_label, 2, 0, 1, 2)
        # CS start
        self.CS_start_label = QLabel(self, text='From cell:')
        self.data_frame.layout().addWidget(self.CS_start_label, 3, 0)
        self.CS_start_textbox = QLineEdit(self, placeholderText="")
        self.data_frame.layout().addWidget(self.CS_start_textbox, 3, 1)
        # CS end
        self.CS_end_label = QLabel(self, text='To cell:')
        self.data_frame.layout().addWidget(self.CS_end_label, 4, 0)
        self.CS_end_textbox = QLineEdit(self, placeholderText="Entire Column")
        self.data_frame.layout().addWidget(self.CS_end_textbox, 4, 1)
        # GR label
        self.GR_label = QLabel(self, text='Final colony size column', wordWrap=True)
        self.data_frame.layout().addWidget(self.GR_label, 2, 2, 1, 2)
        # GR start
        self.GR_start_label = QLabel(self, text='From cell:')
        self.data_frame.layout().addWidget(self.GR_start_label, 3, 2)
        self.GR_start_textbox = QLineEdit(self, placeholderText="")
        self.data_frame.layout().addWidget(self.GR_start_textbox, 3, 3)
        # GR end
        self.GR_end_label = QLabel(self, text='To cell:')
        self.data_frame.layout().addWidget(self.GR_end_label, 4, 2)
        self.GR_end_textbox = QLineEdit(self, placeholderText="Entire Column")
        self.data_frame.layout().addWidget(self.GR_end_textbox, 4, 3)
        # Add section above to left column
        left_column.addWidget(self.data_frame)

        # --Options frame--
        # Region where the user selects parameters for the CVP
        left_column.addWidget(QLabel(self, text='Parameter selection', styleSheet='font-weight: 600'))
        self.options_frame = QFrame(self, frameShape=QFrame.Box)
        self.options_frame.setLayout(QGridLayout())
        # Max individual colony size parameter
        tooltip = 'Colonies up to this colony size are placed in individual groups'
        self.max_individual_cs_label = QLabel(self, text='Max individual colony groups', wordWrap=True)
        self.max_individual_cs_label.setToolTip(tooltip)
        self.options_frame.layout().addWidget(self.max_individual_cs_label, 0, 0, 1, 1)
        self.max_individual_cs = QSpinBox(self, minimum=0, value=5, maximum=100, singleStep=1)
        self.max_individual_cs.setToolTip(tooltip)
        self.options_frame.layout().addWidget(self.max_individual_cs, 0, 1, 1, 1)
        # Number of bins parameter
        tooltip = 'Remaining colonies are equally distributed in these many groups'
        self.large_colony_groups_label = QLabel(self, text='Large colony groups')
        self.large_colony_groups_label.setToolTip(tooltip)
        self.options_frame.layout().addWidget(self.large_colony_groups_label, 0, 2, 1, 1)
        self.large_colony_groups = QSpinBox(self, minimum=0, value=5, maximum=100, singleStep=1)
        self.large_colony_groups.setToolTip(tooltip)
        self.options_frame.layout().addWidget(self.large_colony_groups, 0, 3, 1, 1)
        # Number of bootstrap repeats parameter
        self.bootstrap_repeats_label = QLabel(self, text='Bootstrap repeats')
        self.bootstrap_repeats = QSpinBox(self, minimum=0, maximum=1_000_000, singleStep=1)
        self.bootstrap_repeats.setValue(100)
        self.options_frame.layout().addWidget(self.bootstrap_repeats_label, 1, 0, 1, 1)
        self.options_frame.layout().addWidget(self.bootstrap_repeats, 1, 1, 1, 1)
        # Confidence interval parameter
        self.ci_checkbox = QCheckBox(self, text='Use conf. int.')
        self.ci_checkbox.clicked.connect(self.confidence_setup)
        self.options_frame.layout().addWidget(self.ci_checkbox, 1, 2, 1, 1)
        self.confidence_num = QDoubleSpinBox(self, minimum=0, value=0.95, maximum=1.0, singleStep=0.01)
        self.confidence_num.setEnabled(False)
        self.options_frame.layout().addWidget(self.confidence_num, 1, 3, 1, 1)
        # Filter outliers parameter
        self.outliers_checkbox = QCheckBox(self, text='Filter outliers before plotting')
        self.options_frame.layout().addWidget(self.outliers_checkbox, 2, 0, 1, 2)
        # Plot violins parameter
        self.violin_checkbox = QCheckBox(self, text='Add violins to plot')
        self.options_frame.layout().addWidget(self.violin_checkbox, 2, 2, 1, 2)
        # Add section above to left column
        left_column.addWidget(self.options_frame)

        # --Plot frame--
        # Region where the button to plot is located, as well as the calculated AAC
        plot_grid = QGridLayout()
        # Plot button
        self.plot_button = QPushButton(self, text='Plot CVP')
        self.plot_button.clicked.connect(self.dynafit_run)
        plot_grid.addWidget(self.plot_button, 0, 0, 1, 1)
        # Excel export button
        self.to_excel_button = QPushButton(self, text='Save to Excel')
        self.to_excel_button.clicked.connect(self.save_to_excel_dialog)
        self.to_excel_button.setDisabled(True)
        plot_grid.addWidget(self.to_excel_button, 0, 1, 1, 1)
        # CSV export button
        self.to_csv_button = QPushButton(self, text='Save to csv')
        self.to_csv_button.clicked.connect(self.save_to_csv_dialog)
        self.to_csv_button.setDisabled(True)
        plot_grid.addWidget(self.to_csv_button, 0, 2, 1, 1)
        # CoDy table of values
        self.result_table = QTableWidget(self, rowCount=0, columnCount=4)
        self.result_table.setHorizontalHeaderItem(0, QTableWidgetItem('Parameter'))
        self.result_table.setHorizontalHeaderItem(1, QTableWidgetItem('Value'))
        self.result_table.setHorizontalHeaderItem(2, QTableWidgetItem('Mean X'))
        self.result_table.setHorizontalHeaderItem(3, QTableWidgetItem('Mean Y'))
        self.result_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.result_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.result_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.result_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.result_table.installEventFilter(self)
        plot_grid.addWidget(self.result_table, 1, 0, 3, 3)
        # Add section above to left column
        left_column.addLayout(plot_grid)

        # ### RIGHT COLUMN

        # --Plot canvas--
        # Region where plots are displayed
        self.fig = Figure(facecolor='white')
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
            self.data = openpyxl.load_workbook(query, data_only=True)
        except BadZipFile:
            self.dynafit_raised_exception(BadExcelFile('Cannot load input Excel file. Is it corrupted?'))
        else:
            filename = os.path.basename(query)
            self.input_filename.setText(filename)
            self.input_sheetname.clear()
            self.input_sheetname.addItems(self.data.sheetnames)

    def cs1_cs2_setup(self) -> None:
        """Changes interface when the user chooses data ranges representing CS1 and CS2"""
        self.time_interval_label.setEnabled(True)
        self.time_interval_num.setEnabled(True)
        self.CS_label.setText('Initial colony size column')
        self.GR_label.setText('Final colony size column')

    def cs_gr_setup(self) -> None:
        """Changes interface when the user chooses data ranges representing CS and GR"""
        self.time_interval_label.setEnabled(False)
        self.time_interval_num.setEnabled(False)
        self.CS_label.setText('Colony size column')
        self.GR_label.setText('Growth rate column')

    def confidence_setup(self) -> None:
        """Changes interface when the user clicks on the CI radio button"""
        if self.ci_checkbox.isChecked():
            self.confidence_num.setEnabled(True)
        else:
            self.confidence_num.setEnabled(False)

    def dynafit_run(self) -> None:
        """Runs DynaFit"""
        try:
            self.dynafit_setup()
            dynafit_settings = self.get_dynafit_settings()
            cody_dict = dynafit(**dynafit_settings)
        except Exception as e:
            self.dynafit_raised_exception(e)
        else:
            self.dynafit_no_exceptions_raised(cody_dict)
        finally:
            self.dynafit_cleanup()

    def dynafit_setup(self) -> None:
        """Called by DynaFitWorker when it starts running.
        Sets up DynaFit by modifying the label on the plot button and clearing both Axes"""
        self.plot_button.setText('Plotting...')
        self.plot_button.setEnabled(False)
        self.CVP_ax.clear()
        self.histogram_ax.clear()

    def get_dynafit_settings(self) -> Dict[str, Any]:
        """Bundles the information and parameters selected by the user into a single
        dictionary and then returns it. Does not perform any kind of validation
        (this is delegated to the validator.py module)"""
        return {
            'data': self.data,
            'filename': os.path.splitext(self.input_filename.text())[0],
            'sheetname': self.input_sheetname.currentText(),
            'is_raw_colony_sizes': self.cs1_cs2_button.isChecked(),
            'time_delta': self.time_interval_num.value(),
            'cs_start_cell': self.CS_start_textbox.text(),
            'cs_end_cell': self.CS_end_textbox.text(),
            'gr_start_cell': self.GR_start_textbox.text(),
            'gr_end_cell': self.GR_end_textbox.text(),
            'max_binned_colony_size': self.max_individual_cs.value(),
            'bins': self.large_colony_groups.value(),
            'repeats': self.bootstrap_repeats.value(),
            'show_violin': self.violin_checkbox.isChecked(),
            'show_ci': self.ci_checkbox.isChecked(),
            'filter_outliers': self.outliers_checkbox.isChecked(),
            'confidence': self.confidence_num.value(),
            'fig': self.fig,
            'cvp_ax': self.CVP_ax,
            'hist_ax': self.histogram_ax,
        }

    def dynafit_raised_exception(self, error: Exception) -> None:
        """Called by DynaFitWorker if any Exception is raised.
        Clears both Axes and raises Exception as a message box"""
        self.CVP_ax.clear()
        self.histogram_ax.clear()
        name = f'{error.__class__.__name__}:\n{error}'
        trace = traceback.format_exc()
        self.show_error_message((name, trace))

    def dynafit_no_exceptions_raised(self, return_value: Tuple[Dict[str, Any], List[str], List[str]]):
        """Called by DynaFitWorker if no Exceptions are raised.
        Currently unimplemented"""
        results, xs, ys = return_value
        self.to_excel_button.setEnabled(True)
        self.to_csv_button.setEnabled(True)
        self.result_table.clearContents()
        self.result_table.setRowCount(max(len(results), len(xs), len(ys)))
        for index, (name, value, x, y) in enumerate(zip_longest(list(results.keys()), list(results.values()), xs, ys,
                                                                fillvalue='')):
            self.result_table.setItem(index, 0, QTableWidgetItem(name))
            self.result_table.setItem(index, 1, QTableWidgetItem(str(value)))
            self.result_table.setItem(index, 2, QTableWidgetItem(str(x)))
            self.result_table.setItem(index, 3, QTableWidgetItem(str(y)))

    def dynafit_cleanup(self):
        """Called by DynaFitWorker when it finished running (regardless of Exceptions).
        Restores label on the plot button and removes the axis lines from the histogram"""
        self.plot_button.setText('Plot CVP')
        self.plot_button.setEnabled(True)
        self.histogram_ax.set_axis_off()
        self.canvas.draw()

    def show_error_message(self, error_tuple):
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
        self.cs_gr_button.setChecked(True)
        self.cs_gr_setup()

    # the following methods allow for clipboard copy of CoDy table
    # from: https://stackoverflow.com/questions/40469607/
    # how-to-copy-paste-multiple-items-form-qtableview-created-by-qstandarditemmodel

    # add event filter
    def eventFilter(self, source, event):
        """Event filter"""
        if event.type() == QEvent.KeyPress and event.matches(QKeySequence.Copy):
            self.copy_selection()
            return True
        return super().eventFilter(source, event)

    # add copy method
    def copy_selection(self):
        selection = self.result_table.selectedIndexes()
        if selection:
            rows = sorted(index.row() for index in selection)
            columns = sorted(index.column() for index in selection)
            rowcount = rows[-1] - rows[0] + 1
            colcount = columns[-1] - columns[0] + 1
            table = [[''] * colcount for _ in range(rowcount)]
            for index in selection:
                row = index.row() - rows[0]
                column = index.column() - columns[0]
                table[row][column] = index.data()
            stream = StringIO()
            writer(stream).writerows(table)
            qApp.clipboard().setText(stream.getvalue())

    def save_to_excel_dialog(self):
        if self.results is None:
            self.dynafit_raised_exception(ValueError('No results yet. Please click on the "Plot CVP" button first.'))
            return
        placeholder = f'{self.result_table.item(0, 1).text()}_{self.result_table.item(1, 1).text()}.xlsx'
        query, _ = QFileDialog.getSaveFileName(self, 'Select file to save to', placeholder,
                                               'Excel Spreadsheet (*.xlsx)')
        if query:
            self.save_excel(path=query)

    def save_excel(self, path: str) -> None:
        if not path.endswith('.xlsx'):
            path = path + '.xlsx'
        self.results.to_excel(path, index=None)

    def save_to_csv_dialog(self):
        if self.results is None:
            self.dynafit_raised_exception(ValueError('No results yet!'))
            return
        placeholder = f'{self.result_table.item(0, 1).text()}_{self.result_table.item(1, 1).text()}.csv'
        query, _ = QFileDialog.getSaveFileName(self, 'Select file to save to', placeholder,
                                               'Comma-separated values (*.csv)')
        if query:
            self.save_csv(path=query)

    def save_csv(self, path: str) -> None:
        if not path.endswith('.csv'):
            path = path + '.csv'
        self.results.to_csv(path, index=None)


class BadExcelFile(Exception):
    """Exception raised when openpyxl cannot parse the input Excel file. This is the only exception raised by the GUI
     itself, and not by the ExcelValidator class, since the Excel spreadsheet names must be read in advance"""
    def __init__(self, *args):
        super().__init__(*args)


if __name__ == '__main__':
    app = QApplication()
    dfgui = DynaFitGUI()
    if DEBUG:
        dfgui.debug()
    dfgui.show()
    sys.exit(app.exec_())
