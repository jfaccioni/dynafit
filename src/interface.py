import os
import sys
import traceback
from csv import writer
from io import StringIO
from itertools import zip_longest
from typing import Any, Dict, Tuple
from zipfile import BadZipFile

import matplotlib
import numpy as np
import openpyxl
import pandas as pd
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
    """Class representing the DynaFit GUI as a whole."""
    def __init__(self) -> None:
        """Init method of DynaFitGUI class. Sets up the entire interface."""
        super().__init__()

        # ### MAIN SETUP

        # --Application attributes--
        self.resize(1080, 680)
        self.threadpool = QThreadPool()
        self.data = None
        self.results = None
        # --Central widget--
        self.frame = QWidget(self)
        self.setCentralWidget(self.frame)
        # --Main layout--
        main_layout = QVBoxLayout()
        self.frame.setLayout(main_layout)
        # --Columns--
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
        # Load data button
        self.load_data_button = QPushButton(self, text='Load input data')
        self.load_data_button.clicked.connect(self.load_data_dialog)
        self.input_frame.layout().addWidget(self.load_data_button, 0, 0, 1, 2)
        # Input filename and label
        self.input_filename_helper_label = QLabel(self, text='Data loaded:', styleSheet='font-weight: 600')
        self.input_frame.layout().addWidget(self.input_filename_helper_label, 1, 0)
        self.input_filename_label = QLabel(self, text='')
        self.input_frame.layout().addWidget(self.input_filename_label, 1, 1)
        # Input sheetname and label
        self.input_sheetname_label = QLabel(self, text='Sheet to analyse:', styleSheet='font-weight: 600')
        self.input_frame.layout().addWidget(self.input_sheetname_label, 2, 0)
        self.input_sheetname_combobox = QComboBox(self)
        self.input_sheetname_combobox.setDisabled(True)
        self.input_frame.layout().addWidget(self.input_sheetname_combobox, 2, 1)
        # Add input frame section to left column
        left_column.addWidget(self.input_frame)

        # --Data type/range frame--
        # Region where user selects whether data represents CS1+CS2 or CS+GR and the respective ranges
        left_column.addWidget(QLabel(self, text='Data type and range', styleSheet='font-weight: 600'))
        self.data_type_range_frame = QFrame(self, frameShape=QFrame.Box)
        self.data_type_range_frame.setLayout(QGridLayout())
        # CS1/CS2 button (calculates GR)
        self.cs1_cs2_button = QRadioButton(self, text='Initial and final colony sizes', checked=True)
        self.cs1_cs2_button.clicked.connect(self.cs1_cs2_setup)
        self.data_type_range_frame.layout().addWidget(self.cs1_cs2_button, 0, 0, 1, 2)
        # CS/GR button (pre-calculated GR)
        self.cs_gr_button = QRadioButton(self, text='Initial colony size and growth rate')
        self.cs_gr_button.clicked.connect(self.cs_gr_setup)
        self.data_type_range_frame.layout().addWidget(self.cs_gr_button, 1, 0, 1, 2)
        # Time interval label and value -> TODO: find a better place for these widgets in the GUI!
        self.time_interval_label = QLabel(self, text='Hours between initial and final colony sizes:', wordWrap=True)
        self.data_type_range_frame.layout().addWidget(self.time_interval_label, 0, 2, 1, 1)
        self.time_interval_spinbox = QDoubleSpinBox(self, minimum=0.0, value=24.0, maximum=1000.0, singleStep=1.0)
        self.data_type_range_frame.layout().addWidget(self.time_interval_spinbox, 0, 3, 1, 1)
        # CS label and GR label
        self.CS_label = QLabel(self, text='Initial colony size column', styleSheet='font-weight: 600')
        self.data_type_range_frame.layout().addWidget(self.CS_label, 2, 0, 1, 2)
        self.GR_label = QLabel(self, text='Final colony size column', styleSheet='font-weight: 600')
        self.data_type_range_frame.layout().addWidget(self.GR_label, 2, 2, 1, 2)
        # CS start
        self.CS_start_label = QLabel(self, text='From cell:')
        self.data_type_range_frame.layout().addWidget(self.CS_start_label, 3, 0)
        self.CS_start_textbox = QLineEdit(self, placeholderText="")
        self.data_type_range_frame.layout().addWidget(self.CS_start_textbox, 3, 1)
        # CS end
        self.CS_end_label = QLabel(self, text='To cell:')
        self.data_type_range_frame.layout().addWidget(self.CS_end_label, 4, 0)
        self.CS_end_textbox = QLineEdit(self, placeholderText="Entire Column")
        self.data_type_range_frame.layout().addWidget(self.CS_end_textbox, 4, 1)
        # GR start
        self.GR_start_label = QLabel(self, text='From cell:')
        self.data_type_range_frame.layout().addWidget(self.GR_start_label, 3, 2)
        self.GR_start_textbox = QLineEdit(self, placeholderText="")
        self.data_type_range_frame.layout().addWidget(self.GR_start_textbox, 3, 3)
        # GR end
        self.GR_end_label = QLabel(self, text='To cell:')
        self.data_type_range_frame.layout().addWidget(self.GR_end_label, 4, 2)
        self.GR_end_textbox = QLineEdit(self, placeholderText="Entire Column")
        self.data_type_range_frame.layout().addWidget(self.GR_end_textbox, 4, 3)
        # Add section above to left column
        left_column.addWidget(self.data_type_range_frame)

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
        self.max_individual_cs_spinbox = QSpinBox(self, minimum=0, value=5, maximum=100, singleStep=1)
        self.max_individual_cs_spinbox.setToolTip(tooltip)
        self.options_frame.layout().addWidget(self.max_individual_cs_spinbox, 0, 1, 1, 1)
        # Number of bins parameter
        tooltip = 'Remaining colonies are equally distributed in these many groups'
        self.large_colony_groups_label = QLabel(self, text='Large colony groups')
        self.large_colony_groups_label.setToolTip(tooltip)
        self.options_frame.layout().addWidget(self.large_colony_groups_label, 0, 2, 1, 1)
        self.large_colony_groups_spinbox = QSpinBox(self, minimum=0, value=5, maximum=100, singleStep=1)
        self.large_colony_groups_spinbox.setToolTip(tooltip)
        self.options_frame.layout().addWidget(self.large_colony_groups_spinbox, 0, 3, 1, 1)
        # Number of bootstrap repeats parameter
        self.bootstrap_repeats_label = QLabel(self, text='Bootstrap repeats')
        self.options_frame.layout().addWidget(self.bootstrap_repeats_label, 1, 0, 1, 1)
        self.bootstrap_repeats_spinbox = QSpinBox(self, minimum=0, maximum=1_000_000, singleStep=1)
        self.bootstrap_repeats_spinbox.setValue(100)
        self.options_frame.layout().addWidget(self.bootstrap_repeats_spinbox, 1, 1, 1, 1)
        # Confidence interval parameter
        self.add_conf_int_checkbox = QCheckBox(self, text='Use conf. int.')
        self.add_conf_int_checkbox.clicked.connect(self.confidence_setup)
        self.options_frame.layout().addWidget(self.add_conf_int_checkbox, 1, 2, 1, 1)
        self.conf_int_spinbox = QDoubleSpinBox(self, minimum=0, value=0.95, maximum=0.999, singleStep=0.01, decimals=3)
        self.conf_int_spinbox.setEnabled(False)
        self.options_frame.layout().addWidget(self.conf_int_spinbox, 1, 3, 1, 1)
        # Filter outliers parameter
        self.remove_outliers_checkbox = QCheckBox(self, text='Filter outliers before plotting')
        self.options_frame.layout().addWidget(self.remove_outliers_checkbox, 2, 0, 1, 2)
        # Plot violins parameter
        self.add_violins_checkbox = QCheckBox(self, text='Add violins to plot')
        self.options_frame.layout().addWidget(self.add_violins_checkbox, 2, 2, 1, 2)
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
        self.results_table = QTableWidget(self, rowCount=0, columnCount=4)
        for index, column_name in enumerate(['Parameter', 'Value', 'Mean X', 'Mean Y']):
            self.results_table.setHorizontalHeaderItem(index, QTableWidgetItem(column_name))
            self.results_table.horizontalHeader().setSectionResizeMode(index, QHeaderView.Stretch)
        self.results_table.installEventFilter(self)
        plot_grid.addWidget(self.results_table, 1, 0, 3, 3)
        # Add section above to left column
        left_column.addLayout(plot_grid)

        # ### RIGHT COLUMN

        # --Plot canvas--
        # Region where plots are displayed
        self.fig = Figure(facecolor='white')
        self.canvas = Canvas(self.fig)
        self.canvas.setParent(self)
        # Axes instance used to plot CVP
        self.cvp_ax = self.fig.add_axes([0.1, 0.2, 0.85, 0.75])
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
        """Opens a file dialog, prompting the user to select the data (Excel spreadsheet) to load."""
        query, _ = QFileDialog.getOpenFileName(self, 'Select input file', '', 'Excel Spreadsheet (*.xlsx)')
        if query:
            self.load_data(query=query)

    def load_data(self, query: str) -> None:
        """Loads input data into memory, storing it in the GUI's self.data attribute."""
        try:
            self.data = openpyxl.load_workbook(query, data_only=True)
        except BadZipFile:
            self.raise_error(CorruptedExcelFile('Cannot load input Excel file. Is it corrupted?'))
        else:
            filename = os.path.basename(query)
            self.input_filename_label.setText(filename)
            self.input_sheetname_combobox.setEnabled(True)
            self.input_sheetname_combobox.clear()
            self.input_sheetname_combobox.addItems(self.data.sheetnames)

    def cs1_cs2_setup(self) -> None:
        """Changes interface widgets when the user clicks on the CS1 and CS2 radio button."""
        self.time_interval_label.setEnabled(True)
        self.time_interval_spinbox.setEnabled(True)
        self.CS_label.setText('Initial colony size column')
        self.GR_label.setText('Final colony size column')

    def cs_gr_setup(self) -> None:
        """Changes interface widgets when the user clicks on the CS and GR radio button."""
        self.time_interval_label.setEnabled(False)
        self.time_interval_spinbox.setEnabled(False)
        self.CS_label.setText('Colony size column')
        self.GR_label.setText('Growth rate column')

    def confidence_setup(self) -> None:
        """Changes interface widgets when the user clicks on the confidence interval radio button."""
        if self.add_conf_int_checkbox.isChecked():
            self.conf_int_spinbox.setEnabled(True)
        else:
            self.conf_int_spinbox.setEnabled(False)

    def dynafit_run(self) -> None:
        """Runs DynaFit (main function from logic.py)."""
        self.dynafit_setup()
        try:
            dynafit_settings = self.get_dynafit_settings()
            results = dynafit(**dynafit_settings)
        except Exception as e:
            self.dynafit_raised_exception(e)
        else:
            self.dynafit_no_exceptions_raised(results)
        finally:
            self.dynafit_cleanup()

    def dynafit_setup(self) -> None:
        """Called before DynaFit analysis starts. Modifies the label on the plot button and clears both Axes."""
        self.plot_button.setText('Plotting...')
        self.plot_button.setEnabled(False)
        self.cvp_ax.clear()
        self.histogram_ax.clear()

    def get_dynafit_settings(self) -> Dict[str, Any]:
        """Bundles the information and parameters selected by the user into a single dictionary and then returns it.
        Does not perform any kind of validation (this is delegated to the validator.py module)."""
        return {
            'data': self.data,
            'filename': os.path.splitext(self.input_filename_label.text())[0],
            'sheetname': self.input_sheetname_combobox.currentText(),
            'need_to_calculate_gr': self.cs1_cs2_button.isChecked(),
            'time_delta': self.time_interval_spinbox.value(),
            'cs_start_cell': self.CS_start_textbox.text(),
            'cs_end_cell': self.CS_end_textbox.text(),
            'gr_start_cell': self.GR_start_textbox.text(),
            'gr_end_cell': self.GR_end_textbox.text(),
            'individual_colonies': self.max_individual_cs_spinbox.value(),
            'large_colony_groups': self.large_colony_groups_spinbox.value(),
            'bootstrap_repeats': self.bootstrap_repeats_spinbox.value(),
            'add_confidence_interval': self.add_conf_int_checkbox.isChecked(),
            'confidence_value': self.conf_int_spinbox.value(),
            'remove_outliers': self.remove_outliers_checkbox.isChecked(),
            'add_violin': self.add_violins_checkbox.isChecked(),
            'fig': self.fig,
            'cvp_ax': self.cvp_ax,
            'hist_ax': self.histogram_ax,
        }

    def dynafit_raised_exception(self, error: Exception) -> None:
        """Called if an error is raised during DynaFit analysis. Clears axes and shows the error in a message box."""
        self.cvp_ax.clear()
        self.histogram_ax.clear()
        self.raise_error(error)

    def dynafit_no_exceptions_raised(self, results: Tuple[Dict[str, Any], np.array, np.array]) -> None:
        """Called if no errors are raised during the DynaFit analysis. Writes/saves the results from DynaFit."""
        self.to_excel_button.setEnabled(True)
        self.to_csv_button.setEnabled(True)
        self.results_table.clearContents()
        size = max(len(r) for r in results)
        self.results_table.setRowCount(size)
        result_dict, xs, ys = results
        self.set_results_table(result_dict=result_dict, xs=xs, ys=ys)
        self.set_results_dataframe(result_dict=result_dict, xs=xs, ys=ys, size=size)

    def set_results_table(self, result_dict: Dict[str, Any], xs: np.ndarray, ys: np.ndarray) -> None:
        """Sets values obtained from DynaFit analysis onto results table."""
        for row_index, ((name, value), x, y) in enumerate(zip_longest(result_dict.items(), xs, ys, fillvalue='')):
            for column_index, element in enumerate([name, value, x, y]):
                self.results_table.setItem(row_index, column_index, QTableWidgetItem(str(element)))

    def set_results_dataframe(self, result_dict: Dict[str, Any], xs: np.ndarray, ys: np.ndarray,  size: int) -> None:
        """Saves DynaFit results as a pandas DataFrame (used for Excel/csv export)."""
        params = list(result_dict.keys()) + [None for _ in range(size - len(result_dict))]
        values = list(result_dict.values()) + [None for _ in range(size - len(result_dict))]
        xs = list(xs) + [None for _ in range(size - len(xs))]
        ys = list(ys) + [None for _ in range(size - len(ys))]
        self.results = pd.DataFrame({'Parameter': params, 'Value': values, 'Mean X Values': xs, 'Mean Y Values': ys})

    def dynafit_cleanup(self) -> None:
        """Called when DynaFit analysis finishes running (regardless of errors). Restores label on the plot
        button and removes the axis lines from the histogram"""
        self.plot_button.setText('Plot CVP')
        self.plot_button.setEnabled(True)
        self.histogram_ax.set_axis_off()
        self.canvas.draw()

    def save_to_excel_dialog(self) -> None:
        """Opens a file dialog, prompting the user to select the name/location for the Excel export of the results."""
        if self.results is None:
            self.raise_error(ValueError('No results yet. Please click on the "Plot CVP" button first.'))
            return
        placeholder = f'{self.results_table.item(0, 1).text()}_{self.results_table.item(1, 1).text()}.xlsx'
        query, _ = QFileDialog.getSaveFileName(self, 'Select file to save results', placeholder,
                                               'Excel Spreadsheet (*.xlsx)')
        if query:
            self.save_excel(path=query, placeholder=placeholder)

    def save_excel(self, path: str, placeholder: str) -> None:
        """Saves the DynaFit results to the given path as an Excel spreadsheet."""
        if not path.endswith('.xlsx'):
            path = path + '.xlsx'
        self.results.to_excel(path, index=None, sheet_name=placeholder)

    def save_to_csv_dialog(self) -> None:
        """Opens a file dialog, prompting the user to select the name/location for the csv export of the results."""
        if self.results is None:
            self.raise_error(ValueError('No results yet. Please click on the "Plot CVP" button first.'))
            return
        placeholder = f'{self.results_table.item(0, 1).text()}_{self.results_table.item(1, 1).text()}.csv'
        query, _ = QFileDialog.getSaveFileName(self, 'Select file to save results', placeholder,
                                               'Comma-separated values (*.csv)')
        if query:
            self.save_csv(path=query)

    def save_csv(self, path: str) -> None:
        """Saves the DynaFit results to the given path as a csv file."""
        if not path.endswith('.csv'):
            path = path + '.csv'
        self.results.to_csv(path, index=None)

    def small_sample_warning(self, sample_dict: Dict[str, int]) -> None:
        pass  # TODO: implement warning (GUI blocking) when a bin contains a small number of colonies!

    def raise_error(self, error: Exception) -> None:
        """Generic function for catching errors and re-raising them as properly formatted message boxes."""
        name = f'{error.__class__.__name__}:\n{error}'
        trace = traceback.format_exc()
        self.show_error_message((name, trace))

    def show_error_message(self, error_tuple: Tuple[str, str]) -> None:
        """Shows a given error as a message box in front of the GUI."""
        error, trace = error_tuple
        box = QMessageBox(self, windowTitle='An error occurred!', text=error, detailedText=trace)
        box.show()

    def debug(self) -> None:
        """Implemented for easier debugging."""
        self.load_data(query='data/example.xlsx')
        self.CS_start_textbox.setText('A2')
        self.GR_start_textbox.setText('B2')
        self.cs_gr_button.setChecked(True)
        self.cs_gr_setup()

    # The following methods allows the result table to be copied to the clipboard. Source:
    # https://stackoverflow.com/questions/40469607/

    def eventFilter(self, source: QWidget, event: QEvent) -> bool:
        """Event filter for results table."""
        if event.type() == QEvent.KeyPress and event.matches(QKeySequence.Copy):
            self.copy_selection()
            return True
        return super().eventFilter(source, event)

    def copy_selection(self) -> None:
        """Copies selection on the results table to the clipboard (csv-formatted)."""
        selected_indexes = self.results_table.selectedIndexes()
        if selected_indexes:
            rows = sorted(index.row() for index in selected_indexes)
            columns = sorted(index.column() for index in selected_indexes)
            row_count = rows[-1] - rows[0] + 1
            column_count = columns[-1] - columns[0] + 1
            table = [[''] * column_count for _ in range(row_count)]
            for index in selected_indexes:
                row = index.row() - rows[0]
                column = index.column() - columns[0]
                table[row][column] = index.data()
            stream = StringIO()
            writer(stream).writerows(table)
            qApp.clipboard().setText(stream.getvalue())


class CorruptedExcelFile(Exception):
    """Exception raised when openpyxl cannot parse the input Excel file."""


if __name__ == '__main__':
    app = QApplication()
    dfgui = DynaFitGUI()
    if DEBUG:
        dfgui.debug()
    dfgui.show()
    sys.exit(app.exec_())
