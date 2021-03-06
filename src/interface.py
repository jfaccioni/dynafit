"""interface.py - GUI implementation of DynaFit."""

import os
import sys
import traceback
from csv import writer
from io import StringIO
from queue import Queue
from typing import Any, Dict, Tuple
from zipfile import BadZipFile

import openpyxl
import pandas as pd
from PySide2.QtCore import QEvent, QThreadPool, Qt, Slot
from PySide2.QtGui import QKeySequence
from PySide2.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar,
                               QPushButton, QRadioButton, QScrollArea, QSpinBox, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas, NavigationToolbar2QT as Navbar
from matplotlib.pyplot import Figure

from src.core import dynafit
from src.exceptions import AbortedByUser, CorruptedExcelFile
from src.plotter import Plotter
from src.worker import Worker


class DynaFitGUI(QMainWindow):
    """Class representing the DynaFit GUI as a whole."""
    def __init__(self) -> None:
        """Init method of DynaFitGUI class. Sets up the entire interface."""
        super().__init__(parent=None)

        # ### MAIN SETUP

        # --Application attributes--
        self.threadpool = QThreadPool(self)
        self.workbook = None
        self.results_dataframe = None
        self.cumulative_hypothesis_data = None
        self.endpoint_hypothesis_data = None
        self.save_dir = ''

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
        self.load_data_button.clicked.connect(self.load_data_dialog)  # noqa
        self.input_frame.layout().addWidget(self.load_data_button, 0, 0, 1, 2)  # noqa
        # Input filename and label
        self.input_filename_helper_label = QLabel(self, text='Data loaded:', styleSheet='font-weight: 600')
        self.input_frame.layout().addWidget(self.input_filename_helper_label, 1, 0)  # noqa
        self.input_filename_label = QLabel(self, text='')
        self.input_frame.layout().addWidget(self.input_filename_label, 1, 1)  # noqa
        # Input sheetname and label
        self.input_sheetname_label = QLabel(self, text='Sheet to analyse:', styleSheet='font-weight: 600')
        self.input_frame.layout().addWidget(self.input_sheetname_label, 2, 0)  # noqa
        self.input_sheetname_combobox = QComboBox(self)
        self.input_sheetname_combobox.setDisabled(True)
        self.input_frame.layout().addWidget(self.input_sheetname_combobox, 2, 1)  # noqa
        # Add input frame section to left column
        left_column.addWidget(self.input_frame)

        # --Data type/range frame--
        # Region where user selects whether data represents CS1+CS2 or CS+GR and the respective ranges
        left_column.addWidget(QLabel(self, text='Data type and range', styleSheet='font-weight: 600'))
        self.data_type_range_frame = QFrame(self, frameShape=QFrame.Box)
        self.data_type_range_frame.setLayout(QGridLayout())
        # CS1/CS2 button (calculates GR)
        self.cs1_cs2_button = QRadioButton(self, text='Initial and final colony sizes', checked=True)
        self.cs1_cs2_button.clicked.connect(self.cs1_cs2_button_clicked)  # noqa
        self.data_type_range_frame.layout().addWidget(self.cs1_cs2_button, 0, 0, 1, 2)  # noqa
        # CS/GR button (pre-calculated GR)
        self.cs_gr_button = QRadioButton(self, text='Initial colony size and growth rate')
        self.cs_gr_button.clicked.connect(self.cs_gr_button_clicked)  # noqa
        self.data_type_range_frame.layout().addWidget(self.cs_gr_button, 0, 2, 1, 2)  # noqa
        # Time interval label and value
        self.time_interval_label = QLabel(self, text='Hours between initial and final colony sizes:', wordWrap=True)
        self.data_type_range_frame.layout().addWidget(self.time_interval_label, 1, 0, 1, 2)  # noqa
        self.time_interval_spinbox = QDoubleSpinBox(self, minimum=0.0, value=24.0, maximum=1000.0, singleStep=1.0)
        self.data_type_range_frame.layout().addWidget(self.time_interval_spinbox, 1, 2, 1, 2)  # noqa
        # CS label and GR label
        self.CS_label = QLabel(self, text='Initial colony size column', styleSheet='font-weight: 600')
        self.data_type_range_frame.layout().addWidget(self.CS_label, 2, 0, 1, 2)  # noqa
        self.GR_label = QLabel(self, text='Final colony size column', styleSheet='font-weight: 600')
        self.data_type_range_frame.layout().addWidget(self.GR_label, 2, 2, 1, 2)  # noqa
        # CS start
        self.CS_start_label = QLabel(self, text='From cell:')
        self.data_type_range_frame.layout().addWidget(self.CS_start_label, 3, 0)  # noqa
        self.CS_start_textbox = QLineEdit(self, placeholderText="")
        self.data_type_range_frame.layout().addWidget(self.CS_start_textbox, 3, 1)  # noqa
        # CS end
        self.CS_end_label = QLabel(self, text='To cell:')
        self.data_type_range_frame.layout().addWidget(self.CS_end_label, 4, 0)  # noqa
        self.CS_end_textbox = QLineEdit(self, placeholderText="Entire Column")
        self.data_type_range_frame.layout().addWidget(self.CS_end_textbox, 4, 1)  # noqa
        # GR start
        self.GR_start_label = QLabel(self, text='From cell:')
        self.data_type_range_frame.layout().addWidget(self.GR_start_label, 3, 2)  # noqa
        self.GR_start_textbox = QLineEdit(self, placeholderText="")
        self.data_type_range_frame.layout().addWidget(self.GR_start_textbox, 3, 3)  # noqa
        # GR end
        self.GR_end_label = QLabel(self, text='To cell:')
        self.data_type_range_frame.layout().addWidget(self.GR_end_label, 4, 2)  # noqa
        self.GR_end_textbox = QLineEdit(self, placeholderText="Entire Column")
        self.data_type_range_frame.layout().addWidget(self.GR_end_textbox, 4, 3)  # noqa
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
        self.options_frame.layout().addWidget(self.max_individual_cs_label, 0, 0, 1, 1)  # noqa
        self.max_individual_cs_spinbox = QSpinBox(self, minimum=0, value=5, maximum=100, singleStep=1)
        self.max_individual_cs_spinbox.setToolTip(tooltip)
        self.options_frame.layout().addWidget(self.max_individual_cs_spinbox, 0, 1, 1, 1)  # noqa
        # Number of large colony groups
        tooltip = 'Remaining colonies are equally distributed in these many groups'
        self.large_colony_groups_label = QLabel(self, text='Large colony groups')
        self.large_colony_groups_label.setToolTip(tooltip)
        self.options_frame.layout().addWidget(self.large_colony_groups_label, 0, 2, 1, 1)  # noqa
        self.large_colony_groups_spinbox = QSpinBox(self, minimum=0, value=5, maximum=100, singleStep=1)
        self.large_colony_groups_spinbox.setToolTip(tooltip)
        self.options_frame.layout().addWidget(self.large_colony_groups_spinbox, 0, 3, 1, 1)  # noqa
        # Number of bootstrap repeats parameter
        self.bootstrap_repeats_label = QLabel(self, text='Bootstrap repeats')
        self.options_frame.layout().addWidget(self.bootstrap_repeats_label, 1, 0, 1, 1)  # noqa
        self.bootstrap_repeats_spinbox = QSpinBox(self, minimum=0, maximum=1_000_000, singleStep=1)
        self.bootstrap_repeats_spinbox.setValue(100)
        self.options_frame.layout().addWidget(self.bootstrap_repeats_spinbox, 1, 1, 1, 1)  # noqa
        # Confidence interval parameter
        self.conf_int_checkbox = QCheckBox(self, text='Use conf. int.')
        self.conf_int_checkbox.clicked.connect(self.conf_int_checkbox_clicked)  # noqa
        self.options_frame.layout().addWidget(self.conf_int_checkbox, 1, 2, 1, 1)  # noqa
        self.conf_int_spinbox = QDoubleSpinBox(self, minimum=0, value=0.95, maximum=0.999, singleStep=0.01, decimals=3)
        self.conf_int_spinbox.setEnabled(False)
        self.options_frame.layout().addWidget(self.conf_int_spinbox, 1, 3, 1, 1)  # noqa
        # Filter outliers parameter
        self.remove_outliers_checkbox = QCheckBox(self, text='Filter outliers before plotting')
        self.options_frame.layout().addWidget(self.remove_outliers_checkbox, 2, 0, 1, 2)  # noqa
        # Plot violins parameter
        self.add_violins_checkbox = QCheckBox(self, text='Add violins to plot')
        self.options_frame.layout().addWidget(self.add_violins_checkbox, 2, 2, 1, 2)  # noqa
        # Add section above to left column
        left_column.addWidget(self.options_frame)

        # --Plot frame--
        # Region where the button to plot is located, as well as the calculated AAC
        plot_grid = QGridLayout()
        # Plot button
        self.plot_button = QPushButton(self, text='Plot CVP')
        self.plot_button.clicked.connect(self.dynafit_run)  # noqa
        plot_grid.addWidget(self.plot_button, 0, 0, 1, 1)
        # Excel export button
        self.to_excel_button = QPushButton(self, text='Save to Excel')
        self.to_excel_button.clicked.connect(self.save_excel_dialog)  # noqa
        self.to_excel_button.setDisabled(True)
        plot_grid.addWidget(self.to_excel_button, 0, 1, 1, 1)
        # CSV export button
        self.to_csv_button = QPushButton(self, text='Save to csv')
        self.to_csv_button.clicked.connect(self.save_csv_dialog)  # noqa
        self.to_csv_button.setDisabled(True)
        plot_grid.addWidget(self.to_csv_button, 0, 2, 1, 1)
        # Progress bar and label
        self.progress_bar_label = QLabel(self, text='Progress:')
        self.progress_bar_label.setHidden(True)
        plot_grid.addWidget(self.progress_bar_label, 0, 3, 1, 1)
        self.progress_bar = QProgressBar(self, minimum=0, maximum=100)
        self.progress_bar.setHidden(True)
        plot_grid.addWidget(self.progress_bar, 0, 4, 1, 1)
        # Add section above to left column
        left_column.addLayout(plot_grid)

        # --Results table--
        # Region where the results from the DynaFit analysis are shown
        self.results_table = QTableWidget(self)
        self.results_table.installEventFilter(self)
        # Add widget above to left column
        left_column.addWidget(self.results_table)

        # ### RIGHT COLUMN
        # --Scroll Area--
        # Box with a vertical scroll bar which contains the canvas
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.horizontalScrollBar().setEnabled(False)
        # --Plot canvas--
        # Region where plots are displayed
        self.fig = Figure(facecolor='white', figsize=(12, 12))
        self.canvas = Canvas(self.fig)
        self.canvas.setParent(self)
        self.scroll_area.setWidget(self.canvas)
        # Axes instance used to plot CVP
        self.cvp_ax = self.fig.add_axes([0.1, 0.45, 0.85, 0.5], label='cvp')
        self.cvp_ax.set_visible(False)
        # Axes instance used to plot hypothesis distance (closeness to H0 and H1)
        self.hypothesis_ax = self.fig.add_axes([0.1, 0.25, 0.85, 0.1], label='hypothesis')
        self.hypothesis_ax.set_visible(False)
        # Axes instance used to plot population histogram
        self.histogram_ax = self.fig.add_axes([0.1, 0.05, 0.85, 0.1], label='histogram')
        self.histogram_ax.set_visible(False)
        # Add canvas above to right column
        right_column.addWidget(self.scroll_area)
        # --Canvas footer--
        # Region where canvas navigation bar and related buttons are displayed
        canvas_footer_layout = QHBoxLayout()
        # Button for toggling display of cumulative hypothesis
        cumulative_hypothesis_button = QPushButton(self, text='Toggle cumulative hypothesis plot')
        cumulative_hypothesis_button.clicked.connect(self.toggle_cumulative_hypothesis)  # noqa
        canvas_footer_layout.addWidget(cumulative_hypothesis_button)
        # Button for toggling display of endpoint hypothesis
        endpoint_hypothesis_button = QPushButton(self, text='Toggle endpoint hypothesis plot')
        endpoint_hypothesis_button.clicked.connect(self.toggle_endpoint_hypothesis)  # noqa
        canvas_footer_layout.addWidget(endpoint_hypothesis_button)
        # Canvas navigation bar
        canvas_footer_layout.addWidget(Navbar(self.canvas, self.frame))
        # Add widgets above to right column
        right_column.addLayout(canvas_footer_layout)

        # Set column stretch so that only plot gets rescaled with GUI
        columns.setStretch(1, 10)
        # Maximizes GUI on startup
        self.showMaximized()

    def load_data_dialog(self) -> None:
        """Opens a file dialog, prompting the user to select the data (Excel spreadsheet) to load."""
        query, _ = QFileDialog.getOpenFileName(self, 'Select input file', '', 'Excel Spreadsheet (*.xlsx)')
        if query:
            self.load_data(query=query)

    def load_data(self, query: str) -> None:
        """Loads input data into memory, storing it in the GUI's self.data attribute."""
        try:
            self.workbook = openpyxl.load_workbook(query, data_only=True)
        except BadZipFile:
            self.raise_main_thread_error(CorruptedExcelFile('Cannot load input Excel file. Is it corrupted?'))
        else:
            filename = os.path.basename(query)
            self.load_data_success(filename=filename)

    def load_data_success(self, filename: str) -> None:
        """Runs upon successfully loading an Excel file into the GUI."""
        self.input_filename_label.setText(filename)
        self.input_sheetname_combobox.setEnabled(True)
        self.input_sheetname_combobox.clear()
        self.input_sheetname_combobox.addItems(self.workbook.sheetnames)

    def cs1_cs2_button_clicked(self) -> None:
        """Changes interface widgets when the user clicks on the CS1 and CS2 radio button."""
        self.time_interval_label.setEnabled(True)
        self.time_interval_spinbox.setEnabled(True)
        self.CS_label.setText('Initial colony size column')
        self.GR_label.setText('Final colony size column')

    def cs_gr_button_clicked(self) -> None:
        """Changes interface widgets when the user clicks on the CS and GR radio button."""
        self.time_interval_label.setEnabled(False)
        self.time_interval_spinbox.setEnabled(False)
        self.CS_label.setText('Colony size column')
        self.GR_label.setText('Growth rate column')

    def conf_int_checkbox_clicked(self) -> None:
        """Changes interface widgets when the user clicks on the confidence interval radio button."""
        if self.conf_int_checkbox.isChecked():
            self.conf_int_spinbox.setEnabled(True)
        else:
            self.conf_int_spinbox.setEnabled(False)

    def dynafit_run(self) -> None:
        """Runs DynaFit analysis on a worker thread."""
        try:
            self.dynafit_setup_before_running()
            dynafit_settings = self.get_dynafit_settings()
        except Exception as e:
            self.raise_main_thread_error(e)
        else:
            self.dynafit_worker_run(dynafit_settings=dynafit_settings)

    def get_dynafit_settings(self) -> Dict[str, Any]:
        """Bundles the information and parameters selected by the user into a single dictionary and then returns it.
        Does not perform any kind of validation (this is delegated to the validator.py module)."""
        return {
            'workbook': self.workbook,
            'filename': os.path.splitext(self.input_filename_label.text())[0],
            'sheetname': self.input_sheetname_combobox.currentText(),
            'calculate_growth_rate': self.cs1_cs2_button.isChecked(),
            'time_delta': self.time_interval_spinbox.value(),
            'cs_start_cell': self.CS_start_textbox.text(),
            'cs_end_cell': self.CS_end_textbox.text(),
            'gr_start_cell': self.GR_start_textbox.text(),
            'gr_end_cell': self.GR_end_textbox.text(),
            'individual_colonies': self.max_individual_cs_spinbox.value(),
            'large_groups': self.large_colony_groups_spinbox.value(),
            'bootstrap_repeats': self.bootstrap_repeats_spinbox.value(),
            'show_ci': self.conf_int_checkbox.isChecked(),
            'confidence_value': self.conf_int_spinbox.value(),
            'remove_outliers': self.remove_outliers_checkbox.isChecked(),
            'show_violin': self.add_violins_checkbox.isChecked(),
        }

    def dynafit_worker_run(self, dynafit_settings: Dict[str, Any]) -> None:
        """Runs Worker thread through QThreadPool after instantiating and connecting it."""
        worker = Worker(func=dynafit, **dynafit_settings)
        worker.add_callbacks()
        self.connect_worker(worker=worker)
        self.threadpool.start(worker)

    def connect_worker(self, worker: Worker) -> None:
        """Hooks up worker signals to interface methods."""
        worker.signals.progress.connect(self.dynafit_worker_progress_updated)
        worker.signals.warning.connect(self.dynafit_worker_small_sample_size_warning)
        worker.signals.finished.connect(self.dynafit_worker_has_finished)
        worker.signals.success.connect(self.dynafit_worker_raised_no_exceptions)
        worker.signals.error.connect(self.dynafit_worker_raised_exception)

    def dynafit_setup_before_running(self) -> None:
        """Called before DynaFit analysis starts. Modifies the label on the plot button and clears both Axes."""
        self.cvp_ax.clear()
        self.hypothesis_ax.clear()
        self.histogram_ax.clear()
        self.progress_bar.setHidden(False)
        self.progress_bar_label.setHidden(False)
        self.plot_button.setDisabled(True)
        self.to_excel_button.setDisabled(True)
        self.to_csv_button.setDisabled(True)

    @Slot(int)
    def dynafit_worker_progress_updated(self, number: int) -> None:
        """Updates DynaFit progress"""
        self.progress_bar.setValue(number)

    @Slot(object)
    def dynafit_worker_small_sample_size_warning(self, warning: Tuple[Queue, Dict[int, Tuple[int, int]]]) -> None:
        """Creates a sample size warning message box. Thread is halted while user selects the answer, which is
        sent back to the thread through a Queue object."""
        answer_queue, warning_info = warning
        message = ('Warning: small sample sizes found for some groups. DynaFit analysis may be unreliable or '
                   'impossible to compute.\nDo you want to continue anyway?')
        groups = '\n'.join(f'Group {k}, mean CS {cs}: sample size of {n}' for k, (n, cs) in warning_info.items())
        box = QMessageBox(self, windowTitle='Warning: low sample sizes', text=message, detailedText=groups,
                          standardButtons=QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.No)
        reply = box.exec_()
        if reply == QMessageBox.Yes:
            answer_queue.put(False)
        else:
            answer_queue.put(True)

    def dynafit_worker_raised_exception(self, exception_tuple: Tuple[Exception, str]) -> None:
        """Called if an error is raised during DynaFit analysis. Clears axes and shows the error in a message box."""
        self.fig.suptitle('')
        self.cvp_ax.set_visible(False)
        self.hypothesis_ax.set_visible(False)
        self.histogram_ax.set_visible(False)
        self.results_table.clearContents()
        self.results_dataframe = None
        self.cumulative_hypothesis_data = None
        self.endpoint_hypothesis_data = None
        if not isinstance(exception_tuple[0], AbortedByUser):
            self.raise_worker_thread_error(exception_tuple)

    def dynafit_worker_raised_no_exceptions(self, results: Tuple[Dict[str, Any], Plotter, pd.DataFrame]) -> None:
        """Called if no errors are raised during the DynaFit analysis. Writes/saves the results from DynaFit."""
        self.to_excel_button.setEnabled(True)
        self.to_csv_button.setEnabled(True)
        self.cvp_ax.set_visible(True)
        self.hypothesis_ax.set_visible(True)
        self.histogram_ax.set_visible(True)
        params, plotter, df = results
        plotter.plot_cvp_ax(ax=self.cvp_ax)
        hypothesis_data = plotter.plot_hypothesis_ax(ax=self.hypothesis_ax, xlims=self.cvp_ax.get_xlim())
        self.cumulative_hypothesis_data, self.endpoint_hypothesis_data = hypothesis_data
        plotter.plot_histogram_ax(ax=self.histogram_ax)
        self.set_figure_title(filename=params['filename'], sheetname=params['sheetname'])
        self.set_results_table(df=df)
        self.results_dataframe = self.remove_nan_strings(df=df)

    def set_figure_title(self, filename: str, sheetname: str) -> None:
        """Sets the figure title based on the parameters used in the DynaFit analysis."""
        self.fig.suptitle(f'CVP - Exp: {filename}, Sheet: {sheetname}')

    def set_results_table(self, df: pd.DataFrame) -> None:
        """Sets values obtained from DynaFit analysis onto dataframe_results table."""
        self.results_table.clearContents()
        self.results_table.setRowCount(len(df))
        self.results_table.setColumnCount(len(df.columns))
        for index, column_name in enumerate(df.columns):
            self.results_table.setHorizontalHeaderItem(index, QTableWidgetItem(column_name))
            self.results_table.horizontalHeader().setSectionResizeMode(index, QHeaderView.ResizeToContents)
        string_df = self.remove_nan_strings(df.astype(str))
        for row_index, row_contents in string_df.iterrows():
            for column_index, value in enumerate(row_contents):
                self.results_table.setItem(row_index, column_index, QTableWidgetItem(value))

    @staticmethod
    def remove_nan_strings(df: pd.DataFrame) -> pd.DataFrame:
        """Returns the same DataFrame stored in the self.results_dataframe attribute, but with the strings 'nan'
        replaced by empty strings."""
        return df.replace({'nan': ''}, regex=True)

    def dynafit_worker_has_finished(self) -> None:
        """Called when DynaFit analysis finishes running (regardless of errors). Restores label on the plot
        button and removes the axis lines from the histogram"""
        self.plot_button.setText('Plot CVP')
        self.plot_button.setEnabled(True)
        self.progress_bar_label.setHidden(True)
        self.progress_bar.setHidden(True)
        self.progress_bar.setValue(0)
        self.canvas.draw()

    def save_excel_dialog(self) -> None:
        """Opens a file dialog, prompting the user to select the name/location for the Excel export of the results."""
        filename = self.results_table.item(0, 1).text()
        sheetname = self.results_table.item(1, 1).text()
        placeholder = os.path.join(self.save_dir, f'{filename}_{sheetname}.xlsx')
        query, _ = QFileDialog.getSaveFileName(self, 'Select file to save dataframe_results', placeholder,
                                               'Excel Spreadsheet (*.xlsx)')
        if query:
            self.save_excel(path=query, sheet_name=sheetname)
            self.update_save_dir(query=query)

    def save_excel(self, path: str, sheet_name: str) -> None:
        """Saves the DynaFit dataframe_results to the given path as an Excel spreadsheet."""
        if not path.endswith('.xlsx'):
            path += '.xlsx'
        self.results_dataframe.to_excel(path, index=False, sheet_name=sheet_name)

    def save_csv_dialog(self) -> None:
        """Opens a file dialog, prompting the user to select the name/location for the csv export of the results."""
        filename = self.results_table.item(0, 1).text()
        sheetname = self.results_table.item(1, 1).text()
        placeholder = os.path.join(self.save_dir, f'{filename}_{sheetname}.csv')
        query, _ = QFileDialog.getSaveFileName(self, 'Select file to save dataframe_results', placeholder,
                                               'Comma-separated values (*.csv)')
        if query:
            self.save_csv(path=query)
            self.update_save_dir(query=query)

    def save_csv(self, path: str) -> None:
        """Saves the DynaFit dataframe_results to the given path as a csv file."""
        if not path.endswith('.csv'):
            path += '.csv'
        self.results_dataframe.to_csv(path, index=False)

    def update_save_dir(self, query: str) -> None:
        """Updates the save directory attribute."""
        self.save_dir = os.path.dirname(query)

    def toggle_cumulative_hypothesis(self):
        """Shows/hides the cumulative hypothesis line plot."""
        if self.cumulative_hypothesis_data is None:  # return if the cumulative hypothesis data has not been set yet
            return
        for artist in self.cumulative_hypothesis_data:
            if artist is not None:
                artist.set_visible(False) if artist.get_visible() else artist.set_visible(True)
        self.canvas.draw()

    def toggle_endpoint_hypothesis(self):
        """Shows/hides the endpoint hypothesis line plot."""
        if self.endpoint_hypothesis_data is None:  # return if the endpoint hypothesis data has not been set yet
            return
        for artist in self.endpoint_hypothesis_data:
            if artist is not None:
                artist.set_visible(False) if artist.get_visible() else artist.set_visible(True)
        self.canvas.draw()

    def raise_main_thread_error(self, error: Exception) -> None:
        """Generic function for catching errors in the main GUI thread and re-raising them as properly formatted
        message boxes."""
        name = f'{error.__class__.__name__}:\n{error}'
        trace = traceback.format_exc()
        self.show_error_message(name=name, trace=trace)

    def raise_worker_thread_error(self, exception_tuple: Tuple[Exception, str]) -> None:
        """Generic function for catching errors in the worker thread and re-raising them as properly formatted
        message boxes."""
        error, trace = exception_tuple
        name = f'{error.__class__.__name__}:\n{error}'
        self.show_error_message(name=name, trace=trace)

    def show_error_message(self, name: str, trace: str) -> None:
        """Shows a given error as a message box in front of the GUI."""
        box = QMessageBox(self, windowTitle='An error occurred!', text=name, detailedText=trace)
        box.exec_()

    def resizeEvent(self, e) -> None:
        """Overloaded method that resizes the QScrollArea properly."""
        self.canvas.resize(self.scroll_area.width(), self.canvas.height())
        super().resizeEvent(e)

    # The following methods "eventFilter" and "copy_selection" allows the result table to be copied to the clipboard.
    # Source: https://stackoverflow.com/questions/40469607/

    def eventFilter(self, source: QWidget, event: QEvent) -> bool:
        """Event filter for dataframe_results table."""
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
            QApplication.instance().clipboard().setText(stream.getvalue())


def main() -> None:
    """Entry point for dynafit package."""
    app = QApplication()
    dfgui = DynaFitGUI()
    dfgui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
