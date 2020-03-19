import sys
import traceback

import matplotlib
import openpyxl
from PySide2.QtWidgets import (QApplication, QComboBox, QFileDialog, QFormLayout, QFrame, QGridLayout, QHBoxLayout,
                               QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas, NavigationToolbar2QT as Navbar
from matplotlib.pyplot import Figure

from src.logic import dynafit

# set a small font for plots
matplotlib.rc('font', size=8)
# set debugging flag
DEBUG = True


class DynaFitGUI(QMainWindow):
    def __init__(self):
        """init"""
        super().__init__()
        self.resize(1080, 680)
        self.data = None

        # ### MAIN SETUP

        # --Central widget--
        self.frame = QWidget(self)
        self.setCentralWidget(self.frame)
        # --Main layout--
        main_layout = QVBoxLayout()
        self.frame.setLayout(main_layout)
        # --Top bar--
        # App header
        top_bar = QVBoxLayout()
        main_layout.addLayout(top_bar)
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

        # ### TOP BAR (TITLE)

        self.title = QLabel(self, text='DynaFit GUI', styleSheet='font-size: 16pt; font-weight: 600')
        top_bar.addWidget(self.title)

        # ### LEFT COLUMN

        # --Input frame--
        # Region where users select worksheet and spreadsheet to analyse
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
        # add section above to left column
        self.input_frame.layout()
        left_column.addWidget(self.input_frame)

        # --Cell range frame--
        # Region where user selects the ranges of cells containing CS and GR data
        self.cell_range_frame = QFrame(self, frameShape=QFrame.Box)
        self.cell_range_frame.setLayout(QFormLayout())
        # CS label
        self.CS_label = QLabel(self, text='Select Colony Size column (blank for entire column)')
        self.cell_range_frame.layout().addRow(self.CS_label, None)
        # CS start
        self.CS_start_label = QLabel(self, text='From:')
        self.CS_start_textbox = QLineEdit(self, placeholderText="A1")
        self.cell_range_frame.layout().addRow(self.CS_start_label, self.CS_start_textbox)
        # CS end
        self.CS_end_label = QLabel(self, text='To:')
        self.CS_end_textbox = QLineEdit(self, placeholderText="None")
        self.cell_range_frame.layout().addRow(self.CS_end_label, self.CS_end_textbox)
        # GR label
        self.GR_label = QLabel(self, text='Select Growth Rate column (blank for entire column)')
        self.cell_range_frame.layout().addRow(self.GR_label, None)
        # GR start
        self.GR_start_label = QLabel(self, text='From:')
        self.GR_start_textbox = QLineEdit(self, placeholderText="A1")
        self.cell_range_frame.layout().addRow(self.GR_start_label, self.GR_start_textbox)
        # GR end
        self.GR_end_label = QLabel(self, text='To:')
        self.GR_end_textbox = QLineEdit(self, placeholderText="None")
        self.cell_range_frame.layout().addRow(self.GR_end_label, self.GR_end_textbox)

        left_column.addWidget(self.cell_range_frame)

        # Options grid
        self.options_grid = QFormLayout()

        self.maxbin_colsize_label = QLabel(self, text='Max binned colony size')
        self.maxbin_colsize_num = QSpinBox(self, minimum=0, value=5, maximum=20, singleStep=1)
        self.options_grid.addRow(self.maxbin_colsize_label, self.maxbin_colsize_num)

        self.nbins_label = QLabel(self, text='Number of bins for remaining population')
        self.nbins_num = QSpinBox(self, minimum=0, value=5, maximum=20, singleStep=1)
        self.options_grid.addRow(self.nbins_label, self.nbins_num)

        self.nruns_label = QLabel(self, text='Number of independent runs to perform')
        self.nruns_num = QSpinBox(self, minimum=0, value=10, maximum=100, singleStep=1)
        self.options_grid.addRow(self.nruns_label, self.nruns_num)

        self.nrepeats_label = QLabel(self, text='Number of repeated samples for each run/CS')
        self.nrepeats_num = QSpinBox(self, minimum=0, value=10, maximum=100, singleStep=1)
        self.options_grid.addRow(self.nrepeats_label, self.nrepeats_num)

        self.samplesize_label = QLabel(self, text='Sample size')
        self.samplesize_num = QSpinBox(self, minimum=0, value=20, maximum=100, singleStep=1)
        self.options_grid.addRow(self.samplesize_label, self.samplesize_num)

        left_column.addLayout(self.options_grid)

        # Plot button
        self.plot_btn = QPushButton(self, text='Generate CVP')
        self.plot_btn.clicked.connect(self.dynafit_run)
        left_column.addWidget(self.plot_btn)

        area_above_curve_layout = QHBoxLayout()
        self.AAC_label = QLabel(self, text='Measured area above curve:')
        self.AAC_num = QLabel(self, text='None')
        area_above_curve_layout.addWidget(self.AAC_label)
        area_above_curve_layout.addWidget(self.AAC_num)
        left_column.addLayout(area_above_curve_layout)


        # CVP canvas
        self.fig = Figure(facecolor="white")
        self.CVP_ax = self.fig.add_axes([0.1, 0.2, 0.85, 0.75])
        self.hist_ax = self.fig.add_axes([0.05, 0.0, 0.9, 0.1])
        self.hist_ax.set_axis_off()
        self.canvas = Canvas(self.fig)
        self.canvas.setParent(self)
        right_column.addWidget(self.canvas)
        right_column.addWidget(Navbar(self.canvas, self.frame))
        columns.setStretch(1, 100)
        main_layout.setStretch(1, 100)

    def load_data_dialog(self):
        """load"""
        query, _ = QFileDialog.getOpenFileName(self, 'Select input file', '', 'Excel Spreadsheet (*.xlsx)')
        if query:
            self.load_data(query=query)

    def load_data(self, query: str) -> None:
        """load"""
        try:
            self.data = openpyxl.load_workbook(query)
        except Exception as e:
            self.show_error(e)
        else:
            self.input_filename.setText(f'{query.split("/")[-1]}')
            self.input_sheetname.clear()
            self.input_sheetname.addItems(self.data.sheetnames)

    def dynafit_run(self):
        self.dynafit_setup()
        try:
            dynafit(**self.get_dynafit_settings())
        except Exception as e:
            self.dynafit_raised_exception(e)
        else:
            self.dynafit_no_exceptions_raised()
        finally:
            self.dynafit_cleanup()

    def get_dynafit_settings(self):
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
            'hist_ax': self.hist_ax,
        }

    def dynafit_setup(self):
        self.CVP_ax.clear()
        self.hist_ax.clear()
        self.plot_btn.setText('Plotting...')
        self.plot_btn.setEnabled(False)

    def dynafit_raised_exception(self, e):
        self.show_error(e)
        self.CVP_ax.clear()
        self.hist_ax.clear()

    def dynafit_no_exceptions_raised(self):
        self.canvas.draw()

    def dynafit_cleanup(self):
        self.plot_btn.setText('Generate CVP')
        self.plot_btn.setEnabled(True)
        self.hist_ax.set_axis_off()

    def show_error(self, error):
        text = f'{error.__class__.__name__}:\n{error}'
        box = QMessageBox(self, windowTitle='An error occurred!', text=text, detailedText=traceback.format_exc())
        box.show()

    def debug(self):
        self.load_data(query='data/Pasta para Ju.xlsx')
        self.CS_start_textbox.setText('A4')
        self.GR_start_textbox.setText('B4')


if __name__ == '__main__':
    app = QApplication()
    dfgui = DynaFitGUI()
    if DEBUG:
        dfgui.debug()
    dfgui.show()
    sys.exit(app.exec_())
