import sys

import openpyxl
from PySide2.QtWidgets import (QApplication, QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QMainWindow, QPushButton, QSpinBox, QVBoxLayout, QWidget)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas, NavigationToolbar2QT as Navbar
from matplotlib.pyplot import Figure
import matplotlib; matplotlib.rc('font', size=8)

from src.logic import main as dynafit


class DynaFitGUI(QMainWindow):
    def __init__(self):
        """init"""
        super().__init__()
        self.resize(1080, 680)
        self.data = None

        self.frame = QWidget(self)
        self.setCentralWidget(self.frame)
        main_layout = QVBoxLayout()
        self.frame.setLayout(main_layout)

        top_bar = QVBoxLayout()
        columns = QHBoxLayout()
        main_layout.addLayout(top_bar)
        main_layout.addLayout(columns)

        # Top bar
        self.title = QLabel(self, text="DynaFit GUI")
        self.title.setStyleSheet('QLabel {font-size: 18pt; font-weight: 600}')
        top_bar.addWidget(self.title)

        # Left column (options)
        left_column = QVBoxLayout()
        columns.addLayout(left_column)

        # Input fields
        self.input_btn = QPushButton(self, text='Load input')
        self.input_btn.clicked.connect(self.load_input_file)
        self.input_filename = QLabel(self, text='data loaded: none')
        left_column.addWidget(self.input_btn)
        left_column.addWidget(self.input_filename)

        sheetname_layout = QHBoxLayout()
        self.input_sheetname_label = QLabel(self, text='Sheetname to analyse')
        self.input_sheetname = QComboBox(self)
        self.input_sheetname.addItem('No data yet')
        sheetname_layout.addWidget(self.input_sheetname_label)
        sheetname_layout.addWidget(self.input_sheetname)
        left_column.addLayout(sheetname_layout)

        # CS Columns
        self.CS_label = QLabel(self, text='Select Colony Size column (blank for entire column)')
        left_column.addWidget(self.CS_label)
        self.CS_layout = QHBoxLayout()
        self.CS_start_label = QLabel(self, text='From:')
        self.CS_start_textbox = QLineEdit(self, placeholderText="A1")
        self.CS_layout.addWidget(self.CS_start_label)
        self.CS_layout.addWidget(self.CS_start_textbox)
        self.CS_end_label = QLabel(self, text='To:')
        self.CS_end_textbox = QLineEdit(self, placeholderText="None")
        self.CS_layout.addWidget(self.CS_end_label)
        self.CS_layout.addWidget(self.CS_end_textbox)
        left_column.addLayout(self.CS_layout)

        # GR Columns
        self.GR_label = QLabel(self, text='Select Growth Rate column (blank for entire column)')
        left_column.addWidget(self.GR_label)
        self.GR_layout = QHBoxLayout()
        self.GR_start_label = QLabel(self, text='From:')
        self.GR_start_textbox = QLineEdit(self, placeholderText="A1")
        self.GR_layout.addWidget(self.GR_start_label)
        self.GR_layout.addWidget(self.GR_start_textbox)
        self.GR_end_label = QLabel(self, text='To:')
        self.GR_end_textbox = QLineEdit(self, placeholderText="None")
        self.GR_layout.addWidget(self.GR_end_label)
        self.GR_layout.addWidget(self.GR_end_textbox)
        left_column.addLayout(self.GR_layout)

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
        self.plot_btn.clicked.connect(self.plot_dynafit)
        left_column.addWidget(self.plot_btn)

        AAC_layout = QHBoxLayout()
        self.AAC_label = QLabel(self, text='Measured area above curve:')
        self.AAC_num = QLabel(self, text='None')
        AAC_layout.addWidget(self.AAC_label)
        AAC_layout.addWidget(self.AAC_num)
        left_column.addLayout(AAC_layout)

        # Right column (plots)
        right_column = QVBoxLayout()
        columns.addLayout(right_column)

        # CVP canvas
        self.fig = Figure(facecolor="white")
        self.CVP_ax = self.fig.add_axes([0.1, 0.2, 0.85, 0.75])
        self.histogram_ax = self.fig.add_axes([0.05, 0.0, 0.9, 0.1])
        self.histogram_ax.set_axis_off()
        self.canvas = Canvas(self.fig)
        self.canvas.setParent(self)
        right_column.addWidget(self.canvas)
        right_column.addWidget(Navbar(self.canvas, self.frame))
        columns.setStretch(1, 100)
        main_layout.setStretch(1, 100)

    def load_input_file(self):
        """load"""
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        query, _ = QFileDialog.getOpenFileName(self, "Select input file", "", "All Files (*)", options=options)
        if query:
            self.data = openpyxl.load_workbook(query)
            self.input_filename.setText(f'data loaded: {query.split("/")[-1]}')
            self.input_sheetname.clear()
            self.input_sheetname.addItems(self.data.sheetnames)
        else:
            print('no file loaded')
            self.data = None
            self.input_filename.setText('data loaded: none')
            self.input_sheetname.clear()
            self.input_sheetname.addItem('No data yet')

    def plot_dynafit(self):
        self.CVP_ax.clear()
        self.histogram_ax.clear()
        self.plot_btn.setText('Plotting...')
        self.plot_btn.setEnabled(False)
        try:
            settings = self.get_gui_settings()
            dynafit(**settings)
            self.histogram_ax.set_axis_off()
            self.canvas.draw()
        except Exception as e:
            print(e)
        finally:
            self.plot_btn.setText('Generate CVP')
            self.plot_btn.setEnabled(True)

    def get_gui_settings(self):
        settings = {'data': self.data, 'sheetname': self.input_sheetname.currentText()}
        ws = settings['data'][settings['sheetname']]
        cs_start_cell = self.CS_start_textbox.text()
        if self.CS_end_textbox.text():
            settings['cs_range'] = f'{cs_start_cell}:{self.CS_end_textbox.text()}'
        else:
            col = ws[cs_start_cell].column_letter
            num = str(ws.max_row)
            settings['cs_range'] = f'{cs_start_cell}:{col+num}'

        gr_start_cell = self.GR_start_textbox.text()
        if self.GR_end_textbox.text():
            settings['gr_range'] = f'{gr_start_cell}:{self.GR_end_textbox.text()}'
        else:
            col = ws[gr_start_cell].column_letter
            num = str(ws.max_row)
            settings['gr_range'] = f'{gr_start_cell}:{col + num}'

        settings['max_binned_colony_size'] = self.maxbin_colsize_num.value()
        settings['bins'] = self.nbins_num.value()
        settings['runs'] = self.nruns_num.value()
        settings['repeats'] = self.nrepeats_num.value()
        settings['sample_size'] = self.samplesize_num.value()
        settings['fig'] = self.fig
        settings['cvp_ax'] = self.CVP_ax
        settings['hist_ax'] = self.histogram_ax

        return settings


if __name__ == '__main__':
    app = QApplication()
    dfgui = DynaFitGUI()
    dfgui.show()
    sys.exit(app.exec_())
