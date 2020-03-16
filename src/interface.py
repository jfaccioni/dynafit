import sys
from random import random

import openpyxl
from PySide2.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QPushButton,
                               QVBoxLayout, QWidget)
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as Canvas, NavigationToolbar2QT as Navbar)
from matplotlib.pyplot import Figure


class DynaFitGUI(QMainWindow):
    def __init__(self):
        """init"""
        super().__init__()
        self.data = None

        frame = QWidget(self)
        self.setCentralWidget(frame)
        main_layout = QVBoxLayout()
        frame.setLayout(main_layout)

        top_bar = QVBoxLayout()
        columns = QHBoxLayout()
        main_layout.addLayout(top_bar)
        main_layout.addLayout(columns)

        self.input_btn = QPushButton(self, text='Load input')
        self.input_btn.clicked.connect(self.load_input_file)
        self.input_filename = QLabel(self, text='data loaded: none')
        top_bar.addWidget(self.input_btn)
        top_bar.addWidget(self.input_filename)

        left_column = QVBoxLayout()
        columns.addLayout(left_column)

        self.plot_btn = QPushButton(self, text='Generate random plot')
        self.plot_btn.clicked.connect(self.plot_random)
        left_column.addWidget(self.plot_btn)

        right_column = QVBoxLayout()
        columns.addLayout(right_column)

        self.fig = Figure(facecolor="white")
        self.ax = self.fig.add_subplot(111)
        self.canvas = Canvas(self.fig)
        right_column.addWidget(self.canvas)
        right_column.addWidget(Navbar(self.canvas, frame))

    def load_input_file(self):
        """load"""
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        query, _ = QFileDialog.getOpenFileName(self, "Select input file", "", "All Files (*)", options=options)
        if query != '':
            self.data = openpyxl.load_workbook(query)
            self.input_filename.setText(f'data loaded: {query.split("/")[-1]}')
        else:
            print('no file loaded')
            self.data = None
            self.input_filename = 'data loaded: none'

    def plot_random(self):
        """plot"""
        self.ax.clear()
        self.ax.plot([random(), random(), random(), random(), random(), random()],
                     [random(), random(), random(), random(), random(), random()])
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication()
    dfgui = DynaFitGUI()
    dfgui.show()
    sys.exit(app.exec_())
