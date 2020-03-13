import random
import sys

from PySide2.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as Canvas, NavigationToolbar2QT as NavToolbar)
from matplotlib.pyplot import Figure


class MyMainWindow(QMainWindow):
    def __init__(self, parent=None):
        """Init"""
        super(MyMainWindow, self).__init__(parent)
        frame = QWidget(self)
        self.fig = Figure(facecolor="white")
        self.ax = self.fig.add_subplot(111)
        self.canvas = Canvas(self.fig)
        self.canvas.setParent(frame)
        self.btn = QPushButton(self, text='Generate random plot')
        self.btn.clicked.connect(self.plot_random)
        vbox = QVBoxLayout()
        vbox.addWidget(self.btn)
        vbox.addWidget(self.canvas)
        vbox.addWidget(NavToolbar(self.canvas, frame))
        frame.setLayout(vbox)
        self.setCentralWidget(frame)

    def plot_random(self):
        """plot"""
        self.ax.clear()

        self.ax.plot([random.random(), random.random()], [random.random(), random.random()])
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication()
    MainWindow = MyMainWindow()
    MainWindow.show()
    sys.exit(app.exec_())
