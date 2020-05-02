"""debug_code.py - run DynaFit analysis on debug mode."""

import sys

from PySide2.QtCore import Qt
from PySide2.QtTest import QTest
from PySide2.QtWidgets import QApplication

from src.interface import DynaFitGUI


def run_debug_interface() -> None:
    """Run DynaFit analysis on debug mode."""
    global gui
    gui.load_data(query='./test/test_cases/interface_test_case.xlsx')
    gui.CS_start_textbox.setText('A2')
    gui.GR_start_textbox.setText('B2')
    gui.cs_gr_button.setChecked(True)
    gui.conf_int_checkbox.setChecked(True)
    gui.add_violins_checkbox.setChecked(True)
    gui.remove_outliers_checkbox.setChecked(True)
    QTest.mouseClick(gui.plot_button, Qt.LeftButton)  # noqa


def debug_show_error_message(name: str, trace: str) -> None:
    """Hooks into the DynaFit GUI in order to display errors to console"""
    global gui, app
    print('ERROR:')
    print(name)
    print(trace)
    gui.destroy()
    app.exit()


if __name__ == '__main__':
    app = QApplication()
    gui = DynaFitGUI()
    gui.show_error_message = debug_show_error_message
    run_debug_interface()
    sys.exit(app.exec_())
