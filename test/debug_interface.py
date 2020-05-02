"""debug_interface.py - run DynaFit interface on debug mode."""

import sys

from PySide2.QtWidgets import QApplication

from src.interface import DynaFitGUI


def run_debug_interface() -> None:
    """Run DynaFit interface on debug mode."""
    global gui
    gui.load_data(query='./test/test_cases/interface_test_case.xlsx')
    gui.CS_start_textbox.setText('A2')
    gui.GR_start_textbox.setText('B2')
    gui.cs_gr_button.setChecked(True)
    gui.cs_gr_button_clicked()


if __name__ == '__main__':
    app = QApplication()
    gui = DynaFitGUI()
    run_debug_interface()
    gui.show()
    sys.exit(app.exec_())
