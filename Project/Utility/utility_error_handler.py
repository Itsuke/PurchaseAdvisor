"""
Utility script for error handling in GUI
"""
from PyQt5.QtWidgets import QMessageBox


def show_error_message(informative_text):
    """
    Method to show error message box
    :param string informative_text: error message to be shown
    """
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setText("Error")
    msg.setWindowTitle("Error")
    msg.setInformativeText(informative_text)
    msg.exec_()
