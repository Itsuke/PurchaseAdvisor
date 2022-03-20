"""
Front-end module to handle the data collecting process
"""
import utility_error_handler as ueh
from PyQt5 import QtCore
from PyQt5.QtWidgets import QMainWindow, QLineEdit, QProgressBar, QPushButton, QLabel
from ceneo_handler import CeneoHandler

CENEO_PAGE_URL = "www.ceneo.pl"


class GuiCeneoCollector(QMainWindow):
    """
    Class responsible for running the window to input the page url and count of products to be
    parsed later by CeneoHandler
    """
    def __init__(self):
        """
        Constructor of class GuiCeneoCollector
        """
        super().__init__()
        self._full_data = None
        self._products = None
        self._textbox_ceneo_page = QLineEdit(self)
        self._textbox_product_count = QLineEdit(self)
        self._progress_bar_search_pages = QProgressBar(self)
        self._button_search = QPushButton(self)
        self._label_instruction_link = QLabel(self)
        self._label_instruction_count = QLabel(self)

        self._init_ui()

    def _init_ui(self):
        """
        Initializes the GUI for data collecting. Initializes the UI window.
        """
        self.setGeometry(200, 200, 300, 150)
        self.setWindowTitle("Purchase Advisor")

        self._textbox_ceneo_page.setGeometry(QtCore.QRect(10, 30, 281, 20))
        self._textbox_ceneo_page.setText("https://www.ceneo.pl/Smartfony")

        self._textbox_product_count.setGeometry(QtCore.QRect(10, 70, 101, 20))
        self._textbox_product_count.setText("2")

        self._progress_bar_search_pages.setGeometry(QtCore.QRect(10, 100, 291, 21))
        self._progress_bar_search_pages.setProperty("value", 0)

        self._button_search.setGeometry(QtCore.QRect(210, 70, 81, 21))
        self._button_search.setText("Search")
        self._button_search.clicked.connect(self._validate_and_search_for_data_then_close_window)

        self._label_instruction_link.setGeometry(QtCore.QRect(10, 10, 291, 16))
        self._label_instruction_link.setText("Input filtered web page with products from Ceneo.pl")

        self._label_instruction_count.setGeometry(QtCore.QRect(10, 50, 291, 16))
        self._label_instruction_count.setText("How many products do you want to compare? "
                                              "(2 to 1500)")

    def _validate_and_search_for_data_then_close_window(self):
        """
        Validates if the input data is correct. Creates the class object CeneoHandler and runs it.
        Kills the window once the work is done
        """
        if self._textbox_ceneo_page.text() == "" or self._textbox_product_count.text() == "":
            ueh.show_error_message("Data is missing")

        elif CENEO_PAGE_URL not in self._textbox_ceneo_page.text():
            ueh.show_error_message("The given page is missing the: " + CENEO_PAGE_URL)

        else:
            try:
                product_count = int(float(self._textbox_product_count.text()))

            except ValueError:
                ueh.show_error_message("There was a problem while casting the number of pages to"
                                       "integer")
                return
            if product_count < 2:
                ueh.show_error_message("The given number of products is to low")

            elif product_count > 1500:
                ueh.show_error_message("The given number of products is to high")

        ceneo_handler = CeneoHandler(self._textbox_ceneo_page.text(),
                                     product_count,
                                     self._progress_bar_search_pages)

        self._full_data = ceneo_handler.get_collected_data()
        self._products = ceneo_handler.get_product_urls()
        self.close()

    def get_collected_data(self):
        """
        Returns the extracted specs data

        :return DataFrame: list of extracted criteria and alternatives
        """
        return self._full_data

    def get_product_urls(self):
        """
        Returns the list of urls for each product

        :return list: product urls
        """
        return self._products
