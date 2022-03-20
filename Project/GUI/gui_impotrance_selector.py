"""
Front-end module to handle the data alternative prioritization type selection
"""
import re
import numpy as np
import pandas as pd
import utility_error_handler as ueh
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QListWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, \
                            QPushButton, QAbstractItemView

BUTTON_SIZE = 110
NUMBER_OF_IMPORTANCE_COLUMNS = 5
PROFIT = 'p'
COST = 'c'


class _DropInList(QListWidget):
    """
    Private class to create and handle the drag and drop QListWidget.
    """
    def __init__(self):
        """
        Constructor of class _DropInList. Enables the functionalities to:
            * accept drops in list
            * drag the elements of list
            * sort the elements in list
            * to not overwrite elements in list
            * to select elements in list
        """
        super().__init__()
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setSortingEnabled(True)
        self.setDragDropOverwriteMode(False)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setDefaultDropAction(Qt.MoveAction)

    def dropEvent(self, q_drop_event):
        """
        Method to handle the drop events
        :param event q_drop_event: pointer to a object that has been dropped
        """
        source_widget = q_drop_event.source()
        items = source_widget.selectedItems()
        print(items)
        for item in items:
            source_widget.takeItem(source_widget.indexFromItem(item).row())
            self.addItem(item)


class GuiImportanceSelector(QWidget):
    """
    Class responsible for creating the interactive window to create a importance vector for each
    criteria
    """
    def __init__(self, matrix, column_names):
        """
        Constructor of class GuiImportanceSelector. Initializes the UI window.
        :param DataFrame matrix: matrix with data to create ratings
        :param list column_names: list with the names of criteria
        """
        self._matrix = matrix.copy()
        self._column_names = column_names
        self._columns_size = self._matrix.shape[1]
        self._column_id = 0
        self._cost_profit_list = []

        self._layout_vertical = QVBoxLayout()
        self._layout_horizontal_list_widget_rating = QHBoxLayout()
        self._layout_horizontal_label_rating = QHBoxLayout()
        self._layout_horizontal_buttons = QHBoxLayout()
        self._list_of_list_widgets = []
        self._list_of_labels = []

        for _ in range(NUMBER_OF_IMPORTANCE_COLUMNS):
            self._list_of_list_widgets.append(_DropInList())
            self._list_of_labels.append(QLabel())

        self._label_instruction = QLabel()
        self._button_self_rank = QPushButton()
        self._button_less_is_better = QPushButton()
        self._button_more_is_better = QPushButton()

        super().__init__()
        self._init_ui()

    def _update_list(self):
        """
        Updates the list view with new data once the previous is cleaned. Closes the window once all
        the alternatives are handled
        """
        if self._column_id < self._columns_size:
            self._list_of_list_widgets[0].addItems(self._matrix[self._column_names[
                self._column_id]].unique())
            self._label_instruction.setText("Choose how to set the importance criteria for: "
                                            + self._column_names[self._column_id])

        else:
            self.close()

    def _clean_list(self):
        """
        Cleans the list widgets and updates the column id
        """
        for list_widget in self._list_of_list_widgets:
            list_widget.clear()
        self._column_id += 1

    def _create_automatic_rating(self, option):
        """
        Method used to automatically create rating order
        :param string option: it can be either 'p' or 'c' that stays for profit and cost, used to
                              determine rating order
        """
        print('Main: %s' % QtCore.QThread.currentThreadId())
        _thread_safty_flag = False
        column_name = self._column_names[self._column_id]
        clean_saved_copy = self._matrix[column_name].copy()
        column = self._matrix[column_name]
        column = [item.replace(" ", "") for item in column]
        column = [''.join(re.findall(r"\d*\,\d+|\d+", item)) for item in column]
        column = [item.replace(",", ".") for item in column]

        self._matrix[column_name] = pd.to_numeric(column)

        if option == COST:
            self._matrix[column_name] = self._matrix[column_name].replace(
                np.NaN, np.min(self._matrix[column_name]))
            self._matrix[column_name] = self._matrix[column_name].replace(
                "Cannot proceed", np.min(self._matrix[column_name]))

        else:
            self._matrix[column_name] = self._matrix[column_name].replace(
                np.NaN, np.max(self._matrix[column_name]))
            self._matrix[column_name] = self._matrix[column_name].replace(
                "Cannot proceed", np.max(self._matrix[column_name]))

        if self._matrix[column_name].isnull().all():
            self._matrix[column_name] = clean_saved_copy
            ueh.show_error_message('You have to create you own ranking')
            return

        self._cost_profit_list.append(option)
        self._clean_list()
        self._update_list()

    def _create_self_rating(self):
        """
        Method used to create own rating order using the drag and drop columns
        """
        column_name = self._column_names[self._column_id]
        for list_widget_index in range(len(self._list_of_list_widgets)):
            for index in range(self._list_of_list_widgets[list_widget_index].count()):
                self._matrix[column_name] = self._matrix[column_name].replace(
                    self._list_of_list_widgets[list_widget_index].item(index).text(),
                    list_widget_index + 1)

        self._matrix[column_name] = pd.to_numeric(self._matrix[column_name])
        self._cost_profit_list.append(PROFIT)
        self._clean_list()
        self._update_list()

    def _init_ui(self):
        """
        Method that initializes the importance selector GUI.
        """
        self.setWindowTitle("Purchase Advisor")

        self._update_list()
        for list_widget in self._list_of_list_widgets:
            self._layout_horizontal_list_widget_rating.addWidget(list_widget)

        label_counter = 0
        label_text = ["Not relevant",
                      "Not very important",
                      "Neutral",
                      "Important",
                      "Very important"]
        for label in self._list_of_labels:
            label.setText(label_text[label_counter])
            self._layout_horizontal_label_rating.addWidget(label)
            self._layout_horizontal_label_rating.setAlignment(label, Qt.AlignCenter)

            label_counter += 1

        self._button_less_is_better.setText("The less the better")
        self._button_less_is_better.setMaximumWidth(BUTTON_SIZE)
        self._button_more_is_better.setText("The more the better")
        self._button_more_is_better.setMaximumWidth(BUTTON_SIZE)
        self._button_self_rank.setText("Accept own ranking")
        self._button_self_rank.setMaximumWidth(BUTTON_SIZE)

        self._layout_horizontal_buttons.addWidget(self._button_less_is_better)
        self._layout_horizontal_buttons.addWidget(self._button_more_is_better)
        self._layout_horizontal_buttons.addWidget(self._button_self_rank)

        self._button_less_is_better.clicked.connect(lambda: self._create_automatic_rating(COST))
        self._button_more_is_better.clicked.connect(lambda: self._create_automatic_rating(PROFIT))
        self._button_self_rank.clicked.connect(self._create_self_rating)

        self._layout_vertical.addWidget(self._label_instruction)
        self._layout_vertical.addLayout(self._layout_horizontal_list_widget_rating)
        self._layout_vertical.addLayout(self._layout_horizontal_label_rating)
        self._layout_vertical.addLayout(self._layout_horizontal_buttons)
        self._layout_vertical.setAlignment(self._label_instruction, Qt.AlignCenter)

        self.setLayout(self._layout_vertical)

    def get_modified_matrix(self):
        """
        Getter for data matrix

        :return DataFrame: matrix of criteria and alternatives
        """
        return self._matrix

    def get_cost_and_profit_list(self):
        """
        Getter for cost/profit list

        :return list: list of extracted criteria and alternatives
        """
        return self._cost_profit_list
