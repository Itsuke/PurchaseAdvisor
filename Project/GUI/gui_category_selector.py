"""
Front-end module to handle the criteria selection
"""
import utility_error_handler as ueh
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget, \
                            QAbstractItemView


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


class GuiCategorySelector(QWidget):
    """
    Class responsible for running the interactive drag and drop window for user to select
    categories
    """
    def __init__(self, category_list):
        """
        Constructor of class GuiCategorySelector. Initializes the UI window.
        :param list category_list: list of categories for found products
        """
        self._chosen_criteria_list = []
        self._category_list = category_list
        self._layout_vertical = QVBoxLayout()
        self._layout_horizontal = QHBoxLayout()

        self._list_widget_all_categories = _DropInList()
        self._list_widget_selected_categories = _DropInList()
        self._label_instruction = QLabel()
        self._button_next_window = QPushButton()

        super().__init__()
        self._init_ui()

    def get_chosen_criteria_list(self):
        """
        Returns the chosen by user criteria list

        :return list: list of criteria chosen by user
        """
        return self._chosen_criteria_list

    def _collect_data_and_close_window(self):
        """
        Collects all the data from the list view on the right
        """
        if self._list_widget_selected_categories.count() == 0:
            ueh.show_error_message("No criteria has been selected")
            return

        for index in range(self._list_widget_selected_categories.count()):
            self._chosen_criteria_list.append(
                self._list_widget_selected_categories.item(index).text())

        self.close()

    def _init_ui(self):
        """
        Initializes the GUI for criteria selection
        """
        self.setWindowTitle("Purchase Advisor")

        self._list_widget_all_categories.addItems(self._category_list)
        self._layout_horizontal.addWidget(self._list_widget_all_categories)
        self._layout_horizontal.addWidget(self._list_widget_selected_categories)

        self._label_instruction.setText(
            "Choose categories of your interest by draging them to the window on the right side")

        self._button_next_window.setText("Next")
        self._button_next_window.setMaximumWidth(100)
        self._button_next_window.clicked.connect(self._collect_data_and_close_window)

        self._layout_vertical.addWidget(self._label_instruction)
        self._layout_vertical.addLayout(self._layout_horizontal)
        self._layout_vertical.addWidget(self._button_next_window)
        self._layout_vertical.setAlignment(self._button_next_window, Qt.AlignCenter)

        self.setLayout(self._layout_vertical)
