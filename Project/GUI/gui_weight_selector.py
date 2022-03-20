"""
Front-end module to handle the weights of criteria
"""
import utility_error_handler as ueh
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit

NUMBER_OF_WEIGHTS_PER_HORIZON_LAYOUT = 5
TEXT_EDIT_SIZE = 25
BUTTON_SIZE = 110


class GuiWeightSelector(QWidget):
    """
    Class responsible for running the window to input the page url and count of products to be
    parsed later by CeneoHandler
    """
    def __init__(self, list_of_criteria):
        """
        Constructor of class GuiWeightSelector. Initializes the UI window.
        :param list list_of_criteria: list with the names of criteria
        """
        self._criteria_names = list_of_criteria
        self._criteria_count = len(list_of_criteria)
        self._weights_list = []
        self._list_of_horizons = []
        self._list_of_labels = []
        self._list_of_line_edits = []

        self._layout_vertical = QVBoxLayout()
        self._label_instruction = QLabel()
        self._button_finish = QPushButton()

        super().__init__()
        self._init_ui()

    def _save_weight_values_and_close(self):
        """
        Method used to save weight and close the GUI window
        """
        for line_edit_id in range(self._criteria_count):
            try:
                self._weights_list.append(int(self._list_of_line_edits[line_edit_id].text()))

            except ValueError:
                ueh.show_error_message("Use the right positive integer or floating numbers !")
                return

        self.close()

    def get_weight_list(self):
        """
        Getter for the weight list

        :return list: list of weights for criteria
        """
        return self._weights_list

    def _init_ui(self):
        """
        Method that initializes the weight selector GUI.
        """
        self.setWindowTitle("Purchase Advisor")

        self._label_instruction.setText("Set positive weights for chosen criteria")
        self._layout_vertical.addWidget(self._label_instruction)

        for horizon_id in range(int(self._criteria_count / NUMBER_OF_WEIGHTS_PER_HORIZON_LAYOUT)+1):
            self._list_of_horizons.append(QHBoxLayout())
            self._layout_vertical.addLayout(self._list_of_horizons[horizon_id])
            for label_id in range(NUMBER_OF_WEIGHTS_PER_HORIZON_LAYOUT):
                widget_id = horizon_id * NUMBER_OF_WEIGHTS_PER_HORIZON_LAYOUT + label_id

                if widget_id >= self._criteria_count:
                    tmp_vertical_lay = QHBoxLayout()
                    tmp_vertical_lay.addWidget(QLabel())
                    self._list_of_horizons[horizon_id].addLayout(tmp_vertical_lay)
                    continue
                self._list_of_labels.append(QLabel())
                self._list_of_line_edits.append(QLineEdit())

                self._list_of_labels[widget_id].setText(self._criteria_names[widget_id] + ": ")
                self._list_of_line_edits[widget_id].setText("1")
                self._list_of_line_edits[widget_id].setAlignment(Qt.AlignCenter)
                self._list_of_line_edits[widget_id].setMaximumWidth(TEXT_EDIT_SIZE)

                tmp_vertical_lay = QHBoxLayout()
                tmp_vertical_lay.addWidget(self._list_of_labels[widget_id])
                tmp_vertical_lay.addWidget(self._list_of_line_edits[widget_id])

                tmp_vertical_lay.setAlignment(self._list_of_labels[widget_id], Qt.AlignLeft)
                tmp_vertical_lay.setAlignment(self._list_of_line_edits[widget_id], Qt.AlignLeft)
                self._list_of_horizons[horizon_id].addLayout(tmp_vertical_lay)

        self._button_finish.setText("Get results")
        self._button_finish.setMaximumWidth(BUTTON_SIZE)

        self._layout_vertical.addWidget(self._button_finish)
        self._layout_vertical.setAlignment(self._button_finish, Qt.AlignCenter)
        self._button_finish.clicked.connect(self._save_weight_values_and_close)

        self.setLayout(self._layout_vertical)
