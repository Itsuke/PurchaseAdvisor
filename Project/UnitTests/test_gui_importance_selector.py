"""
Unit tests to verify gui_importance_selector module.
"""
import unittest
import sys

import pandas as pd
from PyQt5.QtWidgets import QApplication
from gui_impotrance_selector import GuiImportanceSelector

app = QApplication(sys.argv)
PROFIT = 'p'
COST = 'c'


class MyTestCase(unittest.TestCase):
    """
    Unit test class for testing gui_importance_selector module
    """
    def setUp(self):
        """
        setUP method. Prepares all the data for the verification.
        """
        self.columns = ["cat1", "cat2", "cat3", "cat4", "cat5"]
        self.df = pd.DataFrame([
            ["2000", "4250Hz", "15x15", "az64",  "Porsche"],
            ["2500", "3500Hz", "20x20", "asd64", "Nissan"],
            ["2800", "3000Hz", "10x10", "az128", "Audi"],
            ["1000", "2000Hz", "Nie dotyczy",   "ca64",  "Mercedes"],
            ["3500", "",       "20x10", "za128", "Volvo"],
            ["3200", "3800Hz", "25x15", "as128", "Opel"],
            ["2500", "3000Hz", "20x30", "a64",   "Ford"]], columns=self.columns)
        self.gis = GuiImportanceSelector(self.df, self.columns)
        self.gis._column_id = 0
        self.gis._cost_profit_list = []
        self.list_items = []

    def test_initial_ui(self):
        """
        Verify if the GUI has proper values set
        """
        self.assertEqual(self.gis._column_id, 0)
        self.assertEqual(self.gis._columns_size, 5)

        for index in range(self.gis._list_of_list_widgets[0].count()):
            self.list_items.append(self.gis._list_of_list_widgets[0].item(index).text())
        self.assertListEqual(sorted(set(self.df[self.columns[0]])), self.list_items)

    def test_create_automatic_rating(self):
        """
        Verify that the ratings are properly created based on cost/profit indicator
        """
        self.gis._create_automatic_rating(COST)
        self.assertListEqual(self.gis._cost_profit_list, [COST])

        self.gis._create_self_rating()
        self.assertListEqual(self.gis._cost_profit_list, [COST, PROFIT])

        self.gis._create_automatic_rating(COST)
        self.assertListEqual(self.gis._cost_profit_list, [COST, PROFIT, COST])

        self.gis._create_self_rating()
        self.assertListEqual(self.gis._cost_profit_list, [COST, PROFIT, COST, PROFIT])

        self.gis._create_self_rating()
        self.assertListEqual(self.gis._cost_profit_list, [COST, PROFIT, COST, PROFIT, PROFIT])


if __name__ == '__main__':
    unittest.main()
