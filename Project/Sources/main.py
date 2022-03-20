"""
Main script to rule them all. In reality, this is the heart of the whole project.
All starts and ends here.
"""
import sys
import numpy as np
import pandas as pd

from gui_category_selector import GuiCategorySelector
from gui_ceneo_collector import GuiCeneoCollector
from gui_impotrance_selector import GuiImportanceSelector
from gui_weight_selector import GuiWeightSelector
from mcda import MCDA
from utility_rank import UtilityNormalization, NormalizationMethod
from PyQt5.QtWidgets import QApplication

PAGE_URL = 'https://www.ceneo.pl/Smartfony'
FILE = ".\..\DataFiles\data.csv"
IS_SAVE = False
COST_AND_PROFIT_ROW = -2
WEIGHTS_ROW = -1
DATA_COLUMN = 1
TOP_RESULTS_TO_PRINT = 10


class GuiMainProgram:
    """
    Main class to run all the system scripts. Collects, processes data and returns the results.
    """
    def __init__(self):
        self._full_data = None
        self._chosen_criteria_data = None
        self._criteria_name_list = None
        self._cost_profit_list = None
        self._weights_list = None
        self._misc_column = None

        self._df_for_csv_save = None

        self._qapp = QApplication(sys.argv)

        self._run_app()

    def _run_app(self):
        """
        Runs the GUI classes one by one processing the data.
        """
        self._run_ceneo_collector()
        self._run_category_selector()
        self._run_importance_selector()
        self._run_weight_selector()

    def _run_ceneo_collector(self):
        """
        Creates and runs the GuiCeneoCollector class. As a result there will be a DataFrame with
        parsed data from Ceneo.
        """
        gcc = GuiCeneoCollector()
        gcc.show()
        self._qapp.exec_()

        self._full_data = gcc.get_collected_data()
        self._misc_column = gcc.get_product_urls()

    def _run_category_selector(self):
        """
        Creates and runs the GuiCategorySelector class. As a result there will be a list of criteria
        to reduce the data
        that will be processed later.
        """
        gcs = GuiCategorySelector(list(self._full_data.columns))
        gcs.show()
        self._qapp.exec_()

        self._criteria_name_list = gcs.get_chosen_criteria_list()
        self._chosen_criteria_data = self._full_data[self._criteria_name_list]

    def _run_importance_selector(self):
        """
        Creates and runs the GuiImportanceSelector class. As a result there will be a new matrix
        with updated data considering the importance of each param. Additionally it return the
        list of cost and profit options for each criteria.
        """
        gis = GuiImportanceSelector(self._chosen_criteria_data, self._criteria_name_list)
        gis.show()
        self._qapp.exec_()

        self._output_data = np.array(gis.get_modified_matrix())
        self._cost_profit_list = gis.get_cost_and_profit_list()
        self._misc_column.append("cost & profit")

        self._df_for_csv_save = gis.get_modified_matrix()
        self._df_for_csv_save.loc[len(self._df_for_csv_save.index)] = self._cost_profit_list

    def _run_weight_selector(self):
        """
        Creates and runs the GuiWeightSelector class. As a result there will be a new vector
        with calculated weight values for each criteria.
        """
        gws = GuiWeightSelector(self._criteria_name_list)
        gws.show()
        self._qapp.exec_()

        self._weights_list = gws.get_weight_list()
        self._misc_column.append("weights")

        self._df_for_csv_save.loc[len(self._df_for_csv_save.index)] = self._weights_list
        self._df_for_csv_save.insert(0, "misc", self._misc_column, True)

    def save_df(self, path):
        """
        Method that saves the final DataFrame to csv file
        :param string path: path for file to be saved
        """
        self._df_for_csv_save.to_csv(path)

    def get_df(self):
        """
        Getter for Data Frame with complete data

        :return DataFrame: Data matrix to be used for further calculations
        """
        return self._df_for_csv_save


if __name__ == "__main__":
    gmp = GuiMainProgram()
    if IS_SAVE:
        gmp.save_df(FILE)
        data = pd.read_csv(FILE, index_col=0).to_numpy()
    else:
        data = gmp.get_df().to_numpy()

    # you can see the saved data structure in data.csv file
    matrix = data[:COST_AND_PROFIT_ROW, DATA_COLUMN:].astype(float)
    profit_cost_list = data[COST_AND_PROFIT_ROW, DATA_COLUMN:].astype(str)
    weights = data[WEIGHTS_ROW, DATA_COLUMN:].astype(int)
    web_links = data[:COST_AND_PROFIT_ROW, :DATA_COLUMN].astype(str)
    un = UtilityNormalization(matrix)
    norm = un.combined_norm(NormalizationMethod.MAXIMUM, profit_cost_list)

    mcda = MCDA(matrix, norm, weights, profit_cost_list)

    topsis_own_values, topsis_own_ranking = np.array(mcda.get_rank_topsis())
    spotis_own_values, spotis_own_ranking = np.array(mcda.get_rank_spotis())
    promethee_own_values, promethee_own_ranking = np.array(mcda.get_rank_promethee_2())

    dict = {}
    for A, B in zip(topsis_own_ranking.astype(int), web_links):
        dict[A] = B[0]

    result_to_print = TOP_RESULTS_TO_PRINT if TOP_RESULTS_TO_PRINT < len(dict) else len(dict)
    print("\nThe best found products based on your data: ")
    for rank in range(1, result_to_print + 1):
        print("Rank", rank, "Link to product", dict[rank])
