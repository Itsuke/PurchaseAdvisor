"""
Back-end Class to handle the data by preparing tables of correlations.
"""
import numpy as np


class UtilityCorrelations:
    """
    Class used to prepare tables of correlations.
    """
    def __init__(self, matrix):
        """
        Constructor of the UtilityCorrelations class
        :param DataFrame matrix:
        """
        self.matrix = matrix

    def rw_correlation(self):
        """
        Method that prepares a correlation table according to spearman rank correlation coefficient.
        formula source: https://www.ine.pt/revstat/pdf/rs060301.pdf

        :return list:
        """
        criteria_count = self.matrix.shape[1]
        rw_corr = np.zeros([criteria_count, criteria_count])

        for x in range(criteria_count):
            for y in range(criteria_count):
                X = self.matrix[:][x]
                Y = self.matrix[:][y]
                N = len(X)
                sum_val = 0
                for i in range(N):
                    sum_val += (np.power((X[i] - Y[i]), 2)) * ((N - X[i] + 1) + (N - Y[i] + 1))

                rw = 1 - ((6 * sum_val) / (np.power(N, 4) + np.power(N, 3) - np.power(N, 2) - N))
                rw_corr[x, y] = rw

        return rw_corr

    def ws_correlation(self):
        """
        Method that prepares a correlation table according to WS Coefficient of Rankings Similarity.
        Formula source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7302865/

        :return list:
        """
        criteria_count = self.matrix.shape[1]
        ws_corr = np.zeros([criteria_count, criteria_count])

        for x in range(criteria_count):
            for y in range(criteria_count):
                X = self.matrix[:][x]
                Y = self.matrix[:][y]
                N = len(X)
                sum_val = 0
                for i in range(N):
                    sum_val += ((2.0 ** (-X[i])) *
                                (np.abs(X[i] - Y[i]) /
                                 (np.maximum(np.abs(X[i] - 1), np.abs(X[i] - N)))))

                ws = 1 - sum_val
                ws_corr[x, y] = ws

        return ws_corr
