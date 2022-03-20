"""
Implementation of weights formulas from article

Sałabun, W., Watróbski, J., & Shekhovtsov, A. (2020). Are mcda methods
benchmarkable? a comparative study of topsis, vikor, copras, and promethee
ii methods. Symmetry, 12 (9), 1549.
"""
import numpy as np


class UtilityWeights:
    """
    Class used for the calculation of criteria weights using different methods
    """
    def __init__(self, matrix):
        """
        Constructor of the UtilityWeights class
        :param ndarray matrix: matrix for which the weights should be determined
        """
        self._matrix = matrix

    def weights_equal(self):
        """
        Method to calculate weights using equal method described in article

        :return list: list of wight calculated using equal method
        """
        criteria_size = self._matrix.shape[1]
        wj = list((1 / criteria_size) * np.ones(criteria_size))
        return wj

    def weights_entropy(self):
        """
        Method to calculate weights using entropy method described in article

        :return list: list of wight calculated using entropy method
        """
        p = np.zeros(self._matrix.shape)
        n = self._matrix.shape[1]
        m = self._matrix.shape[0]

        for j in range(n):
            sums = np.sum(self._matrix[:, j])
            for i in range(m):
                p[i, j] = self._matrix[i, j] / sums

        Ej = [np.divide(np.sum(np.multiply(self._matrix[:, index], np.log(self._matrix[:, index]))),
                        np.log(m)) for index in range(n)]

        wj = [(1 + Ej[j]) / np.sum(np.add(1, Ej)) for j in range(n)]
        return wj

    def weights_std(self):
        """
        Method to calculate weights using std method described in article

        :return list: list of wight calculated using std method
        """
        n = self._matrix.shape[1]
        m = self._matrix.shape[0]

        uj = [np.sum(np.power(self._matrix[:, j] - np.mean(self._matrix[:, j]), 2)) / m
              for j in range(n)]

        wj = [uj[j] / np.sum(uj) for j in range(n)]

        return wj
