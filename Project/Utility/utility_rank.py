"""
Implementation of normalisation formulas from article:

Sałabun, W., Watróbski, J., & Shekhovtsov, A. (2020). Are mcda methods
benchmarkable? a comparative study of topsis, vikor, copras, and promethee
ii methods. Symmetry, 12 (9), 1549.
"""
import sys
from enum import Enum
import numpy as np
import more_itertools as it


class NormalizationMethod(Enum):
    """
    Enum Class to identify normalization methods
    """
    MINMAX = 1
    MAXIMUM = 2
    SUM = 3
    VECTOR = 4


class UtilityNormalization:
    """
    Class used to normalize 2d matrixes in different ways
    """
    def __init__(self, matrix):
        """
        Constructor of the UtilityCorrelations class
        :param ndarray matrix: 2d data matrix
        """
        self._matrix = matrix
        self._rows = np.size(self._matrix, 0)  # Height
        if len(self._matrix.shape) <= 1:
            self._cols = 1  # Width
        else:
            self._cols = np.size(self._matrix, 1)  # Width

    def profit_min_max(self):
        """
        Implementation of formula 30 from article to normalize matrix using the min max method for
        profit

        :return ndarray: Normalized by minmax formula for profit
        """
        matrix = np.zeros(self._matrix.shape)
        for j in range(0, self._cols):
            for i in range(0, self._rows):
                if not it.all_equal(self._matrix[:, j]):
                    matrix[i, j] = (self._matrix[i, j] - np.min(self._matrix[:, j])) / \
                                (np.max(self._matrix[:, j]) - np.min(self._matrix[:, j]))
                else:
                    pass

        return matrix

    def profit_maximum(self):
        """
        Implementation of formula 32 from article to normalize matrix using the maximum method for
        profit

        :return ndarray: Normalized by maximum formula for profit
        """
        matrix = np.zeros(self._matrix.shape)
        for j in range(0, self._cols):
            for i in range(0, self._rows):
                matrix[i, j] = self._matrix[i, j] / np.max(self._matrix[:, j])

        return matrix

    def profit_sum(self):
        """
        Implementation of formula 34 from article to normalize matrix using the sum method for
        profit

        :return ndarray: Normalized by sum formula for profit
        """
        matrix = np.zeros(self._matrix.shape)
        for j in range(0, self._cols):
            for i in range(0, self._rows):
                matrix[i, j] = self._matrix[i, j] / np.sum(self._matrix[:, j])

        return matrix

    def profit_vector(self):
        """
        Implementation of formula 36 from article to normalize matrix using the min max method for
        profit

        :return ndarray: Normalized by vector formula for profit
        """
        matrix = np.zeros(self._matrix.shape)

        for j in range(0, self._cols):
            square_vector = np.power(self._matrix[:, j], 2)

            for i in range(0, self._rows):
                matrix[i, j] = self._matrix[i, j] / np.sqrt(np.sum(square_vector))

        return matrix

    def cost_min_max(self):
        """
        Implementation of formula 31 from article to normalize matrix using the min-max method for
        cost

        :return ndarray: Normalized by minmax formula for cost
        """
        matrix = np.zeros(self._matrix.shape)
        for j in range(0, self._cols):
            for i in range(0, self._rows):
                if not it.all_equal(self._matrix[:, j]):
                    matrix[i, j] = (np.max(self._matrix[:, j]) - self._matrix[i, j]) / \
                                   (np.max(self._matrix[:, j]) - np.min(self._matrix[:, j]))
                else:
                    pass

        return matrix

    def cost_maximum(self):
        """
        Implementation of formula 33 from article to normalize matrix using the maximum method for
        cost

        :return ndarray: Normalized by maximum formula for cost
        """
        matrix = np.zeros(self._matrix.shape)
        for j in range(0, self._cols):
            for i in range(0, self._rows):
                matrix[i, j] = 1 - (self._matrix[i, j] / np.max(self._matrix[:, j]))

        return matrix

    def cost_sum(self):
        """
        Implementation of formula 35 from article to normalize matrix using the sum method for
        cost

        :return ndarray: Normalized by sum formula for cost
        """
        matrix = np.zeros(self._matrix.shape)
        for j in range(0, self._cols):
            divide_vector = np.divide(1, self._matrix[:, j])
            for i in range(0, self._rows):
                matrix[i, j] = (1 / self._matrix[i, j]) / (np.sum(divide_vector))

        return matrix

    def cost_vector(self):
        """
        Implementation of formula 37 from article to normalize matrix using the vector method for
        cost

        :return ndarray: Normalized by vector formula for cost
        """
        matrix = np.zeros(self._matrix.shape)

        for j in range(0, self._cols):
            square_vector = np.power(self._matrix[:, j], 2)

            for i in range(0, self._rows):
                matrix[i, j] = 1 - (self._matrix[i, j] / np.sqrt(np.sum(square_vector)))

        return matrix

    def combined_norm(self, method_id, profit_cost_vector):
        """
        Method that allows to use combination of profit and cost to normalize the matrix with given
        method
        :param NormalizationMethod method_id: Enum pointing a certain normalize method
        :param list profit_cost_vector: list determining what model to use, a cost or profit

        :return ndarray: Normalized matrix by chosen method formula for mixed costs and profits
                         vector
        """
        matrix = np.zeros(self._matrix.shape)
        vector_length = len(profit_cost_vector)

        if np.size(self._matrix, 1) == vector_length:
            for column in range(0, vector_length):
                norm_column = []
                a_column = self._matrix[:, column].reshape([self._matrix.shape[0], 1])
                un_col = UtilityNormalization(a_column)

                if profit_cost_vector[column] == 'p':
                    if method_id == NormalizationMethod.MINMAX:
                        norm_column = un_col.profit_min_max().reshape(self._matrix.shape[0])

                    elif method_id == NormalizationMethod.MAXIMUM:
                        norm_column = un_col.profit_maximum().reshape(self._matrix.shape[0])

                    elif method_id == NormalizationMethod.SUM:
                        norm_column = un_col.profit_sum().reshape(self._matrix.shape[0])

                    elif method_id == NormalizationMethod.VECTOR:
                        norm_column = un_col.profit_vector().reshape(self._matrix.shape[0])

                    else:
                        print("Used wrong normalization method")
                        sys.exit()

                    matrix[:, column] = norm_column

                elif profit_cost_vector[column] == 'c':
                    if method_id == NormalizationMethod.MINMAX:
                        norm_column = un_col.cost_min_max().reshape(self._matrix.shape[0])

                    elif method_id == NormalizationMethod.MAXIMUM:
                        norm_column = un_col.cost_maximum().reshape(self._matrix.shape[0])

                    elif method_id == NormalizationMethod.SUM:
                        norm_column = un_col.cost_sum().reshape(self._matrix.shape[0])

                    elif method_id == NormalizationMethod.VECTOR:
                        norm_column = un_col.cost_vector().reshape(self._matrix.shape[0])

                    else:
                        print("Used wrong normalization method")
                        sys.exit()

                    matrix[:, column] = norm_column

                else:
                    print("Wrong value in profit/cost vector")
                    sys.exit()
        else:
            print("Wrong size of profit/cost vector")
            sys.exit()

        return matrix
