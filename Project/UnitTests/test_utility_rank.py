"""
Unit tests to verify utility_rank module.
"""
import unittest
import numpy as np
from pymcdm import normalizations
from utility_rank import UtilityNormalization, NormalizationMethod

class TestUtilityNormalization(unittest.TestCase):
    """
    Class used for the verification of implementation of normalization formulas
    """
    def setUp(self):
        self.test_mat = np.array([
            [1500, 3000, 20, 64, 4],
            [2000, 4250, 15, 64, 6],
            [2500, 3500, 20, 64, 6],
            [2800, 3000, 10, 128, 8],
            [1000, 2000, 10, 64, 4],
            [3500, 4000, 20, 128, 8],
            [3200, 3800, 25, 128, 8],
            [2500, 3000, 20, 64, 8]
        ])
        self.ones_matrix = np.ones(self.test_mat.shape)
        self.normalization_methods = UtilityNormalization(self.test_mat)

    def test_profit_min_max(self):
        """
        Verify if the min_max method for profit is implemented correctly
        """
        expected_matrix = np.zeros(self.test_mat.shape)
        for col in range(np.size(self.test_mat, 1)):
            expected_matrix[:, col] = normalizations.minmax_normalization(self.test_mat[:, col])
        output_matrix = self.normalization_methods.profit_min_max()
        output_matrix_in_range_0_1 = (output_matrix <= 1).all() and (output_matrix >= 0).all()

        self.assertEqual(output_matrix_in_range_0_1, True)
        np.testing.assert_array_equal(output_matrix, expected_matrix)

    def test_profit_maximum(self):
        """
        Verify if the maximum method for profit is implemented correctly
        """
        expected_matrix = np.zeros(self.test_mat.shape)
        for col in range(np.size(self.test_mat, 1)):
            expected_matrix[:, col] = normalizations.max_normalization(self.test_mat[:, col])

        output_matrix = self.normalization_methods.profit_maximum()
        output_matrix_in_range_0_1 = (output_matrix <= 1).all() and (output_matrix >= 0).all()

        self.assertEqual(output_matrix_in_range_0_1, True)
        np.testing.assert_array_equal(output_matrix, expected_matrix)

    def test_profit_sum(self):
        """
        Verify if the sum method for profit is implemented correctly
        """
        expected_matrix = np.zeros(self.test_mat.shape)
        for col in range(np.size(self.test_mat, 1)):
            expected_matrix[:, col] = normalizations.sum_normalization(self.test_mat[:, col])

        output_matrix = self.normalization_methods.profit_sum()
        output_matrix_in_range_0_1 = (output_matrix <= 1).all() and (output_matrix >= 0).all()

        self.assertEqual(output_matrix_in_range_0_1, True)
        np.testing.assert_array_equal(expected_matrix, output_matrix)

    def test_profit_vector(self):
        """
        Verify if the vector method for profit is implemented correctly
        """
        expected_matrix = np.zeros(self.test_mat.shape)
        for col in range(np.size(self.test_mat, 1)):
            expected_matrix[:, col] = normalizations.vector_normalization(self.test_mat[:, col])
        output_matrix = self.normalization_methods.profit_vector()
        output_matrix_in_range_0_1 = (output_matrix <= 1).all() and (output_matrix >= 0).all()

        self.assertEqual(output_matrix_in_range_0_1, True)
        np.testing.assert_array_equal(output_matrix, expected_matrix)

    def test_cost_min_max(self):
        """
        Verify if the min_max method for cost is implemented correctly
        """
        expected_matrix = np.zeros(self.test_mat.shape)
        for col in range(np.size(self.test_mat, 1)):
            expected_matrix[:, col] = normalizations.minmax_normalization(self.test_mat[:, col],
                                                                          cost=True)
        output_matrix = self.normalization_methods.cost_min_max()
        output_matrix_in_range_0_1 = (output_matrix <= 1).all() and (output_matrix >= 0).all()

        self.assertEqual(output_matrix_in_range_0_1, True)
        np.testing.assert_array_equal(output_matrix, expected_matrix)

    def test_cost_maximum(self):
        """
        Verify if the maximum method for cost is implemented correctly
        """
        expected_matrix = np.zeros(self.test_mat.shape)
        for col in range(np.size(self.test_mat, 1)):
            expected_matrix[:, col] = normalizations.max_normalization(self.test_mat[:, col],
                                                                        cost=True)

        output_matrix = self.normalization_methods.cost_maximum()
        output_matrix_in_range_0_1 = (output_matrix <= 1).all() and (output_matrix >= 0).all()

        self.assertEqual(output_matrix_in_range_0_1, True)
        np.testing.assert_array_almost_equal(output_matrix, expected_matrix)

    def test_cost_sum(self):
        """
        Verify if the sum method for cost is implemented correctly
        """
        expected_matrix = np.zeros(self.test_mat.shape)
        for col in range(np.size(self.test_mat, 1)):
            expected_matrix[:, col] = normalizations.sum_normalization(self.test_mat[:, col],
                                                                       cost=True)

        output_matrix = self.normalization_methods.cost_sum()
        output_matrix_in_range_0_1 = (output_matrix <= 1).all() and (output_matrix >= 0).all()

        self.assertEqual(output_matrix_in_range_0_1, True)
        np.testing.assert_array_equal(output_matrix, expected_matrix)

    def test_cost_vector(self):
        """
        Verify if the vector method for cost is implemented correctly
        """
        expected_matrix = np.zeros(self.test_mat.shape)
        for col in range(np.size(self.test_mat, 1)):
            expected_matrix[:, col] = normalizations.vector_normalization(self.test_mat[:, col],
                                                                          cost=True)
        output_matrix = self.normalization_methods.cost_vector()
        output_matrix_in_range_0_1 = (output_matrix <= 1).all() and (output_matrix >= 0).all()

        self.assertEqual(output_matrix_in_range_0_1, True)
        np.testing.assert_array_equal(output_matrix, expected_matrix)

    def test_min_max_methods(self):
        """
        Verify if the sum for min_max method for cost and profit is equal to matrix of ones
        """
        output_matrix_profit = self.normalization_methods.profit_min_max()
        output_matrix_cost = self.normalization_methods.cost_min_max()
        sum_of_output_matrixes = output_matrix_profit + output_matrix_cost

        np.testing.assert_array_equal(sum_of_output_matrixes, self.ones_matrix)


    def test_maximum_methods(self):
        """
        Verify if the sum for maximum method for cost and profit is equal matrix of ones
        """
        output_matrix_profit = self.normalization_methods.profit_maximum()
        output_matrix_cost = self.normalization_methods.cost_maximum()
        sum_of_output_matrixes = output_matrix_profit + output_matrix_cost

        np.testing.assert_array_equal(sum_of_output_matrixes, self.ones_matrix)

    def test_sum_methods(self):
        """
        Verify if the sum for maximum method for cost and profit is equal to vector of ones
        """
        places = 0
        ones_vector = [1] * self.test_mat.shape[1]
        output_matrix_profit = self.normalization_methods.profit_sum()
        output_matrix_cost = self.normalization_methods.cost_sum()

        sum_output_matrix_profit = np.sum(output_matrix_profit, axis=0)
        sum_output_matrix_cost = np.sum(output_matrix_cost, axis=0)

        np.testing.assert_array_almost_equal(sum_output_matrix_profit, ones_vector, places)
        np.testing.assert_array_almost_equal(sum_output_matrix_cost, ones_vector, places)

    def test_vector_methods(self):
        """
        Verify if the sum for vector method for cost and profit is equal to matrix of ones
        """
        output_matrix_profit = self.normalization_methods.profit_vector()
        output_matrix_cost = self.normalization_methods.cost_vector()
        sum_of_output_matrixes = output_matrix_profit + output_matrix_cost

        np.testing.assert_array_equal(sum_of_output_matrixes, self.ones_matrix)

    def test_combined_norm_full_profit(self):
        """
        Verify if the combined normalization for profit is implemented right
        """
        profit_vector = ['p'] * self.test_mat.shape[1]

        expected_min_max_matrix = self.normalization_methods.profit_min_max()
        expected_maximum_matrix = self.normalization_methods.profit_maximum()
        expected_sum_matrix = self.normalization_methods.profit_sum()
        expected_vector_matrix = self.normalization_methods.profit_vector()

        output_min_max_matrix = \
            self.normalization_methods.combined_norm(NormalizationMethod.MINMAX, profit_vector)
        output_maximum_matrix = \
            self.normalization_methods.combined_norm(NormalizationMethod.MAXIMUM, profit_vector)
        output_sum_matrix = \
            self.normalization_methods.combined_norm(NormalizationMethod.SUM, profit_vector)
        output_vector_matrix = \
            self.normalization_methods.combined_norm(NormalizationMethod.VECTOR, profit_vector)

        np.testing.assert_array_equal(expected_min_max_matrix, output_min_max_matrix)
        np.testing.assert_array_equal(expected_maximum_matrix, output_maximum_matrix)
        np.testing.assert_array_equal(expected_sum_matrix, output_sum_matrix)
        np.testing.assert_array_equal(expected_vector_matrix, output_vector_matrix)

    def test_combined_norm_full_cost(self):
        """
        Verify if the combined normalization for cost is implemented right
        """
        cost_vector = ['c'] * self.test_mat.shape[1]

        expected_min_max_matrix = self.normalization_methods.cost_min_max()
        expected_maximum_matrix = self.normalization_methods.cost_maximum()
        expected_sum_matrix = self.normalization_methods.cost_sum()
        expected_vector_matrix = self.normalization_methods.cost_vector()

        output_min_max_matrix = \
            self.normalization_methods.combined_norm(NormalizationMethod.MINMAX, cost_vector)
        output_maximum_matrix = \
            self.normalization_methods.combined_norm(NormalizationMethod.MAXIMUM, cost_vector)
        output_sum_matrix = \
            self.normalization_methods.combined_norm(NormalizationMethod.SUM, cost_vector)
        output_vector_matrix = \
            self.normalization_methods.combined_norm(NormalizationMethod.VECTOR, cost_vector)

        np.testing.assert_array_equal(expected_min_max_matrix, output_min_max_matrix)
        np.testing.assert_array_equal(expected_maximum_matrix, output_maximum_matrix)
        np.testing.assert_array_equal(expected_sum_matrix, output_sum_matrix)
        np.testing.assert_array_equal(expected_vector_matrix, output_vector_matrix)

    def test_mixed_norm_methods(self):
        """
        Verify if the combined normalization for profit or cost is implemented right
        """
        places = 0
        ones_vector = [1] * self.test_mat.shape[1]
        ones_matrix = np.ones(self.test_mat.shape)
        mixed_vector1 = ['p' if num % 2 else 'c' for num in range(0, self.test_mat.shape[1])]
        mixed_vector2 = ['c' if num % 2 else 'p' for num in range(0, self.test_mat.shape[1])]

        output_min_max_matrix1 = \
            self.normalization_methods.combined_norm(NormalizationMethod.MINMAX, mixed_vector1)
        output_min_max_matrix2 = \
            self.normalization_methods.combined_norm(NormalizationMethod.MINMAX, mixed_vector2)
        sum_of_output_min_max_matrixes = output_min_max_matrix1 + output_min_max_matrix2

        output_maximum_matrix1 = \
            self.normalization_methods.combined_norm(NormalizationMethod.MAXIMUM, mixed_vector1)
        output_maximum_matrix2 = \
            self.normalization_methods.combined_norm(NormalizationMethod.MAXIMUM, mixed_vector2)
        sum_of_output_maximum_matrixes = output_maximum_matrix1 + output_maximum_matrix2

        output_sum_matrix1 = \
            self.normalization_methods.combined_norm(NormalizationMethod.SUM, mixed_vector1)
        output_sum_matrix2 = \
            self.normalization_methods.combined_norm(NormalizationMethod.SUM, mixed_vector2)
        sum_of_output_sum_matrix1 = np.sum(output_sum_matrix1, axis=0)
        sum_of_output_sum_matrix2 = np.sum(output_sum_matrix2, axis=0)

        output_vector_matrix1 = \
            self.normalization_methods.combined_norm(NormalizationMethod.VECTOR, mixed_vector1)
        output_vector_matrix2 = \
            self.normalization_methods.combined_norm(NormalizationMethod.VECTOR, mixed_vector2)
        sum_of_output_vector_matrixes = output_vector_matrix1 + output_vector_matrix2

        np.testing.assert_array_equal(sum_of_output_min_max_matrixes, ones_matrix)
        np.testing.assert_array_equal(sum_of_output_maximum_matrixes, ones_matrix)
        np.testing.assert_array_almost_equal(sum_of_output_sum_matrix1, ones_vector, places)
        np.testing.assert_array_almost_equal(sum_of_output_sum_matrix2, ones_vector, places)
        np.testing.assert_array_equal(sum_of_output_vector_matrixes, ones_matrix)


if __name__ == '__main__':
    unittest.main()
