"""
Unit tests to verify utility_rank module.
"""
import unittest
import numpy as np
from pymcdm import weights
from utility_weights import UtilityWeights

class TestUtilityNormalization(unittest.TestCase):
    """
    Class used for the verification of implementation of normalization formulas
    """
    def setUp(self):
        self.test_mat = np.array([[1, 1, 2, 3, 3, 1],
                                  [2, 3, 1, 2, 1, 2],
                                  [4, 5, 3, 1, 2, 3]])
        self.uw = UtilityWeights(self.test_mat)

    def test_equal_weights(self):
        """
        The standard deviations weights should all be equal. They should have the value of
        1/number_of_criteria
        :return:
        """
        out_weights = np.array(self.uw.weights_equal())
        expected_weights = weights.equal_weights(self.test_mat)

        # Because of summing a very long floating numbers the calculation error shows up. There is a
        # need to use assertAlomstEqual method
        self.assertAlmostEqual(sum(out_weights), 1, 1)
        self.assertEqual(out_weights[0], 1 / len(out_weights))
        self.assertTrue(all(element == out_weights[0] for element in out_weights))
        np.testing.assert_array_equal(out_weights, expected_weights)

    def test_standard_deviation_weights(self):
        """
        The sum of standard deviations weights should be equal one .
        :return:
        """
        out_weights = np.array(self.uw.weights_std())

        self.assertEqual(sum(out_weights), 1)


    def test_entrophy_weights(self):
        """
        The sum of entropy weights should be equal one .
        """
        out_weights = np.array(self.uw.weights_entropy())

        self.assertEqual(sum(out_weights), 1)



if __name__ == '__main__':
    unittest.main()
