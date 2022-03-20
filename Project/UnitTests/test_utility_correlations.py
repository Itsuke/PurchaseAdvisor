"""
Unit tests to verify utility_rank module.
"""
import unittest
import numpy as np
import pandas as pd
from utility_correlations import UtilityCorrelations

class TestUtilityNormalization(unittest.TestCase):
    """
    Class used for the verification of implementation of normalization formulas
    """
    def setUp(self):
        self.test_mat = pd.DataFrame([[1, 1, 2, 3, 3, 1],
                                      [2, 3, 1, 2, 1, 2],
                                      [3, 2, 3, 1, 2, 3]])
        self.un = UtilityCorrelations(self.test_mat)

    def test_spearman_correlation(self):
        """
        The correlations for the the same elements is equal 1. we verify here, if there is a 100%
        correlation between 1st and 6th criteria. Values on diagonal should also be all Ones
        """
        out_corr = self.un.rw_correlation().round(2)

        self.assertEqual(out_corr[0, 5], 1)
        self.assertEqual(out_corr[5, 0], 1)
        np.testing.assert_array_equal(out_corr.diagonal(), [1, 1, 1, 1, 1, 1])

    def test_rank_similarity_coef_correlation(self):
        """
        The correlations for the the same elements is equal 1. we verify here, if there is a 100%
        correlation between 1st and 6th criteria. Values on diagonal should also be all Ones
        """
        out_corr = self.un.ws_correlation().round(2)

        self.assertEqual(out_corr[0, 5], 1)
        self.assertEqual(out_corr[5, 0], 1)
        np.testing.assert_array_equal(out_corr.diagonal(), [1, 1, 1, 1, 1, 1])


if __name__ == '__main__':
    unittest.main()
