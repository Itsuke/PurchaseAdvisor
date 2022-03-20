"""
Unit tests to verify mcda module.
"""
import unittest
import pymcdm
import numpy as np
from mcda import MCDA, SortingType
from utility_rank import UtilityNormalization, NormalizationMethod


class TestMCDA(unittest.TestCase):
    """
    Unit test class for testing mcda module
    """
    def setUp(self):
        self.weights = np.array([0.0390, 0.3730, 0.0572, 0.0556, 0.4753])
        self.pc_vec = ['c', 'c', 'p', 'c', 'p']
        self.types = np.array([1 if importance == "p" else -1 for importance in self.pc_vec])
        self.test_mat = np.array([
            [1500, 3000, 20, 64,  4],
            [2000, 4250, 15, 64,  6],
            [2500, 3500, 20, 64,  6],
            [2800, 3000, 10, 128, 8],
            [1000, 2000, 10, 64,  4],
            [3500, 4000, 20, 128, 8],
            [3200, 3800, 25, 128, 8],
            [2500, 3000, 20, 64,  8]
        ])
        self.normalization_methods = UtilityNormalization(self.test_mat)
        self.norm_test_mat = self.normalization_methods.combined_norm(NormalizationMethod.MAXIMUM,
                                                                      self.pc_vec)
        self.mcda = MCDA(self.test_mat, self.norm_test_mat, self.weights, self.pc_vec)
        self.asc = SortingType(1)
        self.desc = SortingType(0)

    def test_calculate_rank(self):
        """
        Testing if the sorting method determines ranking right
        """
        numbers_from_smallest_to_biggest = [0.1, 0.3, 0.4, 0.5, 0.7, 0.9, 1]
        expected_asc_rank = list(range(1, len(numbers_from_smallest_to_biggest) + 1))

        output_asc_rank = self.mcda._calculate_rank(numbers_from_smallest_to_biggest, self.asc)
        self.assertEqual(expected_asc_rank, output_asc_rank)

        expected_desc_rank = expected_asc_rank[::-1]
        output_desc_rank = self.mcda._calculate_rank(numbers_from_smallest_to_biggest, self.desc)
        self.assertEqual(expected_desc_rank, output_desc_rank)

    def test_spotis_method(self):
        """
        Testing if the spotis formulas are correctly implemented
        """
        output_spotis = self.mcda._MCDA__spotis()

        minus = np.array(self.test_mat - 1)
        plus = np.array(self.test_mat + 1)
        bounds = np.vstack((np.min(minus, axis=0), np.max(plus, axis=0))).T
        pymcdm_spotis = pymcdm.methods.SPOTIS()
        expected_spotis = pymcdm_spotis(self.test_mat, self.weights, self.types, bounds=bounds)

        np.testing.assert_array_equal(expected_spotis, output_spotis)

    def test_topsis_method(self):
        """
        Testing if the topsis formulas are correctly implemented
        """
        output_topsis = np.array(self.mcda._MCDA__topsis())

        topsis = pymcdm.methods.TOPSIS(normalization_function=
                                       pymcdm.normalizations.max_normalization)
        expected_topsis = topsis(self.test_mat, self.weights, self.types)


        np.testing.assert_array_almost_equal(expected_topsis, output_topsis)

    def test_promethee_2_method(self):
        """
        Testing if the promethee_2 formulas are correctly implemented
        """
        output_promethee_2 = np.array(self.mcda._MCDA__promethee_2())

        promethee = pymcdm.methods.PROMETHEE_II(preference_function='usual')
        expected_promethee = promethee(self.test_mat, self.weights, self.types)

        np.testing.assert_array_almost_equal(expected_promethee, output_promethee_2)

    def test_spotis_ranking_method(self):
        """
        Testing if the spotis ranking is correctly determined
        """
        _, output_spotis_ranking = self.mcda.get_rank_spotis()

        minus = np.array(self.test_mat - 1)
        plus = np.array(self.test_mat + 1)
        bounds = np.vstack((np.min(minus, axis=0), np.max(plus, axis=0))).T
        pymcdm_spotis = pymcdm.methods.SPOTIS()

        expected_spotis = pymcdm_spotis(self.test_mat, self.weights, self.types, bounds=bounds)
        expected_spotis_ranking = pymcdm.helpers.rankdata(expected_spotis)

        np.testing.assert_array_equal(expected_spotis_ranking, output_spotis_ranking)

    def test_topsis_ranking_method(self):
        """
        Testing if the topsis ranking is correctly determined
        """
        _, output_topsis_ranking = self.mcda.get_rank_topsis()

        topsis = pymcdm.methods.TOPSIS(normalization_function=
                                       pymcdm.normalizations.max_normalization)
        expected_topsis = topsis(self.test_mat, self.weights, self.types)
        expected_topsis_ranking = pymcdm.helpers.rrankdata(expected_topsis).astype(int)

        np.testing.assert_array_almost_equal(expected_topsis_ranking, output_topsis_ranking)

    def test_promethee_2_ranking_method(self):
        """
        Testing if the promethee_2 ranking is correctly determined
        """
        _, output_promethee_2_ranking = np.array(self.mcda.get_rank_promethee_2())

        promethee = pymcdm.methods.PROMETHEE_II(preference_function='usual')
        expected_promethee = promethee(self.test_mat, self.weights, self.types)
        expected_promethee_ranking = pymcdm.helpers.rrankdata(expected_promethee).astype(int)

        np.testing.assert_array_almost_equal(expected_promethee_ranking, output_promethee_2_ranking)


if __name__ == '__main__':
    unittest.main()
