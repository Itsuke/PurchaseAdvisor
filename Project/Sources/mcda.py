"""
Class for solving MCDA problems.
"""
from enum import Enum
import numpy as np


class SortingType(Enum):
    """ Enum to determine the order type."""
    DESC = 0
    ASC = 1


class MCDA:
    """Class for solving the MCDA problems."""
    def __init__(self, matrix, norm_mat, weights, cost_profit):
        """ Init of a MCDA class.
        :param ndarray matrix: matrix with the alternatives and criteria to be used in ranking
                             methods.
        :param ndarray norm_mat: normalized matrix.
        :param ndarray weights: list of weights for each criteria.
        :param ndarray cost_profit: list of characters 'p' and 'c' staying for profit and cost.

        :param matrix: raw matrix to be used in MCDA methods.
        :param norm_mat: normalized matrix to be used in MCDA methods.
        """
        self._matrix = matrix
        self._norm_mat = norm_mat
        self._weights = weights
        self._cost_profit = cost_profit
        self._rows = np.size(matrix, 0)  # Height
        self._cols = 1 if len(self._matrix.shape) <= 1 else np.size(self._matrix, 1)  # Width

    @staticmethod
    def _calculate_rank(my_list, is_sorted_asc):
        """ Method calculates the ranking in a list.

        :param list my_list: list with values to be ranked.
        :param SortingType is_sorted_asc: enum where 1 means ascending order

        :return list: rank in order.
        """
        ranking = {}
        rank = 1
        for value in (sorted(my_list) if is_sorted_asc == SortingType.ASC
        else sorted(my_list, reverse=True)):
            if value not in ranking:
                ranking[value] = rank
                rank += 1
        return [ranking[value] for value in my_list]

    @staticmethod
    def _create_evaluation_table(matrix):
        criteria = int(np.size(matrix, 1))
        alternatives = int(np.size(matrix, 0))
        sab_3d = np.array(np.zeros([criteria, alternatives, alternatives]))

        for criterion in range(criteria):
            ab_mat = np.array(np.zeros([alternatives, alternatives]))

            for alternative in range(alternatives):
                vec = matrix[:, criterion] - matrix[alternative, criterion]
                ab_mat[:, alternative] = vec

            sab_3d[criterion] = ab_mat

        # Ususal Criterion
        sab_3d[sab_3d > 0] = 1
        sab_3d[sab_3d <= 0] = 0

        return sab_3d

    def __topsis(self):
        """ Method implemented according to:
            Sałabun, W., Watróbski, J., & Shekhovtsov, A. (2020). Are mcda methods benchmarkable?
            a comparative study of topsis, vikor, copras, and promethee ii methods.

        :return: returns the list with values used to determine rankings.
        :rtype: list
        """
        matrix = np.copy(self._norm_mat)
        matrix *= self._weights
        pis = matrix.max(axis=0)
        nis = matrix.min(axis=0)
        dpis = []
        dnis = []

        for i in range(0, self._rows):
            dpi = 0
            dni = 0

            for j in range(0, self._cols):
                dpi += np.power((matrix[i, j] - pis[j]), 2)
                dni += np.power((matrix[i, j] - nis[j]), 2)

            dpis.append(np.sqrt(dpi))
            dnis.append(np.sqrt(dni))

        ci_list = []
        for i in range(0, self._rows):
            c_value = dnis[i] / (dnis[i] + dpis[i])
            ci_list.append(c_value)

        return ci_list

    def __spotis(self):
        """ Method implemented according to:
            Dezert, J., Tchamova, A., Han, D., & Tacnet, J.-M. (2020). The spotis rank
            reversal free method for multi-criteria decision-making support.

        :return: returns the list with values used to determine rankings.
        :rtype: list
        """
        s_min = self._matrix.min(axis=0) - 1
        s_max = self._matrix.max(axis=0) + 1
        s_star = [s_min[idx] if self._cost_profit[idx] == 'c' else s_max[idx]
                  for idx in range(0, len(self._cost_profit))]

        dij = np.zeros(self._matrix.shape)

        for j in range(0, self._cols):
            for i in range(0, self._rows):
                dij[i, j] = np.abs(self._matrix[i, j] - (s_star[j])) / np.abs(s_max[j] - s_min[j])

        rank = np.sum((np.multiply(dij, self._weights)), axis=1)
        return rank

    def __promethee_2(self):
        evt = self._create_evaluation_table(self._matrix)
        cevt = np.zeros(evt.shape[1:])

        for mat_id in range(evt.shape[0]):
            if self._cost_profit[mat_id] == 'c':
                # transpose for cost calculations
                evt[mat_id] = np.transpose(evt[mat_id])

            evt[mat_id] *= self._weights[mat_id]
            cevt += evt[mat_id]

        #cevt + np.transpose(cevt)

        fi_plus = (1 / (self._rows - 1)) * np.transpose(cevt.sum(axis=1))
        fi_minus = (1 / (self._rows - 1)) * cevt.sum(axis=0)
        fi = fi_plus - fi_minus

        return fi

    def get_rank_topsis(self):
        """ Getter for a topsis rank

        :return: rank determined by topsis method.
        :rtype: list
        """

        topsis = self.__topsis()
        return topsis, self._calculate_rank(topsis, SortingType.DESC)

    def get_rank_spotis(self):
        """ Getter for a spotis rank.

        :return: rank determined by spotis method.
        :rtype: list
        """

        spotis = self.__spotis()
        return spotis, self._calculate_rank(spotis, SortingType.ASC)

    def get_rank_promethee_2(self):
        """ Getter for a spotis rank.

        :return: rank determined by spotis method.
        :rtype: list
        """

        promethee_2 = self.__promethee_2()
        return promethee_2, self._calculate_rank(promethee_2, SortingType.DESC)
