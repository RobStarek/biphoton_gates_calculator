# -*- coding: utf-8 -*-
"""
Turn sympy matrix expressions that describe single-photon mapping
into their corresponding two-photon computational operator in
coincidence basis. 

The conversion sp_construct_operator() needs a list of allowed input
mode combinations and a filtering matrix.
The filtering matrix is obtained using sp_create_filter_matrix()
from the list of allowed coincidences.
"""

import numpy as np
import sympy as sp


def sp_construct_operator(single_photon_map, allowed_inputs, filter_matrix):
    """
    Take single photon map, feed the inputs with allowed combination of mode population
    and then filter out the results for allowed coincidences.
    Symbolic version.

    Args:
    single_photon_map ... n x m sympy matrix
    allowed_inputs ... list of 2-tuples that define allowed inputs and are
       sorted in ascending order, binary representation of index encodes
       logic input computational state. Maximal number in tuples is (m-1).
    allowed_outputs ... filtering matrix that determines how combinations of photons
       are interpreted as computational base states and which combinations are
       acceppted.

    Returns:
    operator ... sympy matrix with constructed computatioanl basis operator
    """

    # 0) initialize things and save aux variables
    (n_map_rows, _) = single_photon_map.shape
    n_inputs = len(allowed_inputs)
    n_creation_combinations = n_map_rows**2
    creation_matrix = sp.zeros(n_creation_combinations, n_inputs)

    # 1) iterate over allowed inputs and populate
    # creation operator matrix
    for input_index, (mode_index_1, mode_index_2) in enumerate(allowed_inputs):
        # instead of matrix multiplication, I just select the corresponding column
        creation_vector_1 = single_photon_map[:, mode_index_1]
        creation_vector_2 = single_photon_map[:, mode_index_2]
        # calculate all the coefficients for combination of creation operators
        # note that * stand for sympy matrix multiplication
        sub_creation_matrix = creation_vector_1 * creation_vector_2.T
        # fill creation matrix
        creation_matrix[:, input_index] = sub_creation_matrix.reshape(
            n_creation_combinations, 1)

    # 2) filter out the results and assembly
    operator = filter_matrix * creation_matrix
    return operator


def sp_create_filter_matrix(allowed_outputs: list, \
    n_map_rows: int, \
    first_comb: bool = True,\
    second_comb: bool = True\
    ) -> sp.matrices.dense.MutableDenseMatrix:
    """
    Create filtering matrix from a list describing allowed
    coincidences.

    Args:
        allowed_outputs ... list of 2-tuples, integers inside
            decribe which modes have to be populated to accept the
            output. The binary representation of tuple index
            describes which computational state the given
            combination encodes and therefore the
            order is important. Maximal value in tuple
            is n-1 where n is number of output modes.
        n_map_rows ... integer, number of output modes
        first_comb, second_comb: booleans, set one of them to False
            to make the photons distinguishable and accept only
            certain combinations. Set both to True to make
            the photon indistinduishable.

    Returns:
        filter_matrix ... filtering sympy matrix
    """
    n_creation_combinations = n_map_rows**2
    n_allowed_outputs = len(allowed_outputs)
    filter_matrix = np.zeros((n_allowed_outputs, n_creation_combinations))
    for i, (j, k) in enumerate(allowed_outputs):
        if first_comb:
            #filter_matrix[i, j*4+k] = 1
            filter_matrix[i, n_map_rows*j+k] = 1
        if second_comb:
            #filter_matrix[i, k*4+j] = 1
            filter_matrix[i, n_map_rows*k+j] = 1
    return sp.Matrix(filter_matrix)

# # Example:
# # single-photon map for 2-mode beam splitter
# # operating on polarized light
# if __name__ == '__main__':

#     # mode indexing:
#     # 0 ... port A, H polarization
#     # 1 ... port A, V polarization
#     # 2 ... port B, H polarization
#     # 3 ... port B, V polarization
#     my_map = sp.Matrix(
#         [
#             ['t', 0, 'r', 0],
#             [0, 't', 0, 'r'],
#             ['-r', 0, 't', 0],
#             [0, '-r', 0, 't']
#         ]
#     )
#     # for example: computational |00> is encoded into
#     # one H-polarized photon in port A and the
#     # other H-polarized photon in port B
#     # the list is ordered by computational value
#     # in asceding manner, the ordering is important
#     allowed_inputs = [
#         (0, 2),  # |00>
#         (0, 3),  # |01>
#         (1, 2),  # |10>
#         (1, 3)  # |11>
#     ]
#     # similarly we define allowed output encoding
#     allowed_indices = [(0, 2), (0, 3), (1, 2), (1, 3)]
#     # compute filtering matrix
#     # last two bool arguments are used to create
#     # matrices for distinguishable photons
#     sp_filter_matrix = sp_create_filter_matrix(allowed_indices, 4, True, True)
#     # calculate symbolic expression fot the operator
#     mat = sp_construct_operator(my_map, allowed_inputs, sp_filter_matrix)
