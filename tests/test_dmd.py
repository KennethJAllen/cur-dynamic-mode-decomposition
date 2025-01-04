'''Tests for dmd.py'''
# pylint: disable=redefined-outer-name

import pytest
import numpy as np
from src import dmd

@pytest.fixture(name='upper_triangular_matrix')
def fixture_upper_triangular_matrix():
    '''An example matrix'''
    example_upper_triangular_matrix = np.array([[1, 2, 3], [0, 1, 4], [0, 0, 1]])
    return example_upper_triangular_matrix

# test dmd
def test_svd_dmd(upper_triangular_matrix):
    '''Test the eigenvalues of the dynamic mode decomposition.'''
    rank = upper_triangular_matrix.shape[0]
    eig_values, _ = dmd.svd_dmd(upper_triangular_matrix, rank)
    assert pytest.approx(eig_values[0], 0)
