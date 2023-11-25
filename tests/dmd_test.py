'''Tests for dmd.py'''
# pylint: disable=redefined-outer-name

import pytest
import numpy as np
from dynamic_mode_decomposition import dmd

@pytest.fixture
def example_upper_triangular_matrix():
    '''An example matrix'''
    example_upper_triangular_matrix = np.array([[1, 2], [0, 4]])
    return example_upper_triangular_matrix

# test dmd
def test_eigenvalues_of_dmd(example_upper_triangular_matrix):
    '''Test the eigenvalues of the dynamic mode decomposition.'''
    rank = example_upper_triangular_matrix.shape[0]
    [eig_values, _] = dmd.dmd(example_upper_triangular_matrix, rank)
    assert np.array_equal(eig_values, np.diag(example_upper_triangular_matrix))