'''Tests for dmd.py'''
# pylint: disable=redefined-outer-name

import pytest
import numpy as np
from dynamic_mode_decomposition import dmd

@pytest.fixture
def example_upper_triangular_matrix():
    '''An example matrix'''
    example_upper_triangular_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    return example_upper_triangular_matrix

# test dmd
def test(example_upper_triangular_matrix):
    '''Test the eigenvalues of the dynamic mode decomposition.'''
    rank = example_upper_triangular_matrix.shape[0]
    [eig_values, _] = dmd.dmd(example_upper_triangular_matrix, rank)
    assert pytest.approx(eig_values[0], 0)