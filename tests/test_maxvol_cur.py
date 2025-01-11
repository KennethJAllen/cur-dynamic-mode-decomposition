"""Tests for maxvol_cur.py"""
import pytest
import numpy as np
from src import maxvol_cur as mc

@pytest.fixture(name='upper_triangular_matrix')
def fixture_upper_triangular_matrix():
    """An example matrix that is upper triangular."""
    example_upper_triangular_matrix = np.array([[1, 2, 3], [0, 1, 4], [0, 0, 1], [0, 0, 0]])
    return example_upper_triangular_matrix

def test_alternating_maxvol(upper_triangular_matrix):
    initial_row_indices = np.array([0])
    initial_column_indices = initial_row_indices
    row_indices, column_indices = mc.alternating_maxvol(upper_triangular_matrix, initial_row_indices, initial_column_indices)
    assert (row_indices == np.array([1])).all()
    assert (column_indices == np.array([2])).all()

def test_cur_decomposition(upper_triangular_matrix):
    row_indices = np.array([0, 1])
    column_indices = np.array([1, 2])
    c, u, r = mc.cur_decomposition(upper_triangular_matrix, row_indices, column_indices)
    assert (c == np.array([[2, 3], [1, 4], [0, 1], [0, 0]])).all()
    assert (u == np.array([[2, 3], [1, 4]])).all()
    assert (r == np.array([[1, 2, 3], [0, 1, 4]])).all()
