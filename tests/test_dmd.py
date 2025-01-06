"""Tests for dmd.py"""
import pytest
import numpy as np
import dmd

@pytest.fixture(name='upper_triangular_matrix')
def fixture_upper_triangular_matrix():
    """An example matrix"""
    upper_triangular_matrix = np.array([[1, 2, 3, 6], [0, 1, 4, 8], [0, 0, 1, 9], [0, 0, 0, 10], [0, 0, 0, 0]])
    return upper_triangular_matrix

# test dmd
def test_svd_dmd(upper_triangular_matrix):
    """Test the eigenvalues of the dynamic mode decomposition."""
    rank = 2
    eig_values, modes = dmd.svd_dmd(upper_triangular_matrix, rank)
    n = upper_triangular_matrix.shape[0]
    assert len(eig_values) == rank
    assert (modes.shape == np.array([n, rank])).all()

def test_cur_forecast(upper_triangular_matrix):
    """Tests timeseries forecasting via DMD."""
    rank = 2
    num_forecasts = 10
    forecast_results = dmd.forecast(upper_triangular_matrix, rank, num_forecasts, 'cur')
    n = upper_triangular_matrix.shape[0]
    assert (forecast_results.shape == np.array([n, num_forecasts])).all()

def test_svd_forecast(upper_triangular_matrix):
    """Tests timeseries forecasting via DMD."""
    rank = 2
    num_forecasts = 10
    forecast_results = dmd.forecast(upper_triangular_matrix, rank, num_forecasts, 'svd')
    n = upper_triangular_matrix.shape[0]
    assert (forecast_results.shape == np.array([n, num_forecasts])).all()
