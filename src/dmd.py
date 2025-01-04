"""Implementations of the dynamic mode decomposition (DMD)
including the CUR DMD developed by K. Allen and S. De Pascuale."""
import numpy as np
from numpy import linalg as LA
import sklearn.utils.extmath as skmath
from src import maxvol_cur as mc

def cur_dmd(data: np.ndarray,
            rank: int,
            initial_row_indices: np.ndarray = None,
            initial_column_indices: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs the column-submatrix-row (CUR) based dynamic mode decomposition (DMD) on n x p time series data.
    If initial row or column indices are not supplied, they will be chosen randomly.

    Parameters:
        data (np.ndarray): Time series data where each column is a time snapshot.
        rank (int): Number of singular values for SVD decomposition
        initial_row_indices(np.ndarray, optional): Initial row indices for maxvol algorithm
        initial_column_indices(np.ndarray, optional): Initial column for maxvol algorithm

    Returns:
        DMD eigenvalues and DMD modes of the dataset.

    Example:
        dmd_eig_values, dmd_modes = dmd(np.random.rand(100, 10), 5)
    """
    if initial_row_indices is None:
        n, _ = data.shape
        initial_row_indices = np.random.randint(n, size=rank)
    if initial_column_indices is None:
        _, p = data.shape
        initial_column_indices = np.random.randint(p, size=rank)
    if len(initial_row_indices) != rank or len(initial_column_indices) != rank:
        raise ValueError("The number of initial indices must be equal to the desired rank.")
    optimal_row_indices, optimal_column_indices = mc.alternating_maxvol(data, initial_row_indices, initial_column_indices)
    c, u, r = mc.cur_decomposition(data, optimal_row_indices, optimal_column_indices)

    y = data[:,1:] # All but first column

    # Analogous to Koopman operator
    pinv_r = LA.pinv(r)
    a_tilde = LA.pinv(c) @ y @ pinv_r @ u
    dmd_eig_values, w = LA.eig(a_tilde)
    dmd_modes = y @ pinv_r * u @ w

    return dmd_eig_values, dmd_modes

def svd_dmd(data: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs the simgular value decomposition (SVD) based dynamic mode decomposition on n x p time series data.
    The case when rank = p corresponds to the exact DMD.

    Parameters:
        data (np.ndarray): Time series data where each column is a time snapshot.
        rank (int): Number of singular values for SVD decomposition

    Returns:
        DMD eigenvalues and DMD modes of the dataset.

    Example:
        dmd_eig_values, dmd_modes = dmd(np.random.rand(100, 10), 5)
    """
    if rank > min(data.shape):
        raise ValueError("The rank must be smaller than each dimension of the data.")

    # data matrices
    x = data[:,:-1] # All but last column
    y = data[:,1:] # All but first column

    # fixed rank SVD is faster than computing full SVD (Halko, et al. (2009))
    u, sigma, vh = skmath.randomized_svd(x, n_components = rank, random_state = None)

    inv_sigma = 1/sigma
    # Koopman operator
    a_tilde = u.conj().T @ y @ vh.conj().T * inv_sigma

    dmd_eig_values, w = LA.eig(a_tilde)
    # The DMD modes corresponds to eigenvectors of timestep operator a such that ax = y
    dmd_modes = y @ vh.conj().T * inv_sigma @ w

    return dmd_eig_values, dmd_modes

def forecast(data: np.ndarray, rank: int, num_forecasts: int, cur_or_svd: str = 'cur'):
    """Forecast timeseries data using the dynamic mode decomposition of given rank."""
    # TODO: Verify that dmd modes should not be scaled by eigenvalues
    # TODO: refactor multiplying by eigenvalues so it does not multiply by a diagonal matrix.
    if cur_or_svd == 'cur':
        eig_values, dmd_modes = cur_dmd(data, rank)
    elif cur_or_svd == 'svd':
        eig_values, dmd_modes = svd_dmd(data, rank)
    else:
        raise ValueError(f"cur_or_svd must be 'cur' or 'svd'. Instead got {cur_or_svd}.")
    diag_eig_values = np.diag(eig_values)

    initial_condition = data[:,-1] # uses last vector in data for initial condition
    data_dimension = initial_condition.shape[0]
    forecast_results = np.zeros((data_dimension, num_forecasts), dtype=np.complex_)

    # time stepping
    low_dim_forecast = LA.lstsq(dmd_modes, initial_condition)[0]
    for t in range(num_forecasts):
        low_dim_forecast = diag_eig_values @ low_dim_forecast
        forecast_results[:,t] =  dmd_modes @ low_dim_forecast

    forecast_results = forecast_results.real # take real component of results
    return forecast_results
